'''
Utils for loading, manipulating and analyzing MaCoCu parallel sentences corpus:
https://www.clarin.si/repository/xmlui/handle/11356/1814
'''

import csv
import random
from typing import Callable, Iterable, Union
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset

from data_tools.dataset_utils import HFTokenCounter, split_hf_dataset, pairs_to_instructions, print_dset_sample
from data_tools.prompt_tools import TranslationPromptComposer, hr_en_translate_prompt, get_prompt
from settings import MACOCU_SENTENCES_FILE_SMALL, MACOCU_SENTENCES_FILE, MACOCU_SENT_ORIG_HF

# from settings import MACOCU_SENTENCES_FILE
from utils import config_utils

def macocu_sentence_load(file_path: str, verbose=False) -> DataFrame:
    '''
    Loader of MaCoCu parallel en-hr sentences corpus from .txt file to a pandas DataFrame.
    '''
    df = pd.read_csv(file_path, header=0, sep="\t", encoding='utf-8', quoting= csv.QUOTE_NONE)
    if verbose:
        print('Dataset size', len(df))
        print(df.dtypes) # Print column names and their types
    return df

def macocu_sentence_to_huggingface():
    #df = macocu_sentence_load(MACOCU_SENTENCES_FILE_SMALL, verbose=True)
    df = macocu_sentence_load(MACOCU_SENTENCES_FILE, verbose=True)
    hf_ds = Dataset.from_pandas(df)
    hf_ds.save_to_disk(MACOCU_SENT_ORIG_HF)
    print(hf_ds)

def macocu_analyze_score(df: DataFrame, score='bicleaner_ai_score'):
    scores = df[score]
    # 5-number summary
    summary = scores.describe()
    print(f"5-Number Summary of {score}:")
    print(summary)
    # Plot the histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(scores.dropna(), bins=30, color='blue', kde=True)
    plt.title(f'Histogram of {score}')
    plt.xlabel(score)
    plt.ylabel('Frequency')
    plt.show()

def macocu_select_top_by_score(df: DataFrame, score='bicleaner_ai_score', top_n=1000):
    # select top-n rows by score
    sorted_df = df.sort_values(by=score, ascending=False)
    top_n_rows = sorted_df.head(top_n)
    # print lowest score in top-n, along with 5-num summary
    lowest_top_n_score = top_n_rows[score].min()
    summary = df[score].describe()
    print(f"5-Number Summary of {score}:")
    print(summary)
    print(f"The lowest score of the top-{top_n} sentences is: {lowest_top_n_score}")

def macocu_sentence_analyze():
    df = macocu_sentence_load('/data/datasets/corpora/classla/parallel/hr-en-macocu-cc/MaCoCu-hr-en.sent.1000.txt')
    macocu_analyze_score(df)
    #macocu_select_top_by_score(df, score='bicleaner_ai_score', top_n=10000)

def macocu_dataset_creator(df: DataFrame, pc: TranslationPromptComposer,
                           token_counter: Callable[[str], int], max_tokens: int, size: int) -> Dataset:
    '''
    Filter MaCoCu parallel sentences by quality score and by number of tokens, and create huggingface Dataset.
    Take, from all the sentence pairs, those that produce prompts with less than max_tokens tokens,
    and select the #size of them, ordered by a quality score.
    :param df: DataFrame containing MaCoCu parallel sentences
    :param pc: TranslationPromptComposer object
    :param token_counter: function that counts number of tokens in a string
    :param max_tokens: maximum number of tokens in a sentence
    :param size: number of sentences to select
    '''
    # sort df by harmonic mean of bicleaner_ai_score and bleualign_score
    df['sort_score'] = ((2 * df['bicleaner_ai_score'] * df['bleualign_score']) /
                   (df['bicleaner_ai_score'] + df['bleualign_score']))
    df = df.sort_values(by='sort_score', ascending=False)
    result = {'hr': [], 'en': []}
    for i, row in df.iterrows():
        txt_hr = row['src_text']
        txt_en = row['trg_text']
        hr2en = pc.train_prompt(txt_hr, txt_en, 'hr')
        en2hr = pc.train_prompt(txt_en, txt_hr,'en')
        if token_counter(hr2en) > max_tokens or token_counter(en2hr) > max_tokens:
            continue
        result['hr'].append(txt_hr)
        result['en'].append(txt_en)
        if len(result['hr']) == size: break
    # create huggingface Dataset from result
    return Dataset.from_dict(result)

def analyze_numbers(numbers: Union[Series, Iterable[int]], label: str = 'score'):
    '''
    Print 5-number summary and plot histogram of a sequence of numbers.
    :param numbers:
    :param label: string label to use in the printout and the plot
    :return:
    '''
    if not isinstance(numbers, Series): numbers = pd.Series(numbers)
    summary = numbers.describe()
    print(f"5-Number Summary of {label}:")
    print(summary)
    # Plot the histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(numbers.dropna(), bins=30, color='blue', kde=True)
    plt.title(f'Histogram of {label}')
    plt.xlabel(label)
    plt.ylabel('Frequency')
    plt.savefig(f'{label}_hist.png')

def macocu_token_length_analyser(df: DataFrame, pc: TranslationPromptComposer,
                           token_counter: Callable[[str], int], subsample: int = None, rnd_seed: int = 7451):
    if subsample and subsample < len(df):
        df = df.sample(n=subsample, random_state=rnd_seed)
        print('Subsampled data.')
    prompt_lens = []
    raw_lens = []
    cnt = 0
    for _, row in df.iterrows():
        cnt += 1
        txt_hr = row['src_text']
        txt_en = row['trg_text']
        hr2en = pc.train_prompt(txt_hr, txt_en, 'hr')
        en2hr = pc.train_prompt(txt_en, txt_hr,'en')
        prmpt_len = (token_counter(hr2en) + token_counter(en2hr)) / 2
        raw_len = (token_counter(txt_hr) + token_counter(txt_en))
        prompt_lens.append(prmpt_len)
        raw_lens.append(raw_len)
        if cnt % 1000 == 0: print(f"Processed {cnt} sentences.")
    analyze_numbers(prompt_lens, 'prompt_length')
    analyze_numbers(raw_lens, 'raw_length')

def run_macocu_length_analysis():
    #df = macocu_sentence_load(MACOCU_SENTENCES_FILE_SMALL)
    df = macocu_sentence_load(MACOCU_SENTENCES_FILE, verbose=True)
    print('Data loaded.')
    prompt_composer = get_prompt("mt-en-hr-003-it", "en",
                                 randomize_prompts=True, instruction_tune=True)
    counter = HFTokenCounter('google/gemma-2-2b-it')
    macocu_token_length_analyser(df, prompt_composer, counter, subsample=100000)

def create_macocu_final_dset_v1(fpath, size, train_ratio=0.8, test_ratio=0.1,
                                val_ratio=0.1, max_tokens=512, print_dsets=False, create_text_dset=False):
    '''
    Create a huggingface Dataset from MaCoCu parallel sentences, filtered by quality score and by number of tokens.
    '''
    df = macocu_sentence_load(fpath)
    prompt_composer = hr_en_translate_prompt()
    counter = HFTokenCounter('google/gemma-2-2b')
    dataset = macocu_dataset_creator(df, prompt_composer, counter, max_tokens=max_tokens, size=size)
    dataset = split_hf_dataset(dataset, train_ratio, test_ratio, val_ratio)
    if print_dsets:
        for split, split_dset in dataset.items():
            print_dset_sample(split_dset, split, 10)
            print('\n')
    dataset.save_to_disk(str(config_utils.get_macocu_pairs_dataset_dir()))
    if create_text_dset:
        # dataset for direct training, replace pairs with single-string instruction prompts, and shuffle
        # do not touch 'test' split since it can be used for true translation evaluation
        dataset['train'] = pairs_to_instructions(dataset['train'], prompt_composer)
        dataset['validation'] = pairs_to_instructions(dataset['validation'], prompt_composer)
        if print_dsets:
            print_dset_sample(dataset['train'], 'train', 20)
            print()
            print_dset_sample(dataset['validation'], 'validation', 20)
        dataset.save_to_disk(str(config_utils.get_macocu_text_dataset_dir()))

def create_macocu_dset_v1(size: int=100000):
    """Create macocu HF dataset."""
    create_macocu_final_dset_v1(config_utils.get_macocu_sentences_file_path(), size=size,
                                train_ratio=0.7, test_ratio=0.15, val_ratio=0.15,
                                print_dsets=True, create_text_dset=True)


if __name__ == "__main__":
    #macocu_sentence_load('/data/datasets/corpora/classla/parallel/hr-en-macocu-cc/MaCoCu-hr-en.sent.10000.txt', True)
    #macocu_sentence_analyze()
    #create_macocu_train_dset_v1(MACOCU_SENTENCES_FILE, size=50, print_dsets=True) # test version
    #create_macocu_dset_v1()
    #run_macocu_length_analysis()
    macocu_sentence_to_huggingface()