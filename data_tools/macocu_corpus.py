'''
Utils for loading, manipulating and analyzing MaCoCu parallel sentences corpus:
https://www.clarin.si/repository/xmlui/handle/11356/1814
'''

import csv
import random
from typing import Callable
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset

from data_tools.dataset_utils import HFTokenCounter, split_hf_dataset, pairs_to_instructions, print_dset_sample
from data_tools.prompt_tools import TranslationPromptComposer, hr_en_translate_prompt

# from settings import MACOCU_SENTENCES_FILE
from utils.config_utils import get_macocu_sentences_file_path
from utils.config_utils import get_macocu_pairs_dataset_dir
from utils.config_utils import get_macocu_text_dataset_dir

def macocu_sentence_load(file_path: str, print_columns=False) -> DataFrame:
    '''
    Loader of MaCoCu parallel en-hr sentences corpus from .txt file to a pandas DataFrame.
    '''
    df = pd.read_csv(file_path, header=0, sep="\t", encoding='utf-8', quoting= csv.QUOTE_NONE)
    if print_columns: print(df.dtypes) # Print column names and their types
    return df

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
    dataset.save_to_disk(get_macocu_pairs_dataset_dir())
    if create_text_dset:
        # dataset for direct training, replace pairs with single-string instruction prompts, and shuffle
        # do not touch 'test' split since it can be used for true translation evaluation
        dataset['train'] = pairs_to_instructions(dataset['train'], prompt_composer)
        dataset['validation'] = pairs_to_instructions(dataset['validation'], prompt_composer)
        if print_dsets:
            print_dset_sample(dataset['train'], 'train', 20)
            print()
            print_dset_sample(dataset['validation'], 'validation', 20)
        dataset.save_to_disk(get_macocu_text_dataset_dir())

if __name__ == "__main__":
    #macocu_sentence_load('/data/datasets/corpora/classla/parallel/hr-en-macocu-cc/MaCoCu-hr-en.sent.10000.txt', True)
    #macocu_sentence_analyze()
    #create_macocu_train_dset_v1(MACOCU_SENTENCES_FILE, size=50, print_dsets=True) # test version
    create_macocu_final_dset_v1(get_macocu_sentences_file_path(), size=100000,
                                train_ratio=0.7, test_ratio=0.15, val_ratio=0.15,
                                print_dsets=True, create_text_dset=True)