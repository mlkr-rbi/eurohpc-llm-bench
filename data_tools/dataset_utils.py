from typing import Union

from transformers import AutoTokenizer, PreTrainedTokenizerBase
from huggingface_hub import login

from data_tools.prompt_tools import TranslationPromptComposer
from settings import HUGGINGFACE_TOKEN


class HFTokenCounter:
    '''
    Counts tokens produced by a huggingface tokenizer
    '''

    def __init__(self, tokenizer: Union[str, PreTrainedTokenizerBase]):
        '''
        param tokenizer: either a huggingface tokenizer, or a string that is a name of a huggingface tokenizer
        '''
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer

    def __call__(self, s: str) -> int:
        return len(self.tokenizer(s, padding=False, truncation=False)['input_ids'])

from datasets import Dataset, load_dataset, DatasetDict
import numpy as np

def split_hf_dataset(dataset: Dataset, train_ratio=0.7, test_ratio=0.15, val_ratio=0.15, rseed=7812) -> DatasetDict:
    '''
    Split a huffingface dataset into train, test, and validation parts.
    '''
    if train_ratio + test_ratio + val_ratio != 1.0:
        raise ValueError("The sum of train, test, and dev ratios must be 1.0")
    # Calculate sizes
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    test_size = int(total_size * test_ratio)
    # Shuffle and split the dataset
    dataset = dataset.shuffle(seed=rseed)
    train_dataset = dataset.select(range(train_size))
    test_dataset = dataset.select(range(train_size, train_size + test_size))
    val_dataset = dataset.select(range(train_size + test_size, total_size))
    # Create dataset dictionary
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset,
        'validation': val_dataset
    })
    return dataset_dict

def split_tst():
    sample_dataset = load_dataset('imdb', split='train')
    train_ratio = 0.7
    test_ratio = 0.15
    dev_ratio = 0.15
    split_datasets = split_hf_dataset(sample_dataset, train_ratio, test_ratio, dev_ratio)
    print({key: len(value) for key, value in split_datasets.items()})

def token_counter_tst():
    login(HUGGINGFACE_TOKEN)
    counter = HFTokenCounter('google/gemma-2-2b')
    print(counter('hello world'))
    print(counter('o what a tangled web we weave when first we practice to deceive'))

def pairs_to_instructions(dataset, pc: TranslationPromptComposer, rnd_seed=1337):
    '''
    Converts a dataset of pairs of strings into a dataset of translation instructions.
    '''
    texts = []
    for i in range(len(dataset)):
        hr = dataset[i]['hr']
        en = dataset[i]['en']
        texts.append(pc.train_prompt(hr, en, 'hr'))
        texts.append(pc.train_prompt(en, hr, 'en'))
    # convert texts to hf dataset
    result = Dataset.from_dict({'text': texts})
    result = result.shuffle(seed=rnd_seed)
    return result

def print_dset_sample(dataset: Dataset, title:str = 'dataset', sample_size=10, rnd_seed=9123):
    '''
    Prints a sample of a dataset.
    '''
    print(f'{title}:')
    for i in np.random.RandomState(rnd_seed).choice(len(dataset), sample_size, replace=False):
        print(dataset[int(i)])

if __name__ == '__main__':
    #token_counter_tst()
    split_tst()