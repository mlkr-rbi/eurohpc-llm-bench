'''
Metods for creating production (training and/or testing ready) datasets.
'''
from datasets import load_dataset, Dataset, DatasetDict

from utils import config_utils
from data_tools import macocu_corpus

def get_bertic_dataset(subsample=5000, rseed=5439):
    '''
    Full dataset for training the BERTic model, from HF.
    Huge, can break during the download - use local copy if possible.
    '''
    dset = load_dataset('classla/xlm-r-bertic-data', trust_remote_code=True)
    dset = dset['train'] # sample efficiently!
    if subsample and subsample < len(dset):
        dset = dset.shuffle(seed=rseed)
        dset = dset.select(range(subsample))
    dset = dset['output']
    # clean whitespaces, make new hf dataset
    dset = Dataset.from_list([{ 'text': txt.strip() } for txt in dset])
    return dset


def get_test_cro_dataset(subsample=5000, rseed=5439):
    '''
    Small cro dataset from HF for testing purposes.
    '''
    dset = load_dataset('saillab/alpaca-croatian-cleaned')
    dset = dset['train']
    if subsample and subsample < len(dset):
        dset = dset.shuffle(seed=rseed)
        dset = dset.select(range(subsample))
    dset = dset['output']
    # clean whitespaces, make new hf dataset
    dset = Dataset.from_list([{ 'text': txt.strip() } for txt in dset])
    return dset

@config_utils.raise_deprecated_warning_decorator
def get_macocu_v1():
    '''
    Load MaCoCu v1 dataset from disk.
    '''
    dset = DatasetDict.load_from_disk(config_utils.get_macocu_dataset_dir())
    print('MaCoCu v1 dataset loaded:')
    print(dset)
    print()
    return dset

def get_macocu_pairs_v1():
    '''Load MaCoCu pairs v1 dataset from disk.'''
    if not config_utils.get_macocu_pairs_dataset_dir().exists():
        macocu_corpus.create_macocu_dset_v1()
    dset = DatasetDict.load_from_disk(config_utils.get_macocu_pairs_dataset_dir())
    print('MaCoCu pairs v1 dataset loaded:')
    print(dset)
    print()
    return dset

def get_macocu_text_v1():
    '''Load MaCoCu text v1 dataset from disk.'''
    if not config_utils.get_macocu_text_dataset_dir().exists():
        macocu_corpus.create_macocu_dset_v1()
    dset = DatasetDict.load_from_disk(config_utils.get_macocu_text_dataset_dir())
    
    print('MaCoCu text v1 dataset loaded:')
    print(dset)
    print()
    return dset

DATASETS = {
    'macocu': get_macocu_v1,
    'macocu-pairs': get_macocu_pairs_v1,
    'macocu-texts': get_macocu_text_v1,
    'bertic': get_bertic_dataset,
    # 'alpaca': get_test_cro_dataset,
}

def get_original_dataset(dataset_name: str="macocu") -> DatasetDict:
    '''Load dataset by dataset name.'''
    if dataset_name.lower() in DATASETS:
        return DATASETS[dataset_name.lower()]()
    else:
        raise NotImplementedError(f"The given dataset ({dataset_name}) is not implemented jet. " +
                                  f"The only implemented datasets are: ({list(DATASETS.keys())})")


if __name__ == '__main__':
    get_macocu_v1()