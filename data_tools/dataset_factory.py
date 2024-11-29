'''
Metods for creating production (training and/or testing ready) datasets.
'''
from datasets import load_dataset, Dataset, DatasetDict

from settings import MACOCU_DATASET_V1


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

def get_macocu_v1():
    '''
    Load MaCoCu v1 dataset from disk.
    '''
    dset = DatasetDict.load_from_disk(MACOCU_DATASET_V1)
    print('MaCoCu v1 dataset loaded:')
    print(dset)
    print()
    return dset

if __name__ == '__main__':
    get_macocu_v1()