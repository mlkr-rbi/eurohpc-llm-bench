'''
For experimenting with huggingface datasets.
'''

from datasets import load_dataset, DatasetDict, concatenate_datasets, Dataset
from sklearn.model_selection import train_test_split

def resplit_dataset(dataset_id: str):
    '''
    Split a dataset into train, validation, and test sets using stratified splitting.
    If the input is a DatasetDict, it will be concatenated into a single Dataset.
    :param init_dataset: huggingface id of the dataset
    '''
    init_dataset = load_dataset(dataset_id)
    print(init_dataset)
    if isinstance(init_dataset, dict):
        merged_dataset = concatenate_datasets([split for split in init_dataset.values()])
        # print lengths of splits, and final length
        for split in init_dataset.keys():
            print(f"{split}: {len(init_dataset[split])}")
        print(f"Total examples after merging: {len(merged_dataset)}")
        dataset = merged_dataset
    else:
        dataset = init_dataset
    # Convert the dataset to a Pandas DataFrame for stratified splitting
    df = dataset.to_pandas()
    print(df)
    # Perform stratified splitting using scikit-learn's `train_test_split`
    train_df, temp_df = train_test_split(
        df,
        test_size=0.3,  # Reserve 30% of data for dev and test sets
        stratify=df["label"],  # Stratify by the label column
        random_state=42  # Set a seed for reproducibility
    )
    dev_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,  # Split remaining data equally into dev and test sets
        stratify=temp_df["label"],
        random_state=42
    )
    # Convert the DataFrames back to Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_df)
    dev_dataset = Dataset.from_pandas(dev_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Combine splits into a new DatasetDict
    split_dataset = DatasetDict({
        "train": train_dataset,
        "validation": dev_dataset,
        "test": test_dataset,
    })

    # Verify the splits
    print(split_dataset)

    # Check class distribution in each split (optional)
    for split_name, split_data in split_dataset.items():
        print(f"{split_name} class distribution:")
        print(split_data.to_pandas()["label"].value_counts(normalize=True))

def nllb_dset():
    # ("eng_Latn", "hrv_Latn"),
    dataset = load_dataset("allenai/nllb", "eng_Latn-hrv_Latn")
    print(dataset)

def bactrianx_dset():
    dataset = load_dataset("MBZUAI/Bactrian-X", "hr")
    print(dataset)
    print(dataset["train"][0])

if __name__ == "__main__":
    #resplit_dataset("ag_news")
    bactrianx_dset()