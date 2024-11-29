from datasets import load_dataset

def biblenelp_corpus():
    ''' Parallel en-hr sentences from the Bible '''
    dataset = load_dataset('davidstap/biblenlp-corpus-mmteb', 'eng-hrv')
    print(dataset)

def ccmatrix_corpus():
    '''
    Parallel en-hr sentences from the CCMatrix corpus:
    https://huggingface.co/datasets/sentence-transformers/parallel-sentences-ccmatrix
    '''
    dataset = load_dataset('sentence-transformers/parallel-sentences-ccmatrix', 'en-hr')
    print(dataset)

def elrc_medical_v2():
    dataset = load_dataset('qanastek/ELRC-Medical-V2', 'en-hr')
    print(dataset)

if __name__ == "__main__":
    biblenelp_corpus()
    #ccmatrix_corpus()
    #elrc_medical_v2()
