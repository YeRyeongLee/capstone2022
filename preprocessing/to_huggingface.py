from datasets import load_dataset


path = '../data/'
data_files = {
    'train': path + 'train.csv',
    'valid': path + 'valid.csv',
    'test': path + 'test.csv',
}

dataset = load_dataset("csv", data_files=data_files)

dataset.push_to_hub("YeRyeongLee/reddit_mental_health", private=True)