from sklearn.model_selection import train_test_split
import pandas as pd


train_val = pd.read_csv('../data/train_val.csv')

train, val = train_test_split(train_val, test_size=0.1, random_state=1234, stratify=train_val['label'])

train.to_csv('../data/train.csv', index=False)
val.to_csv('../data/valid.csv', index=False)