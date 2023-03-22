import pandas as pd

PATH = '../data/filtered/'

subreddits = ['schizophrenia', 'ADHD', 'bipolar', 'depression', 'Anxiety']


train_val = pd.DataFrame(columns=['text', 'label'])

for subreddit in subreddits:
    for year in range(2012, 2022):
        data = pd.read_csv(PATH + "filtered_" + subreddit + str(year) + '.csv')
        data['text'] = data['0']
        data['label'] = subreddit
        
        train_val = pd.concat([train_val, data[['text', 'label']]], ignore_index=True)

train_val.to_csv("../data/train_val.csv")



test = pd.DataFrame(columns=['text', 'label'])

for subreddit in subreddits:
    year = 2022
    data = pd.read_csv(PATH + "filtered_" + subreddit + str(year) + '.csv')
    data['text'] = data['0']
    data['label'] = subreddit
    
    test = pd.concat([test, data[['text', 'label']]], ignore_index=True)

test.to_csv('../data/test.csv')