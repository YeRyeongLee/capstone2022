
from filters import *
import pandas as pd
import numpy as np

subreddits = ['schizophrenia', 'ADHD', 'bipolar', 'depression', 'Anxiety']

PATH = '../data/'


def filtering(subreddit, year):
    
    d = pd.read_csv(PATH + "raw/" + subreddit + str(year) + '.csv', lineterminator='\n')
    data = d['title'] + " " + d['text_context']
    
    write_metadata('raw', subreddit, year, data)
    
    data = data.drop_duplicates()
    data = data.dropna()
    
    write_metadata('drop', subreddit, year, data)
    
    filtered_texts = []
    
    for text in data:
      if is_filtering(text, subreddit):
        filtered_texts.append(text)
    
    data = pd.Series(filtered_texts)
    
    write_metadata('drop and filter', subreddit, year, data)
    
    data.to_csv(PATH + "filtered/" + "filtered_" + subreddit + str(year) + '.csv')
    
    print("="*5, subreddit, year, '='*5)


def is_filtering(text, subreddit): # 필터링 단어가 존재함
    for subtext in filter_dict[subreddit]:
        if subtext in text.lower():
            return True
    return False

    
def write_metadata(feature, subreddit, year, data):
    with open('../data/_metadata.txt', 'a') as f:
        f.write(f"Feature: \t{feature}\n")
        f.write(f"Class: \t\t{subreddit}\n")
        f.write(f"Year: \t\t{year}\n")
        f.write(f"{data.describe()}\n\n\n")


if __name__ == '__main__':
    for subreddit in subreddits:
        for year in range(2012, 2023):
            filtering(subreddit, year)