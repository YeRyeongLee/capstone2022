import json
from pprint import pprint
import pandas as pd
import numpy as np


subreddits = ['schizophrenia', 'ADHD', 'bipolar', 'depression', 'Anxiety', 'SuicideWatch']


def from_archive(subreddits):
    with open('RS_2013-12') as f:
        docs = {}
        for subreddit in subreddits:
            docs[subreddit] = []
        for line in f:
            data = json.loads(line)
            if data['subreddit'] in subreddits:
                docs[data['subreddit']].append(data)
    
    return docs

docs = from_archive(subreddits)


def make_csv(subreddit):
    reddit_lst = docs[subreddit]

    a_list = [] # author
    t_list = [] # title
    s_list = [] # selftext(text context)

    for data in reddit_lst:
        if data['selftext'] != "":
            a_list.append(data['author'])
            t_list.append(data['title'])
            s_list.append(data['selftext'])

    table = pd.DataFrame({'subreddit': subreddit,
                        'author': a_list,
                        'title': t_list,
                        'text_context': s_list})
    
    table.to_csv(subreddit+str(201312)+'.csv', index=False)

for subreddit in subreddits:
    make_csv(subreddit)