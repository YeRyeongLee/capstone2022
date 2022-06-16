import psaw
from psaw import PushshiftAPI
import pandas as pd
import datetime as dt
import time

api = PushshiftAPI()

def subreddit_year(subreddit, year):
  path = "./data/"
  
  start = int(dt.datetime(year, 1, 1).timestamp())
  end = int(dt.datetime(year+1, 1, 1).timestamp())

  temp = list(api.search_submissions(after=start,
                                     before=end,
                                     subreddit=subreddit,
                                     filter=['url','author', 'title', 'subreddit', 'selftext', 'created_by']))
  
  print(subreddit, year, len(temp))
  
  a_list = [] # author
  t_list = [] # title
  s_list = [] # selftext(text context)
  times = []

  for i in range(len(temp)):
    if temp[i][2] != '[removed]':
      a_list.append(temp[i][0])
      times.append( dt.datetime.fromtimestamp(temp[i][1]).strftime('%Y-%m-%d %H:%M:%S') )
      t_list.append(temp[i][4])
      s_list.append(temp[i][2])

  table = pd.DataFrame({'subreddit': subreddit,
                        'author': a_list,
                        'title': t_list,
                        'text_context': s_list,
                        'date': times})
  
  table.to_csv(path+subreddit+str(year)+'.csv', index=False)

years = [i for i in range(2012, 2023)]
subreddits = ['schizophrenia', 'ADHD', 'bipolar', 'depression', 'Anxiety', 'SuicideWatch']

"""
for year in years:
    for subreddit in subreddits:
        subreddit_year(subreddit, year)
"""

"""
for year in range(2019, 2023):
  for subreddit in subreddits:
    subreddit_year(subreddit, year)
"""

for subreddit in subreddits:
    subreddit_year(subreddit, 2021)