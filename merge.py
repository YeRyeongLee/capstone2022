import pandas as pd


subreddits = ['schizophrenia', 'ADHD', 'bipolar', 'depression', 'Anxiety', 'SuicideWatch']
months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

for subreddit in subreddits:
    files = []
    for month in months:
        files.append(pd.read_csv(subreddit + '2013' + month + '.csv', lineterminator='\n'))

    file2013 = pd.concat(files)
    file2013.to_csv(subreddit + '2013.csv', index=False)

    