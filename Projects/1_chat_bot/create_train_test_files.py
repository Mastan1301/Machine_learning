import sqlite3
import pandas as pd

DATABASE_PATH = './data/train.db'
connection = sqlite3.connect(DATABASE_PATH)
c = connection.cursor()
limit = 1000
curr_length = limit
counter = 0
test_done = False
last_id = 0

while curr_length == limit:
    df = pd.read_sql(f'SELECT * FROM parent_reply WHERE id >= {last_id} AND question NOT NULL LIMIT {limit};', connection)
    last_id = df.tail(1)['id'].values[0]
    curr_length = len(df)

    if not test_done:
        with open('./data/test.from', 'a', encoding='utf8') as f:
            for content in df['question'].values:
                f.write(content+'\n')
        with open('./data/test.to', 'a', encoding='utf8') as f:
            for content in df['answer'].values:
                f.write(str(content)+'\n')
        
        test_done = True

    else:
        with open('./data/train.from', 'a', encoding='utf8') as f:
            for content in df['question'].values:
                f.write(content+'\n')
        with open('./data/train.to', 'a', encoding='utf8') as f:
            for content in df['answer'].values:
                f.write(str(content)+'\n') 

    counter += 1
    if counter % 20 == 0:
        print(counter * limit, 'rows completed so far')
     