import sqlite3
import json
from datetime import datetime
from xmlrpc.client import TRANSPORT_ERROR


DATABASE_PATH = './data/train.db'
TRAIN_DATASET_PATH = './data/train.json'

sql_transaction = []

try:
    connection = sqlite3.connect(DATABASE_PATH)
    print("Connected to database.")
    c = connection.cursor()
except Exception as e:
    print(str(e))
    exit()

def create_table():
    c.execute("CREATE TABLE IF NOT EXISTS parent_reply(id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, question TEXT, answer TEXT);")

def print_sample(file, index):
    with open(file, 'r') as f:
        data = json.load(f)
        print(data[index])

def transaction_bldr(sql):
    global sql_transaction
    sql_transaction.append(sql)
    if len(sql_transaction) > 1000:
        c.execute('BEGIN TRANSACTION;')
        for s in sql_transaction:
            try:
                c.execute(s)
                print('Executed transaction.')
            except Exception as e:
                print(s)
                print(str(e))

        connection.commit()
        sql_transaction = []

def sql_insert_ques_ans_Pair(title, question, answer):
    try:
        sql = f"INSERT INTO parent_reply(title, question, answer) VALUES(\"{str(title)}\", \"{str(question)}\", \"{str(answer)}\");"
        transaction_bldr(sql)
    except Exception as e:
        print('s0 insertion', str(e))
    
if __name__ == '__main__':
    create_table()

    row_counter = 0
    # Use buffering if the file is too large to fit in-memory
    with open(TRAIN_DATASET_PATH, 'r', buffering=1000) as f:
        for chunk in f:
            chunk_as_json = json.loads(chunk)
            for entry in chunk_as_json:
                question = entry['question']
                answer = entry['nq_answer'][0]
                title = entry['nq_doc_title']
                row_counter += 1
                sql_insert_ques_ans_Pair(title, question, answer)

                if row_counter % 1000 == 0:
                    print(f"Total rows read: {row_counter}, Time: {datetime.now()}")

        c.execute('VACUUM')
        connection.commit()
        connection.close()

        
        
    
