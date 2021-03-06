# coding: utf-8
import mysql.connector
import pyarrow.parquet as pq
import pandas as pd
import numpy as np 
import pickle
import sys
import os 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from konlpy.tag import Mecab

os.environ["CUDA_VISIBLE_DEVICES"]=''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def set_pandas_display_options() -> None:
    display = pd.options.display
    display.max_columns = 100
    display.max_rows = 100
    display.max_colwidth = 199
    display.width = None

def timer(fn):
    from time import perf_counter

    def inner(*args, **kwargs):
        start_time = perf_counter()
        to_execute = fn(*args, **kwargs)
        end_time = perf_counter()
        execution_time = end_time - start_time
        print('{0} took {1:.8f}s to execute'.format(fn.__name__, execution_time))
        return to_execute

    return inner
@timer
def mysql_data_loader():
    conn = mysql.connector.connect(host='172.31.1.210', port='8883', database='hogangnono', user='zigbang_data', password='zigbang09!', charset='utf8')
    cur = conn.cursor()
    cur.execute("SELECT * FROM hogangnono.apt_review where date > DATE_ADD(now(), INTERVAL -1 HOUR)")
    data_review = pd.DataFrame(cur.fetchall())
    data_review.columns = cur.column_names
    cur.execute("SELECT * FROM hogangnono.abuse_process_log where created_at > DATE_ADD(now(), INTERVAL -1 HOUR)")
    data_log = pd.DataFrame(cur.fetchall())
    data_log.columns = cur.column_names
    return data_review, data_log

@timer
def s3_data_loader():
    fs = s3fs.S3FileSystem()
    bucket = 'zigbang-data'
    path = 'ods/hogangnono_apt_review'
    bucket_uri = f's3://{bucket}/{path}'
    data_review = pq.ParquetDataset(bucket_uri, filesystem=fs)
    path = 'ods/hogangnono_abuse_process_log'
    bucket_uri = f's3://{bucket}/{path}'
    data_log = pq.ParquetDataset(bucket_uri, filesystem=fs)

    return data_review, data_log

@timer
def pq2df_transformer(data):
    table = data.read()
    df = table.to_pandas()
    return df

@timer
def preprocessing(data_review, data_log):
    # ???????????? ????????? ??????
    data_log = data_log[(data_log["type"] == 1) & (data_log["comment_id"].isnull())]  # Type == 1 : ????????? ??????
    data_log = data_log[~((data_log["reason"].str.contains("??????")) | (data_log["reason"].str.contains("??????")))]
    data_log = data_log[["review_id", "reason"]].dropna().drop_duplicates("review_id")

    # ??????
    data_review = data_review[["id", "content", "is_blocked", "is_deleted", "is_blinded"]]
    data = pd.merge(data_review, data_log.rename(columns={"review_id": "id"}), on="id", how="left")
    print(data)
    # ?????????
    data = data[~((data["is_deleted"] == 1) & (data["is_blocked"] != 1) & (data["is_blinded"] != 1))]  # ??????????????? ?????????
    data = data[~data["content"].isnull()]  # ????????? ???????????? ?????? ?????????
    data = data[data["content"].str.len() > 20]  # ?????? 20??? ?????? ??????
    print(data)
    # ?????? ??????: is_blocked = 1?????????, ??????????????? ????????? ?????????
    data["is_spam"] = np.where((data["is_blocked"] == 1) | (~data["reason"].isnull()), 1, 0)
    print(data)
    # ??? ?????????, 1??? ?????? ?????? ????????? ????????? is_spam?????? 1, ???????????? 0?????? ?????? (?????? ???????????? ??????)
    data = data.groupby("content").sum()[["is_spam"]].reset_index()
    data["is_spam"] = np.where(data["is_spam"] > 0, 1, 0)
    print(data)
    return data

@timer    
def split_morphs(data):
    tagger = Mecab()
    morphs = list()
    for i in range(len(data)):
        corpus = data["content"].iloc[i]
        try:
            keywords = tagger.morphs(corpus.upper())
            morphs.append([keywords]) 
        except KeyboardInterrupt:
            break 
        except:
            morphs.append([])

    return morphs

@timer
def join(morphs, data):
    data = pd.concat([pd.DataFrame(morphs)[0], data["is_spam"].reset_index(drop=True)], axis=1).rename(
        columns={0: "content"}).dropna()
    data["content"] = data["content"].apply(lambda row: " ".join(row))

    return data

@timer
def postprocess(data):
    data["content"] = data["content"].str.replace("\u200b","")
    return data

@timer
def load_trained_model():
    model = load_model("../Spam_Classifier.h5")
    with open('../Tokenizer.pickle', 'rb') as tokenizer:
        tokenizer = pickle.load(tokenizer)
    return model, tokenizer

@timer
def inference(model, data):
    return model.predict(data)

@timer
def main(): 
    data_review, data_log = mysql_data_loader() 
    data = preprocessing(data_review, data_log)    
    morphs = split_morphs(data)
    data = join(morphs, data) 
    data = postprocess(data)   
    model, tokenizer = load_trained_model()

    sentences = tokenizer.texts_to_sequences(data['content'])
    sentences = pad_sequences(sentences, maxlen = 512)

    probs = inference(model, sentences) 
    data['spam_probability'] = [np.round(prob * 100, 2) for prob in probs.ravel()]
    
    print(data)

if __name__ == "__main__":
    main() 
 