# coding: utf-8

import s3fs
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

import pickle
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"]='0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from konlpy.tag import Mecab

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
def data_loader():
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

@timer
def preprocessing(data_review, data_log):
    # 처리내역 데이터 가공
    data_log = data_log[(data_log["type"] == 1) & (data_log["comment_id"].isnull())]  # Type == 1 : 아파트 리뷰
    data_log = data_log[~((data_log["reason"].str.contains("무효")) | (data_log["reason"].str.contains("복원")))]
    data_log = data_log[["review_id", "reason"]].dropna().drop_duplicates("review_id")

    # 병합
    data_review = data_review[["id", "content", "is_blocked", "is_deleted", "is_blinded"]]
    data = pd.merge(data_review, data_log.rename(columns={"review_id": "id"}), on="id", how="left")

    # 필터링
    data = data[~((data["is_deleted"] == 1) & (data["is_blocked"] != 1) & (data["is_blinded"] != 1))]  # 자진삭제한 데이터
    data = data[~data["content"].isnull()]  # 내용이 기재되지 않은 데이터
    data = data[data["content"].str.len() > 20]  # 최소 20자 이상 리뷰

    # 스팸 기준: is_blocked = 1이거나, 처리사유가 기재된 데이터
    data["is_spam"] = np.where((data["is_blocked"] == 1) | (~data["reason"].isnull()), 1, 0)

    # 각 문장별, 1회 이상 스팸 처리된 문장은 is_spam에서 1, 나머지는 0으로 처리 (완전 중복문장 제거)
    data = data.groupby("content").sum()[["is_spam"]].reset_index()
    data["is_spam"] = np.where(data["is_spam"] > 0, 1, 0)

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
def main():
    data_review = pd.read_csv("../apt_review_20220106.csv", encoding="UTF-8")
    data_log = pd.read_csv('../abuse_process_log_20220106.csv', encoding="UTF-8")
    #data_review, data_log = data_loader()
    #df_review = pq2df_transformer(data_review)
    #df_log = pq2df_transformer(data_log)
    #data = preprocessing(df_review, df_log) 
    data = preprocessing(data_review, data_log) 
    print(data)

    morphs = split_morphs(data)
    data = join(morphs, data)
    data = postprocess(data) 
    print(data)
    
    model = load_model("Spam_Classifier.h5")
    with open('Tokenizer.pickle', 'rb') as tokenizer:
        tokenizer = pickle.load(tokenizer)

    sentences = tokenizer.texts_to_sequences(data['content'][:10])
    sentences = pad_sequences(sentences, maxlen = 512)

    prob = round(model.predict(sentences)[0][0] * 100, 2)
    print(f"Spam Probability: {prob}%")
 
    # morphs = split_morphs(data)
    # data = join(morphs, data)
    # X_train, X_test, y_train, y_test = split_dataset(data)
    # vocab_size, X_train_padded, tokenizer = tokenize(X_train)
    # model = lstm_model(vocab_size)
    # model = fit(model, X_train_padded, y_train)
    # evaluate(model, tokenizer, X_test, y_test)
    # save(model, tokenizer)

if __name__ == "__main__":
    main() 
 