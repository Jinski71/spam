#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import numpy as np
import pandas as pd
from konlpy.tag import Mecab
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Dense, SimpleRNN, LSTM, GRU
import warnings
warnings.filterwarnings("ignore")

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

def join(morphs, data):
    data = pd.concat([pd.DataFrame(morphs)[0], data["is_spam"].reset_index(drop=True)], axis=1).rename(
        columns={0: "content"}).dropna()
    data["content"] = data["content"].apply(lambda row: " ".join(row))

    return data

def split_dataset(data):
    # X, y 세팅
    X = data["content"]
    y = data["is_spam"]

    # 학습-테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    return X_train, X_test, y_train, y_test

def tokenize(X_train):
    # 리뷰 문장 토큰화
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)
    X_train_encoded = tokenizer.texts_to_sequences(X_train)

    # 단어집합 총 크기
    word_to_index = tokenizer.word_index
    vocab_size = len(word_to_index) + 1

    # 훈련 데이터 패딩
    X_train_padded = pad_sequences(X_train_encoded, maxlen=512)

    return vocab_size, X_train_padded, tokenizer

def lstm_model(vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, 64))
    model.add(LSTM(32))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['acc'])

    return model

def fit(model, X_train_padded, y_train):
    model.fit(X_train_padded, y_train, epochs=2, batch_size=512, validation_split=0.25)

    return model

def evaluate(model, tokenizer, X_test, y_test):
    X_test_encoded = tokenizer.texts_to_sequences(X_test)
    X_test_padded = pad_sequences(X_test_encoded, maxlen=512)
    print(f"테스트 정확도: {model.evaluate(X_test_padded, y_test)[1]}")

def save(model, tokenizer):
    model.save("Spam_Classifier.h5")
    with open('Tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def main():
    data_review = pd.read_csv("apt_review_20220106.zip", encoding="UTF-8")
    data_log = pd.read_csv('abuse_process_log_20220106.zip', encoding="UTF-8")
    data = preprocessing(data_review, data_log)
    morphs = split_morphs(data)
    data = join(morphs, data)
    X_train, X_test, y_train, y_test = split_dataset(data)
    vocab_size, X_train_padded, tokenizer = tokenize(X_train)
    model = lstm_model(vocab_size)
    model = fit(model, X_train_padded, y_train)
    evaluate(model, tokenizer, X_test, y_test)
    save(model, tokenizer)

if __name__ == "__main__":
    main()