import s3fs
import pyarrow.parquet as pq
import pandas as pd

import pickle
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from konlpy.tag import Mecab

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

    dataset = pq.ParquetDataset(bucket_uri, filesystem=fs)

    return dataset

@timer
def pq2df_transformer(data):
    table = data.read()
    df = table.to_pandas()

def filter_data(data):
    filtered_data = data[["id", "content", "is_blocked", "is_deleted", "is_blinded"]]


def set_pandas_display_options() -> None:
    display = pd.options.display
    display.max_columns = 100
    display.max_rows = 100
    display.max_colwidth = 199
    display.width = None
#set_pandas_display_options()
 

data = data_loader()
df = pq2df_transformer(data)
print(df)

if __name__ == "__main__":
    try:
        sentence = sys.argv[1].replace("\u200b","")

        tagger = Mecab()
        sentence = " ".join(tagger.morphs(sentence))

        model = load_model("Spam_Classifier.h5")
        with open('Tokenizer.pickle', 'rb') as tokenizer:
            tokenizer = pickle.load(tokenizer)

        sentence = tokenizer.texts_to_sequences([sentence])
        sentence = pad_sequences(sentence, maxlen = 512)

        prob = round(model.predict(sentence)[0][0] * 100, 2)
        print(f"Spam Probability: {prob}%")

    except IndexError as err:
        print("An Error Occurred", err)

