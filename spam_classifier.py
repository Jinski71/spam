import pickle
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from konlpy.tag import Mecab

if __name__ == "__main__":
    try: 
        sentence = sys.argv[1].replace("\u200b","")
        
        # 형태소 분리
        tagger = Mecab()
        sentence = " ".join(tagger.morphs(sentence))
        
        # 모형 로딩
        model = load_model("Spam_Classifier.h5")
        with open('Tokenizer.pickle', 'rb') as tokenizer:
            tokenizer = pickle.load(tokenizer)
        
        # 토큰화
        sentence = tokenizer.texts_to_sequences([sentence])
        sentence = pad_sequences(sentence, maxlen = 512)
        
        # 스팸 확률 예측
        prob = round(model.predict(sentence)[0][0] * 100, 2)
        print(f"Spam Probability: {prob}%")
        
    except IndexError:
        print("An Error Occurred")