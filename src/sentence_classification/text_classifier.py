import pickle
import string
import re

import nltk
from nltk.stem                          import WordNetLemmatizer
from nltk.corpus                        import stopwords, wordnet

from keras.preprocessing.sequence       import pad_sequences


def lemmatize_word(word, lemmatizer):
    lemma = lemmatizer.lemmatize(word, pos=wordnet.VERB)
    return lemmatizer.lemmatize(lemma, pos=wordnet.NOUN)


def process_text(text, lemmatizer, portuguese_stopword):
    text = text.lower()

    text = ''.join([char for char in text if char not in string.punctuation])
    
    words = text.split()
    filtered_and_lemmatized_words = [lemmatize_word(word, lemmatizer) for word in words if word.lower() not in portuguese_stopword]
    text = ' '.join(filtered_and_lemmatized_words)
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


class SentenceClassifier(object):
    def __init__(self):
        self.multi_label_binarizer = pickle.load(open("data/06_models/mlb.pkl", "rb"))
        self.tokenizer = pickle.load(open("data/06_models/tokenizer.pickle", "rb"))
        self.MAX_SEQUENCE_LENGTH = 512

    def data_cleaning(self, df_raw):
        # # 2.0 Data Descrition
        data_analysis = df_raw.copy()

        # # 4.0 Data Processing
        nltk.download('wordnet')
        nltk.download('stopwords')

        lemmatizer = WordNetLemmatizer()
        portuguese_stopword = set(stopwords.words('portuguese'))

        # here we remove the stop words
        data_analysis['sentence'] = data_analysis['sentence'].apply(lambda row: process_text(row, lemmatizer, portuguese_stopword))

        # # 5.0 Data Preparation
        data_preparation = data_analysis.copy()

        return data_preparation
    
    def data_preparation(self, data_preparation):
        tokenizer = self.tokenizer

        sequences_train = tokenizer.texts_to_sequences(data_preparation['sentence'])

        X_train = pad_sequences(sequences_train, maxlen=self.MAX_SEQUENCE_LENGTH)

        return X_train

    def get_prediction(self, model, X_train):
        thresholds = [0.33, 0.27, 0.14, 0.33, 0.21]

        prediction = model.predict(X_train)

        predicted_categories_list = []

        for pred in prediction:
            predicted_categories = [self.multi_label_binarizer.classes_[i] for i, p in enumerate(pred) if p >= thresholds[i]]
            predicted_categories_list.append(predicted_categories)

        return predicted_categories_list
