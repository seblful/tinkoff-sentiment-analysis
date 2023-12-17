import re
import requests
import json
import random
import time

import pandas as pd
from tqdm import tqdm

import nltk
from nltk.corpus import stopwords
import spacy
import demoji

from autocorrect import Speller
from deep_translator import GoogleTranslator
from deep_translator.exceptions import RequestError

from fake_headers import Headers

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F


class UtilsClass:
    def __init__(self):
        self.__stop_words = None
        self.translator_url = "http://127.0.0.1:5000/translate"

        self.load_nlp()

    @property
    def stop_words(self):
        if self.__stop_words is None:
            not_stop_words = ["без", "более", "да", "другой", "лучше", "много", "можно",
                              "надо", "не", "нельзя", "нет", "ни", "никогда", "ничего", "хорошо"]
            self.__stop_words = [word for word in stopwords.words(
                'russian') if word not in not_stop_words]
        return self.__stop_words

    def load_nlp(self):
        # !python -m spacy download ru_core_news_sm
        self.nlp = spacy.load("ru_core_news_sm")
        nltk.download('stopwords')
        nltk.download('wordnet')

    def clean_text(self, text):
        text = text.lower()
        text = re.sub("\{.*?\}+", " ", text)
        text = re.sub("#\w+", " ", text)
        text = re.sub("https?://\S+|www\.\S+", " ", text)
        text = demoji.replace(text, ' ')
        text = Speller(lang='ru')(text)
        text = re.sub(r"[^a-zA-ZА-Яа-я]", " ", text)

        text = " ".join(word for word in text.split()
                        if word not in self.stop_words)

        text = [token.lemma_ for token in self.nlp(text)]
        text = ' '.join(text)

        return text

    def translate_text(self, text):

        if len(text.strip()) == 0:
            return text

        payload = {
            "q": text,
            "source": "ru",
            "target": "en",
            "format": "text",
            "api_key": ""
        }

        headers = {
            "Content-Type": "application/json"
        }

        response = requests.post(
            self.translator_url, json=payload, headers=headers)

        translated = response.json()['translatedText']

        return translated

    def classify_text(self, text, model, tokenizer):
        # Decreate length of text to model maximum
        text = text[:500]

        encoded_input = tokenizer(text, return_tensors='pt')

        # Get the logits
        output = model(**encoded_input)
        logits = output.logits
        probabilities = F.softmax(logits, dim=1)

        # Access the id2label mapping
        predicted_class_id = logits.argmax().item()
        predicted_class = model.config.id2label[predicted_class_id]
        predicted_probability = probabilities.squeeze()[
            predicted_class_id].item()

        return pd.Series((predicted_class, predicted_probability))

    def load_model(self, model_name):
        # Load the model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        return model, tokenizer


class TrainDataCreator:
    def __init__(self,
                 csv_filename,
                 sample_size,
                 model_name
                 ):
        self.df = pd.read_csv(csv_filename)
        self.train_df = self.df.sample(sample_size)
        self.sample_size = sample_size

        self.util = UtilsClass()

        self.model, self.tokenizer = self.util.load_model(
            model_name=model_name)
        self.model_id2label = self.model.config.id2label

    def process(self):
        tqdm.pandas(desc='Cleaning text')
        self.train_df['clean_text'] = self.train_df.loc[:,
                                                        'text'].progress_apply(self.util.clean_text)

        tqdm.pandas(desc='Translating text')
        self.train_df['eng_text'] = self.train_df.loc[:,
                                                      'clean_text'].progress_apply(self.util.translate_text)
        # self.translate_text_df(name_of_column='clean_text')

        tqdm.pandas(desc='Classifying text')
        self.train_df[['predicted_label', 'predicted_score']] = self.train_df.loc[:,
                                                                                  'eng_text'].progress_apply(self.util.classify_text, args=[self.model, self.tokenizer])

    def save_data_to_excel(self):
        self.train_df.to_excel('train_data.xlsx')

    def save_data_to_csv(self):
        self.train_df.to_csv('train_data.xlsx')
