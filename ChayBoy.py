import numpy as np
import random
import json
import pickle
import nltk

from nltk.stem import WordNetLemmatizer
from keras.models import load_model
lemmatizer = WordNetLemmatizer()

#Importamos los archivos que sean necesarios generados en el codigo, entonces debemos mejorar los espacios
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pk1', 'rb'))
classes = pickle.load(open('classes.pk1', 'rb'))
model = load_model('chatbot_model.h5')

#Ahora pasamos a mejorar las palabras  de las oraciones a su forma raìz
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

#Ahora convertiremos la informacion de la raìz en unos y ceros segun si es que estan presentes en los patrones
def bag_of_words(sentence):
    sentence_words = clear_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    print(bag)
    return np.array(bag)
