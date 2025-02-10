import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
lemmitazier = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
model = load_model('chatbot_model.h5')
def clean_up_sentences(sentences):
    sentences_words = nltk.word_tokenize(sentences)
    sentences_words = [lemmitazier.lemmatize(word) for word in sentences_words]
    return sentences_words
def bag_of_words(sentences):
    sentences_words = clean_up_sentences(sentences)
    bag = [0]*len(words)
    for w in sentences_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] =1
    return np.array(bag)      
def predict_class(sentences):
    bow = bag_of_words(sentences)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    result = [[i,r]for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    result.sort(key=lambda x: x[1],reverse=True)
    return_list = []
    for r in result:
        return_list.append({'intent':classes[r[0]],'probability':str(r[1])})      
    return return_list
def get_response(intent_list,intent_json):
    tag = intent_list[0]['intent']
    list_of_intent = intent_json['intents']
    for i in list_of_intent:
        if i ['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result
print("Go! Bot is running") 
while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints,intents)       
    print(res)