import random
import json
import numpy as np
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

# Instantiate the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents
intents = json.loads(open('intents.json').read())

# Initialize lists
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# Process each intent
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)  # Use extend to add words to the list
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize words and remove duplicates
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))
training = []
output_empty = [0]*len(classes)
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    
    for word in words:
        bag.append(1 if word in word_patterns else 0)
    
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    
    training.append([bag, output_row]) 
random.shuffle(training)
training = np.array(training,dtype=object)
train_x = np.array(list(training[:,0]),dtype=np.float32)
train_y = np.array(list(training[:,1]),dtype=np.float32)
model = Sequential()
model.add(Dense(128,input_shape=(len(train_x[0]),),activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dense(len(train_y[0]),activation='softmax'))
sgd = SGD(lr=0.01,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer =sgd,metrics=['accuracy'])
hist = model.fit(np.array(train_x),np.array(train_y),epochs=200,batch_size = 5,verbose=1)
model.save('chatbot_model.h5',hist)
print('Done')