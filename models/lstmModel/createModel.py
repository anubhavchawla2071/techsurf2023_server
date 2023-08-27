import tensorflow as tf
import pandas as pd
import numpy as np

with open('D:/Text-Generator/server/models/lstmModel/robert_frost.txt', encoding='utf-8') as story:
    story_data = story.read()

# print(story_data)


# data cleaning process
import re                                # Regular expressions to use sub function for replacing the useless text from the data

def clean_text(text):
    text = re.sub(r',', '', text)
    text = re.sub(r'/', '', text)
    text = re.sub(r'/"', '', text)
    text = re.sub(r'/\(', '', text)
    text = re.sub(r'/\)', '', text)
    text = re.sub(r'/n', '', text)
    text = re.sub(r'“', '', text)
    text = re.sub(r'”', '', text)
    text = re.sub(r'’', '', text)
    text = re.sub(r'\.', '', text)  # Escape the dot to match a literal period
    text = re.sub(r';', '', text)
    text = re.sub(r':', '', text)
    text = re.sub(r'/', '', text)  # Remove a single slash
    text = re.sub(r'-', '', text)  # Remove a single dash
    return text

# cleaning the data
lower_data = story_data.lower()           # Converting the string to lower case to get uniformity

split_data = lower_data.splitlines()      # Splitting the data to get every line seperately but this will give the list of uncleaned data

# print(split_data) #working                        

final = ''                                # initiating a argument with blank string to hold the values of final cleaned data

for line in split_data:
  line = clean_text(line)
  final += '/n' + line

# print(final) #working here

final_data = final.split('/n')       # splitting again to get list of cleaned and splitted data ready to be processed
# print(final_data) #DONE

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Instantiating the Tokenizer
max_vocab = 1000000
tokenizer = Tokenizer(num_words=max_vocab)
tokenizer.fit_on_texts(final_data)
# print("Hello")
import pickle
tokenizer_filename = 'tokenizer.pkl'
with open(tokenizer_filename, 'wb') as tokenizer_file:
    pickle.dump(tokenizer, tokenizer_file)

# # Getting the total number of words of the data.
word2idx = tokenizer.word_index
# print(len(word2idx))
# print(word2idx)
vocab_size = len(word2idx) + 1        # Adding 1 to the vocab_size because the index starts from 1 not 0. This will make it uniform when using it further
# print(vocab_size)

# # """## Creating n-gram sequences from the sentences

# # * Consider this sentence : ['two roads diverged in a yellow wood']. Here we will use ['two roads diverged in a yellow'] to predict ['wood']. This is the basic concept of forecasting which can be applied here to generate text.

# # * An advacement of this will be to use single word or every combination words possible from the sentence to predict the next word. And this is loosely termed as n_gram sequences

# # * The sentence ['two roads diverged in a yellow wood'] will have sequence as [112, 113, 114, 7, 5, 190, 75]

# # * so we will use combinations of words to make our model better

# # * [112, 113], 
# # * [112, 113, 114], 
# # * [112, 113, 114, 7], 
# # * [112, 113, 114, 7, 5], 
# # * [112, 113, 114, 7, 5, 190], 
# # * [112, 113, 114, 7, 5, 190, 75]

# # * we train our model that if 112 comes then it has to predict 113.
# # * if combination of 112, 113, comes then it has to predict 114 and so on.
# # """

# # # We will turn the sentences to sequences line by line and create n_gram sequences

input_seq = []

for line in final_data:
  token_list = tokenizer.texts_to_sequences([line])[0]
  for i in range(1, len(token_list)):
    n_gram_seq = token_list[:i+1]
    input_seq.append(n_gram_seq)

# print(input_seq)

# # Getting the maximum length of sequence for padding purpose
max_seq_length = max(len(x) for x in input_seq)
print(max_seq_length)

# # Padding the sequences and converting them to array
# input_seq = np.array(pad_sequences(input_seq, maxlen=max_seq_length, padding='pre'))
# # print(input_seq)

# # Taking xs and labels to train the model.

# xs = input_seq[:, :-1]        # xs contains every word in sentence except the last one because we are using this value to predict the y value
# labels = input_seq[:, -1]     # labels contains only the last word of the sentence which will help in hot encoding the y value in next step
# # print("xs: ",xs)
# # print("labels:",labels)

# from tensorflow.keras.utils import to_categorical

# # one-hot encoding the labels according to the vocab size

# # The matrix is square matrix of the size of vocab_size. Each row will denote a label and it will have 
# # a single +ve value(i.e 1) for that label and other values will be zero. 

# ys = to_categorical(labels, num_classes=vocab_size)
# print(ys)

# from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Dropout, Bidirectional, GlobalMaxPooling1D
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.models import Sequential

# # # using the functional APIs of keras to define the model

# i = Input(shape=(max_seq_length - 1, ))                           # using 1 less value becasuse we are preserving the last value for predicted word 
# x = Embedding(vocab_size, 124)(i)
# x = Dropout(0.2)(x)
# x = LSTM(520, return_sequences=True)(x)
# x = Bidirectional(layer=LSTM(340, return_sequences=True))(x)
# x = GlobalMaxPooling1D()(x)
# x = Dense(1024, activation='relu')(x)
# x = Dense(vocab_size, activation='softmax')(x)

# # print("hello")
# model = Model(i,x)

# # # using the pipeline method of sequential to define a model

# # # model = Sequential()
# # # model.add(Embedding(vocab_size, 124, input_length=max_seq_length-1))
# # # model.add(Dropout(0.2))
# # # model.add(LSTM(520, return_sequences=True))
# # # model.add(Bidirectional(LSTM(340, return_sequences=True)))
# # # model.add(GlobalMaxPooling1D())
# # # model.add(Dense(1024, activation='relu'))
# # # model.add(Dense(vocab_size, activation='softmax'))

# model.compile(optimizer=Adam(lr=0.001),
#               loss = 'categorical_crossentropy',
#               metrics=['accuracy'])

# # # model.summary()                                       # We can know about the shape of the model

# r = model.fit(xs,ys,epochs=100)

# # Evaluating the model on accuracy

# # Defining a function to take input of seed text from user and no. of words to be predicted

# # def predict_words(seed, no_words):
# #   for i in range(no_words):
# #     token_list = tokenizer.texts_to_sequences([seed])[0]
# #     token_list = pad_sequences([token_list], maxlen=max_seq_length-1, padding='pre')
# #     predicted = np.argmax(model.predict(token_list), axis=1)

# #     new_word = ''

# #     for word, index in tokenizer.word_index.items():
# #       if predicted == index:
# #         new_word = word
# #         break
# #     seed += " " + new_word
# #   print(seed)

# # predicting or generating the poem with the seed text

# # seed_text = 'i am feeling good today'
# # next_words = 20

# # predict_words(seed_text, next_words)
# model.save('poem_generator.h5') # Will create a HDF5 file of the model

# import pickle

# # Save the trained model using pickle
# model_filename = 'lstm_model.pkl'
# with open(model_filename, 'wb') as file:
#     pickle.dump(model, file)