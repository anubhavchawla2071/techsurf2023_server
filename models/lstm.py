import joblib
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

def generate_text(input_text, model_filename='D:/Text-Generator/server/lstm_model.pkl', tokenizer_filename='D:/Text-Generator/server/tokenizer.pkl', max_seq_length=11, next_words=25):
    # Load the trained model
    with open(model_filename, 'rb') as model_file:
        model = joblib.load(model_file)

    # Load the tokenizer
    with open(tokenizer_filename, 'rb') as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)

    generated_text = input_text

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([generated_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_length-1, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=1)

        new_word = tokenizer.index_word[predicted[0]]  # Convert index to word

        generated_text += " " + new_word

    return generated_text