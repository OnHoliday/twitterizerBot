

# Deep learing for NLP
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

with open("pooland_engineer.txt", "r") as a:
  article = a.read()

import string
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.corpus import wordnet
nltk.download('wordnet')

# Punctuation and stop words to be removed later
punctuation = set(string.punctuation)
stoplist = set(stopwords.words('english'))
lemma = WordNetLemmatizer()
# For LDA training later

def remove_punctuation(text):
    """
    Remove punctuation from text by checking each character against a set of punctation characters
    :text: string
    :return: string
    """
    return ''.join(char for char in text if char not in punctuation)

def remove_numbers(text):
    """
    Remove numbers from text as they aren't of value to our model
    :text: string
    :return: string
    """
    return ''.join(char for char in text if not char.isdigit())

def remove_stop_words(text):
    """
    Remove common words as they won't add any value to our model
    :text: string
    :return: string
    """
    return ' '.join([word for word in text.split() if word not in stoplist])

def remove_single_characters(text):
    """
    Remove any remaining single-character words
    :text: string
    :return: string
    """
    return ' '.join([word for word in text.split() if len(word) > 1])

def lemmatize(text):
    """
    Use NLTK lemma functionality to get the route word
    :text: string
    :return: string
    """
    return ' '.join([lemma.lemmatize(word) for word in text.split()])

def get_cleaned_text(text):
    """
    Return the page with stopwords, digits, punctuation and single character words removed
    :text: string
    :return: string
    """
    # Remove \n characters (Wikipedia has a lot of them in the page content!)
    text = text.replace('\n', '')
    # Remove numbers
    text = remove_numbers(text)
    # Remove stop words
    text = remove_stop_words(text)
    # Remove punctuation
    text = remove_punctuation(text)
    # Remove single character words
    text = remove_single_characters(text)
    # Lemmatize the document
    text = lemmatize(text)
    return text

clean_text = get_cleaned_text(article)

import re
clean = re.sub(r"[^A-Za-z]", " ", clean_text)

# load ascii text and covert to lowercase

raw_text = clean.lower()
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))


n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)

	
# prepare the dataset of input to output pairs encoded as integers
seq_length = 75
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)


# reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)

# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
model.fit(X, y, epochs=5, batch_size=64, callbacks=callbacks_list)

import sys
int_to_char = dict((i, c) for i, c in enumerate(chars))
# load the network weights
filename = "weights-improvement-50-0.1909-bigger.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')
# pick a random seed
start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print("Seed:")
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
# generate characters
for i in range(1000):
	x = np.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)
	index = np.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	sys.stdout.write(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
print("\nDone.")

