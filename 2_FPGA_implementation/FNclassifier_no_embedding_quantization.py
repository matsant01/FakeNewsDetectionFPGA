#!/usr/bin/env python
# coding: utf-8

# Fake News Classifier - Without embedding layer
## NL1 project - Matteo Santelmo
#
# Dataset: https://www.kaggle.com/c/fake-news/data#

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow_model_optimization.quantization.keras import vitis_quantize
from tensorflow_model_optimization.quantization.keras.vitis.layers import vitis_activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from platform import python_version
import nltk
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import numpy as np


df=pd.read_csv('train.csv') #import dataset
df.head()
df=df.dropna() #drop missing values (nan values)
x=df.drop('label',axis=1) #drop also the label
y=df['label'] #the label that tells me wheter the news is fake or not will be the output

########################
# DATA PRE-PROCESSING
########################

voc_size=5000 #vocabulary size
messages=x.copy()
messages.reset_index(inplace=True)
nltk.download('stopwords')
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['title'][i]) #replace non literal characters with a space
    review = review.lower() #lowercase
    review = review.split() #split into a list of words
                            #in the following line I apply the stemming process to every single word
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


onehot_repr=[one_hot(words,voc_size)for words in corpus]
sent_length=20
embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length) # fix sentences' lentgh
#embedded_docs=embedded_docs/voc_size
dataset=np.array(embedded_docs[0:128])
embedding_vector_features=64

#####################
# MODEL DEFINITION
#####################

inputs=keras.layers.Input(shape=(20,))
x=keras.layers.Dense(64,activation='relu')(inputs)
x=keras.layers.Dense(20,activation='relu')(x)
x=keras.layers.Dense(1,activation='sigmoid')(x)
predictions=x
model=keras.Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])


len(embedded_docs),y.shape
x_final=np.array(embedded_docs)
y_final=np.array(y)

##############################
#      MODEL TRAINING
##############################

print(x_final.shape,y_final.shape)
x_train, x_test, y_train, y_test = train_test_split(x_final, y_final, test_size=0.33, random_state=42)
model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10,verbose=2)
predict_x=model.predict(x_test)
y_pred=(model.predict(x_test) > 0.5).astype("int32")
print('____________________________________________\n')
print('\n    Accuracy = ',accuracy_score(y_test,y_pred))
print('\n____________________________________________\n')

##############################
#     MODEL QUANTIZATION
##############################

model.save('float_model_no_embedding.h5')
quantizer = vitis_quantize.VitisQuantizer(model)
quantized_model = quantizer.quantize_model(calib_dataset=dataset, include_cle=True, cle_steps=10, include_fast_ft=True)
# saved quantized model
quantized_model.save('quantized_model_no_embedding.h5')
print('Saved quantized model')

#quantized_model = keras.models.load_model(quantized_model)
quantized_model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
quantized_model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10,verbose=2)
predict_x=quantized_model.predict(x_test)
y_pred=(quantized_model.predict(x_test) > 0.5).astype("int32")
print('____________________________________________\n')
print('\n Quantized accuracy = ',accuracy_score(y_test,y_pred))
print('\n____________________________________________\n')
