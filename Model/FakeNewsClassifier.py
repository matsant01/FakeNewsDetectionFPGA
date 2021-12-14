#!/usr/bin/env python
# coding: utf-8

# # Fake News Classifier
# ## NL1 project - Matteo Santelmo
#
# Dataset: https://www.kaggle.com/c/fake-news/data#

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow_model_optimization.quantization.keras import vitis_quantize
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
print("Tensorflow -> ",tf.__version__) # 2.7.0
print("Python -> ",python_version())   # 3.8.12


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
dataset=np.array(embedded_docs)


embedding_vector_features=64
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(voc_size,embedding_vector_features,input_length=sent_length),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()


len(embedded_docs),y.shape
x_final=np.array(embedded_docs)
y_final=np.array(y)


#x_final.shape,y_final.shape
x_train, x_test, y_train, y_test = train_test_split(x_final, y_final, test_size=0.33, random_state=42)
model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10,verbose=2)



model.save('float.h5')
#model=load_model('float_model.h5')
print('------------------------------------\n')
print('              2                     \n')
print('------------------------------------\n')
quantizer = vitis_quantize.VitisQuantizer(model)
print('------------------------------------\n')
print('              3                     \n')
print('------------------------------------\n')
quantized_model = quantizer.quantize_model(calib_dataset=dataset)
print('------------------------------------\n')
print('              4                     \n')
print('------------------------------------\n')
# saved quantized model
quantized_model.save('quantized_model.h5')
print('Saved quantized model to')


#predict_x=model.predict(x_test)
#y_pred=(model.predict(x_test) > 0.5).astype("int32")
#y_pred=np.argmax(model.predict(x_test),axis=1)

#confusion_matrix(y_test,y_pred)
#print(accuracy_score(y_test,y_pred))
