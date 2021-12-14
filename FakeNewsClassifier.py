#!/usr/bin/env python
# coding: utf-8

# # Fake News Classifier
# ## NL1 project - Matteo Santelmo
#
# Dataset: https://www.kaggle.com/c/fake-news/data#

import pandas as pd
import tensorflow as tf
from tensorflow imort keras
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from platform import python_version
import nltk
from keras.utils.vis_utils import plot_model
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import numpy as np


df=pd.read_csv('train.csv') #import dataset


# In[3]:


df.head()


# In[4]:


df=df.dropna() #drop missing values (nan values)
x=df.drop('label',axis=1) #drop also the label
y=df['label'] #the label that tells me wheter the news is fake or not will be the output


# In[5]:


print("Tensorflow -> ",tf.__version__) # 2.7.0
print("Python -> ",python_version())   # 3.8.12


# In[6]:


voc_size=5000 #vocabulary size


# ## Data cleaning

# In[7]:


messages=x.copy()
messages.reset_index(inplace=True)


# In[8]:


messages['title'][2]
# messages['text'][3]


# In[9]:


nltk.download('stopwords')


# ### Data preprocessing

# In[10]:


#stemming is the process of reducing words to their word stem, base or root form
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


# In[11]:


# through one hot every word is represented as a number that is specific for that single word
onehot_repr=[one_hot(words,voc_size)for words in corpus]


# ## Creating model

# In[13]:


sent_length=20
embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length) # fix sentences' lentgh


# In[14]:


# the embedding layer converts the input into a vector with a specific number of features
embedding_vector_features=50
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(voc_size,embedding_vector_features,input_length=sent_length),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()


# In[15]:


len(embedded_docs),y.shape


# In[16]:


x_final=np.array(embedded_docs)
y_final=np.array(y)


# In[17]:


x_final.shape,y_final.shape


# In[18]:


x_train, x_test, y_train, y_test = train_test_split(x_final, y_final, test_size=0.33, random_state=42)


# ### Model Training

# In[19]:


model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10,verbose=2)


# In[24]:


model.save('float_model')


# ### Performance Metrics And Accuracy

# In[21]:


predict_x=model.predict(x_test)
#y_pred=(model.predict(x_test) > 0.5).astype("int32")
y_pred=np.argmax(model.predict(x_test),axis=1)
y_pred[0]


# In[22]:


confusion_matrix(y_test,y_pred)


# In[23]:


accuracy_score(y_test,y_pred)
