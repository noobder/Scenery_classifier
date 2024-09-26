import pandas as pd
import numpy as np
import seaborn as sns
sns.set(style="whitegrid")
import matplotlib.pyplot as plt
%matplotlib inline

import os
import glob as gb
import cv2
import tensorflow as tf
import keras

import warnings
warnings.filterwarnings('ignore')
train_path ='seg_train/'
test_path ='seg_test/'
valid_path ='seg_pred/'
for folder in  os.listdir(train_path + 'seg_train') : 
    files = gb.glob(pathname= str( train_path +'seg_train//' + folder + '/*.jpg'))

for folder in  os.listdir(test_path +'seg_test') : 
    files = gb.glob(pathname= str( test_path +'seg_test//' + folder + '/*.jpg'))

files = gb.glob(pathname= str(valid_path +'seg_pred/*.jpg'))

code = {'buildings':0 ,'forest':1,'glacier':2,'mountain':3,'sea':4,'street':5}

def getcode(n) : 
    for x , y in code.items() : 
        if n == y : 
            return x   

size = []
for folder in  os.listdir(train_path +'seg_train') : 
    files = gb.glob(pathname= str( train_path +'seg_train//' + folder + '/*.jpg'))
    for file in files: 
        image = plt.imread(file)
        size.append(image.shape)
pd.Series(size).value_counts()

size = []
for folder in  os.listdir(test_path +'seg_test') : 
    files = gb.glob(pathname= str( test_path +'seg_test//' + folder + '/*.jpg'))
    for file in files: 
        image = plt.imread(file)
        size.append(image.shape)
pd.Series(size).value_counts()

size = []
files = gb.glob(pathname= str(valid_path +'seg_pred/*.jpg'))
for file in files: 
    image = plt.imread(file)
    size.append(image.shape)
pd.Series(size).value_counts()

s = 100
x_train = []
y_train = []
for folder in  os.listdir(train_path +'seg_train') : 
    files = gb.glob(pathname= str( train_path +'seg_train//' + folder + '/*.jpg'))
    for file in files: 
        image = cv2.imread(file)
        image_array = cv2.resize(image , (s,s))
        x_train.append(list(image_array))
        y_train.append(code[folder])

plt.figure(figsize=(25,25))
for n , i in enumerate(list(np.random.randint(0,len(x_train),36))) : 
    plt.subplot(6,6,n+1)
    plt.imshow(x_train[i])   
    plt.axis('off')
    plt.title(getcode(y_train[i]))

x_test = []
y_test = []
for folder in  os.listdir(test_path +'seg_test') : 
    files = gb.glob(pathname= str(test_path + 'seg_test//' + folder + '/*.jpg'))
    for file in files: 
        image = cv2.imread(file)
        image_array = cv2.resize(image , (s,s))
        x_test.append(list(image_array))
        y_test.append(code[folder])

plt.figure(figsize=(25,25))
for n , i in enumerate(list(np.random.randint(0,len(x_test),36))) : 
    plt.subplot(6,6,n+1)
    plt.imshow(x_test[i])    
    plt.axis('off')
    plt.title(getcode(y_test[i]))

x_pred = []
files = gb.glob(pathname= str(valid_path + 'seg_pred/*.jpg'))
for file in files: 
    image = cv2.imread(file)
    image_array = cv2.resize(image , (s,s))
    x_pred.append(list(image_array))   

plt.figure(figsize=(25,25))
for n , i in enumerate(list(np.random.randint(0,len(x_pred),36))) : 
    plt.subplot(6,6,n+1)
    plt.imshow(x_pred[i])    
    plt.axis('off')

x_train = np.array(x_train)
x_test = np.array(x_test)
x_pred_array = np.array(x_pred)
y_train = np.array(y_train)
y_test = np.array(y_test)

print(f'X_train shape  is {x_train.shape}')
print(f'X_test shape  is {x_test.shape}')
print(f'X_pred shape  is {x_pred_array.shape}')
print(f'y_train shape  is {y_train.shape}')
print(f'y_test shape  is {y_test.shape}')


KerasModel = keras.models.Sequential([
        keras.layers.Conv2D(200,kernel_size=(3,3),activation='relu',input_shape=(s,s,3)),
        keras.layers.Conv2D(150,kernel_size=(3,3),activation='relu'),
        keras.layers.MaxPool2D(4,4),
        keras.layers.Conv2D(120,kernel_size=(3,3),activation='relu'),    
        keras.layers.Conv2D(80,kernel_size=(3,3),activation='relu'),    
        keras.layers.Conv2D(50,kernel_size=(3,3),activation='relu'),
        keras.layers.MaxPool2D(4,4),
        keras.layers.Flatten() ,    
        keras.layers.Dense(120,activation='relu') ,    
        keras.layers.Dense(100,activation='relu') ,    
        keras.layers.Dense(50,activation='relu') ,        
        keras.layers.Dropout(rate=0.5) ,            
        keras.layers.Dense(6,activation='softmax') ,    
        ])



KerasModel.compile(optimizer ='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
print(KerasModel.summary())

ThisModel = KerasModel.fit(x_train, y_train, epochs=40,batch_size=64,verbose=1)
KerasModel.evaluate(x_test, y_test)
KerasModel.predict(x_test)
y_result=KerasModel.predict(x_pred_array)
KerasModel.save('image_classification_model.h5')

plt.figure(figsize=(25,25))
for n , i in enumerate(list(np.random.randint(0,len(x_pred),36))) : 
    plt.subplot(6,6,n+1)
    plt.imshow(x_pred[i])    
    plt.axis('off')
    plt.title(getcode(np.argmax(y_result[i])))