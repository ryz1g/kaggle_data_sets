# %% [code] {"jupyter":{"outputs_hidden":false}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"jupyter":{"outputs_hidden":false}}
data=pd.read_csv('/kaggle/input/mobile-price-classification/train.csv')
data_y=data.price_range
data_x=data.iloc[:,:-1].values
data_y=tf.keras.utils.to_categorical(data.iloc[:,-1].values)
print(data_x.shape)

# %% [code] {"jupyter":{"outputs_hidden":false}}
class myCall(tf.keras.callbacks.Callback) :
    def on_epoch_end(self, epoch, logs={}) :
        if(logs.get('accuracy')>0.9) :
            self.model.stop_training=True
callbacks=myCall()

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.show()

# %% [code] {"jupyter":{"outputs_hidden":false}}
model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(30, input_shape=[20], activation='relu'))
model.add(tf.keras.layers.Dense(20, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='relu'))
model.add(tf.keras.layers.Dense(4, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history=model.fit(data_x,data_y,verbose=0,epochs=2000, callbacks=[callbacks], batch_size=2048)

plot_graphs(history, 'accuracy')

# %% [code] {"jupyter":{"outputs_hidden":false}}
test=pd.read_csv('/kaggle/input/mobile-price-classification/test.csv')
#test.head()
ans=[]
for c in range(len(test)):
    ans.append(np.argmax(model.predict(test.iloc[c][1:].values.reshape(1,20)), axis=-1)[0])
    if c%100==0:
        print(c)
#print(test.loc[0].values.reshape(1,20))

# %% [code] {"jupyter":{"outputs_hidden":false}}
print(ans)
