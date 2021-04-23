import csv
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam

x_train=[]
y_train=[]
x_test=[]

#hyperparameters
num_epochs=1000
bat=891
op=Adam(lr=0.00001)
lol="BinaryCrossentropy"
verb=2

def plot_graphs(history, strr):
    for i in range(len(strr)):
        plt.plot(history.history[strr[i]])
    plt.xlabel("Epochs")
    plt.ylabel(strr[0])
    plt.show()

csvfile=open("train.csv")
reader=csv.reader(csvfile, delimiter=",")
next(reader)
for row in reader:
    temp=[]
    for i in range(12):
        if i==3 or i==1 or i==10 or i==0:
            continue
        elif i==4:
            if row[4]=="male":
                temp.append(float(0))
            else:
                temp.append(float(1))
        elif i==11:
            if row[11]=='S':
                temp.append(float(0))
            elif row[11]=='Q':
                temp.append(float(1))
            else:
                temp.append(float(2))
        elif i==8:
            st=row[8]
            temp.append(float(st[st.find(' ')+1:]))
        else:
            temp.append(float(row[i]))
    x_train.append(temp)
    y_train.append(int(row[1]))

x_train=np.array(x_train)
y_train=np.array(y_train).reshape(891,1)

model= tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(1000, input_shape=(8,), activation="relu"))
#model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Dense(900, activation="relu"))
model.add(tf.keras.layers.Dense(750, activation="relu"))
model.add(tf.keras.layers.Dense(600, activation="relu"))
model.add(tf.keras.layers.Dense(400, activation="relu"))
model.add(tf.keras.layers.Dense(250, activation="relu"))
model.add(tf.keras.layers.Dense(120, activation="relu"))
#model.add(tf.keras.layers.Dropout(0.08))
model.add(tf.keras.layers.Dense(100, activation="relu"))
#model.add(tf.keras.layers.Dropout(0.03))
model.add(tf.keras.layers.Dense(80, activation="relu"))
model.add(tf.keras.layers.Dense(50, activation="relu"))
model.add(tf.keras.layers.Dense(20, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="relu"))
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))
#model.load_weights("model_final.h5")
model.compile(optimizer=op, loss=lol, metrics=['accuracy'])
history=model.fit(x_train,y_train,verbose=verb,epochs=num_epochs,batch_size=bat)

model.summary()

plot_graphs(history, ["accuracy"])

ti=input("save weights?(Y/N)")
if ti=="Y" :
    model_json=model.to_json()
    with open("model_final.json", "w") as json_file :
        json_file.write(model_json)
    model.save_weights("model_final.h5")
