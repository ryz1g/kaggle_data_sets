import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.models import model_from_json
import csv
from os import system

json_file = open('model_final.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model_final.h5")

x_test=[]

csvfile=open("test.csv")
reader=csv.reader(csvfile, delimiter=",")
next(reader)
for row in reader:
    temp=[]
    for i in range(11):
        if i==2 or i==0 or i==7 or i==9:
            continue
        elif i==3:
            if row[3]=="male":
                temp.append(float(0))
            else:
                temp.append(float(1))
        elif i==10:
            if row[10]=='S':
                temp.append(float(0))
            elif row[10]=='Q':
                temp.append(float(1))
            else:
                temp.append(float(2))
        else:
            temp.append(float(row[i]))
    x_test.append(temp)
csvfile.close()

x_test=np.array(x_test)

def clear() :
    _=system("cls")

clear()

pred=[]
for i in range(len(x_test)):
    pred.append(loaded_model.predict(x_test[i].reshape(1,7)))

pred=np.array(pred).reshape(418,1)
pred=np.where(pred<0.5,0,1)
c=892
f=open("submission.csv", "w")
f.write("PassengerId,Survived\n")
for i in range(len(pred)):
    f.write(str(c))
    c=c+1
    f.write(",")
    f.write(str(pred[i][0]))
    f.write("\n")
f.close()
