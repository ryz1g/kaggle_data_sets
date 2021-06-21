import os
from PIL import Image, ImageOps
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical

data = {}
missing = [94, 97, 98]
val_split = 0.05

for i in range(1, 102):
    if i in missing:
        continue
    tmp = []
    suf = str(i)
    if i < 10:
        suf = '00'+suf
    elif i < 100:
        suf = '0'+suf
    s = 'data/'+suf+'/'
    entries = os.scandir(s)
    for entry in entries:
        im = (ImageOps.grayscale(Image.open(s+entry.name)))
        # im = Image.open(s+entry.name)
        tmp.append(np.asarray(im))
    data[i] = tmp
    print(f"Age-{i}-done!")

i = 1
data_r = []
tmp = []

for k, v in data.items():
    if k < 91:
        if len(v) <= 180:
            data_r.append(v)
        else:
            data_r.append(v[:180])
    else:
        tmp = tmp + v
data_r.append(tmp)

"""
age = [i for i in range(1, 92)]
num = [len(i) for i in data_r]
plt.plot(age, num)
plt.xlabel("Distribution")
plt.ylabel("Number")
plt.show()
"""

data_x = []
data_y = []
val_x = []
val_y = []
c = 1
for i in data_r:
    for j in i:
        data_x.append(j)
        data_y.append(c)
    c = c+1

random.seed(10)
random.shuffle(data_x)
random.shuffle(data_x)
random.shuffle(data_x)
random.seed(10)
random.shuffle(data_y)
random.shuffle(data_y)
random.shuffle(data_y)

val_x = data_x[:int(val_split*len(data_x))]
val_y = data_y[:int(val_split*len(data_y))]

data_x = data_x[int(val_split*len(data_x)):]
data_y = data_y[int(val_split*len(data_y)):]

data_x = np.array(data_x).reshape((len(data_x), 200, 200, 1))
data_y = np.array(data_y)
#data_y = to_categorical(data_y, num_classes=32)
val_x = np.array(val_x).reshape((len(val_x), 200, 200, 1))
val_y = np.array(val_y)
#val_y = to_categorical(val_y, num_classes=32)
print(data_y.shape)
print(val_y.shape)

np.save("data_x_", data_x)
np.save("data_y_", data_y)
np.save("val_x_", val_x)
np.save("val_y_", val_y)
