import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import model_from_json


def plot_graphs(history, string, t):
    if t==1:
      plt.plot(history.history["val_loss"][1:], color="green")
    plt.plot(history.history[string][1:], color='red')
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.show()

    
class myCall(tf.keras.callbacks.Callback) :
    def on_epoch_end(self, epoch, logs={}) :
        if(logs.get('val_loss')<7) :
            self.model.stop_training=True


data_x = np.load("data_x_.npy")
data_y = np.load("data_y_.npy")
val_x = np.load("val_x_.npy")
val_y = np.load("val_y_.npy")


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(64, (5, 5), input_shape=(200, 200, 1), activation="relu"))
model.add(tf.keras.layers.MaxPooling2D(3, 3))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu"))
model.add(tf.keras.layers.MaxPooling2D(2, 2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(1, activation="relu"))

# model.load_weights("model.h5")

model.compile(optimizer="Adam", loss="mse", metrics=['accuracy'])
history = model.fit(data_x, data_y, validation_data=(val_x, val_y), shuffle=True, verbose=1, epochs=25, batch_size=128, callbacks=[myCall()])

model.summary()

plot_graphs(history, "loss")

ti = input("save weights?(Y/N)")
if ti == "Y":
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")
