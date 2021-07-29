import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history["val_loss"])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.show()


class myCall(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss')<0.5):
            self.model.stop_training = True


data_x = np.load("train_x.npy")
data_y = np.load("train_y.npy")
val_x = np.load("val_x.npy")
val_y = np.load("val_y.npy")

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(64, (5, 5), input_shape=(600, 300, 3), activation="relu"))
model.add(tf.keras.layers.MaxPooling2D(3, 3))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu"))
model.add(tf.keras.layers.MaxPooling2D(2, 2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dense(5, activation='softmax'))

# model.load_weights("model.h5")

model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=[])
history = model.fit(data_x, data_y, validation_data=(val_x, val_y), shuffle=True, verbose=1, epochs=3, batch_size=32, callbacks=[myCall()])

model.summary()

plot_graphs(history, "loss")

ti = input("save weights?(Y/N)")
if ti == "Y":
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")
