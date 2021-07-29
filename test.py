from tensorflow.keras.models import model_from_json
import numpy as np
import random

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model.h5")

val_x = np.load("val_x.npy")
val_y = np.load("val_y.npy")

accuracy = 0
total = 100
# random.seed(10)
for i in range(1, total):
    r = random.randint(0, len(val_x))
    pp = model.predict(val_x[r].reshape(1, 600, 300, 3))
    # print(f"True:{val_y[r]} | Predicted:{np.argmax(pp[0])}")
    if val_y[r][0] == np.argmax(pp[0]):
        accuracy = accuracy + 1

print(f"Accuracy:{accuracy/total}")
