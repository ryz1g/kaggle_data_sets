from PIL import Image, ImageOps
from tensorflow.keras.models import model_from_json
import numpy as np

json_file = open('model_new.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model_new.h5")
correct = [-1, 22, 3, 25, 55, 60, 25, 80, 75, 50, 50, 81, 3, 55, 70, 45, 55, 20, 25, 50, 17, 3, 20, 28]
corr = [-1, 45, 45, 45, 7, 7, 31, 31, 56, 56]

for i in range(1, 10):
    s = "ima"+str(i)+".png"
    im = Image.open(s)
    width, height = im.size
    # im = im.crop((0.06 * width, 0.06 * height, 0.94 * width, 0.94 * height))
    im = im.resize((200, 200))
    # im.show()
    im = ImageOps.grayscale(im)
    im = np.array(im).reshape((1, 200, 200, 1))
    ar = loaded_model.predict(im)[0]
    # print(ar)
    # ages=[i*3 for i in range(1,33)]
    # plt.plot(ages,ar)
    # plt.show()
    # print(f"{i}){np.argmax(ar)*3}-{(np.argmax(ar)+1)*3} years of age!")
    print(f"{i} - {int(ar[0])} years - {corr[i]}")

# ages=[i*3 for i in range(1,33)]
# plt.plot(ages,ar)
# plt.show()
