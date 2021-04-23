averages=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])
for i in x_train:
    for j in range(len(i)):
        averages[j]=averages[j]+i[j]
aveages=averages/891

x_train=x_train/averages


model= tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(7, input_shape=(1,7), activation="relu"))
#model.add(tf.keras.layers.Dropout(0.05))
model.add(tf.keras.layers.Dense(5, activation="relu"))
model.add(tf.keras.layers.Dense(3, activation="relu"))
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))
#model.load_weights("model_final.h5")
model.compile(optimizer=op, loss=lol, metrics=['accuracy'])
history=model.fit(x_train,y_train,verbose=verb,epochs=num_epochs,batch_size=bat)

model.summary()

plot_graphs(history, "accuracy")

ti=input("save weights?(Y/N)")
if ti=="Y" :
    model_json=model.to_json()
    with open("model_final.json", "w") as json_file :
        json_file.write(model_json)
    model.save_weights("model_final.h5")
