from keras.applications import InceptionV3

model = InceptionV3(weights="imagenet")
model.save('model-full.h5')