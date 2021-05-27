#import the packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

#initialize the initial learning rate, number of epochs to train for and batch size
Init_LR = 1e-4
Epochs = 20
BaseSize = 32

Dataset_Directory = r"D:\Documents\Tan Jia Yin (FSKTM Year 3 Sem 2)\WIX3001 Soft Computing\Face-Mask-Detection\dataset"
Categories_Mask = ["with_mask", "without_mask"]

#take the list of images in our dataset directory, then initialize the list of data example images and class images
print("Loading images")

data = []
labels = []

for category in Categories_Mask:
    path = os.path.join(Dataset_Directory, category)
    for img in os.listdir(path):
    	img_path = os.path.join(path, img)
    	image1 = load_img(img_path, target_size=(224, 224))
    	image1 = img_to_array(image1)
    	image1 = preprocess_input(image1)

    	data.append(image1)
    	labels.append(category)

#perform encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

#construct the training image generator for data augmentation
augmentation = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

#load the MobileNetV2 network, ensuring the head FC layer sets are left off
Base_Model = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

#construct the head of the model that will be placed on top of the base model
Head_Model = Base_Model.output
Head_Model = AveragePooling2D(pool_size=(7, 7))(Head_Model)
Head_Model = Flatten(name="flatten")(Head_Model)
Head_Model = Dense(128, activation="relu")(Head_Model)
Head_Model = Dropout(0.5)(Head_Model)
Head_Model = Dense(2, activation="softmax")(Head_Model)

#place the head FC model on top of the base model (this will become the actual model we will train)
model = Model(inputs=Base_Model.input, outputs=Head_Model)

#loop over all layers in the base model and freeze them so they will not be updated during the first training process
for layer in Base_Model.layers:
	layer.trainable = False

#compile the model
print("Compiling model")
opt = Adam(lr=Init_LR, decay=Init_LR / Epochs)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

#train the head of the network
print("Training head")
Head = model.fit(
	augmentation.flow(trainX, trainY, batch_size=BaseSize),
	steps_per_epoch=len(trainX) // BaseSize,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BaseSize,
	epochs=Epochs)

#make predictions on the testing set
print("Evaluating network")
prediction = model.predict(testX, batch_size=BaseSize)

#for each image in the testing set we need to find the index of the label with corresponding largest predicted probability
prediction = np.argmax(prediction, axis=1)

#show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), prediction,
	target_names=lb.classes_))

#serialize the model to disk
print("Saving mask detector model")
model.save("mask_detector.model", save_format="h5")

#plot the training loss and accuracy
E = Epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, E), Head.history["loss"], label="train_loss")
plt.plot(np.arange(0, E), Head.history["val_loss"], label="value_loss")
plt.plot(np.arange(0, E), Head.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, E), Head.history["val_accuracy"], label="value_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")
plt.savefig("Train Loss and Accuracy Plot.png")
