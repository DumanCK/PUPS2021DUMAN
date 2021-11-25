#--------------------------------------------------------------------
#  Fine Tuning Model
#--------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import os

# dataset gambar
dataset = "dataset"

# model output
omodel = "./output/modelpisang.h5"

# label output
olabel = "./output/labelpisang.lb"

# set epoch
setepoch = 20

# plot output
oplot = "./output/plotpisang.png"

# inisialisasi label yang ingin kita latih
LABELS = set(["bunga", "pisang", "singa"])

# ambil gambar yang ingin kita latih
print("[INFO] loading images...")
imagePaths = list(paths.list_images(dataset))
data = []
labels = []

# loop ke semua image dalam path
for imagePath in imagePaths:
	# ambil labelnya dari path (folder)
	label = imagePath.split(os.path.sep)[-2]

	# kalau labelnya tidak ada dalam list tidak usah diambil
	if label not in LABELS:
		continue

	# ambil image, convert ke RGB (tadinya BGR), resize
	# fix 224x224 pixel, tidak melihat aspect ratio
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (224, 224))

	# update data dan label di list, berurutan
	data.append(image)
	labels.append(label)

# konversi data dan label menjadi NumPy array
data = np.array(data)
labels = np.array(labels)
print(labels)

# lakukan one-hot encode ke label (jadikan binarizer)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

#labels = labels.reshape(-1, 1)
print(labels)

# partisi date menjadi train dan test menjadi
# 75% training dan 25% testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25, stratify=labels, random_state=14)

print(testY)

# inisialisai data training menjadi data augmentation
trainAug = ImageDataGenerator(
	rotation_range=30,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# inisialisasi data validasi augmentasi (yang akan kita gunakan)
valAug = ImageDataGenerator()

# define the ImageNet mean subtraction (in RGB order) and set the
# lakukan ImageNet mean substraction (urutan RGB) dan set nilai mean substraction
# untuk setiap data augmentation
mean = np.array([123.68, 116.779, 103.939], dtype="float32")
trainAug.mean = mean
valAug.mean = mean

# buka ResNet-50, pastikan FC layer dimatikan
baseModel = ResNet50(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# buat head model yang akan kita letakkan diatas base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(lb.classes_), activation="softmax")(headModel)

# letakkan model kita (FC) ini akan menjadi model yang akan kita latih
model = Model(inputs=baseModel.input, outputs=headModel)

# lakukan pelatihan ke semua layer di base model dan freeze
# sehingga base model tidak di update pada saat pelatihan
for layer in baseModel.layers:
	layer.trainable = False

# compile our model
print("[INFO] compiling model...")
opt = SGD(lr=1e-4, momentum=0.9, decay=1e-4 / setepoch)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

print("[INFO] training head...")
H = model.fit(
	x=trainAug.flow(trainX, trainY, batch_size=32),
	steps_per_epoch=len(trainX) // 32,
	validation_data=valAug.flow(testX, testY),
	validation_steps=len(testX) // 32,
	epochs=setepoch)

# evaluasi
print("[INFO] evaluating network...")
predictions = model.predict(x=testX.astype("float32"), batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))

# plot train loss grafik
N = setepoch
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(oplot)

# save model
print("[INFO] serializing network...")
model.save(omodel, save_format="h5")

# save label
f = open(olabel, "wb")
f.write(pickle.dumps(lb))
f.close()
