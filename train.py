import argparse
import os
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-g", "--gpu_no", type=int, default=3, help="no of gpu")
ap.add_argument("-e", "--epochs", type=int, default=100, help="number of epochs")
ap.add_argument("-f", "--figure", type=str, default=None, help="figure path")
args = vars(ap.parse_args())

os.environ["CUDA_VISIBLE_DEVICES"]=str(args['gpu_no'])
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datasets import DatasetLoader
from architectures import MiniVGGNet
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np


print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]

sdl = DatasetLoader()
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

print("[INFO] compiling model...")
opt = SGD(lr=0.05)
model = MiniVGGNet.build(width=64, height=64, depth=3, classes=len(classNames))
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=args['epochs'], verbose=1)

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=classNames))

if args['figure'] is not None:
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, args['epochs']), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, args['epochs']), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, args['epochs']), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, args['epochs']), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(args['figure'])