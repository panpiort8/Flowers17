import argparse
import os
from utils import arch_semi_parse
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-g", "--gpu_no", type=int, default=3, help="no of gpu")
ap.add_argument("-e", "--epochs", type=int, default=100, help="number of epochs")
ap.add_argument("-s", "--suffix", type=str, default=None, help="output names of files suffix")
opts = ['sgd', 'adam', 'rms']
ap.add_argument("-o", "--optimizer", type=str, default="sgd", help="optimalization method ({})".format(opts))
ap.add_argument("-l", "--lrt", type=float, default=0.05, help="learning rate")
args, net_cls = arch_semi_parse(ap)

os.environ["CUDA_VISIBLE_DEVICES"]=str(args['gpu_no'])
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datasets import DatasetLoader
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
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
opt_name = args['optimizer'].lower()
if opt_name not in opts:
    opt_name = opts[0]

if opt_name == 'sgd':
    opt = SGD(lr=args['lrt'])
elif opt_name == 'adam':
    opt = Adam(lr=args['lrt'])
else:
    opt = RMSprop(lr=args['lrt'])


model = net_cls.build(width=64, height=64, depth=3, classes=len(classNames))
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

aug = ImageDataGenerator(rotation_range=10, width_shift_range=0.1,
                         height_shift_range=0.1, horizontal_flip=True, fill_mode="nearest")

print("[INFO] training network...")
# H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=args['epochs'], verbose=1)
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=64),
                        validation_data=(testX, testY), epochs=args['epochs'],
                        steps_per_epoch=len(trainX) // 64, verbose=1)


print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
report = str(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=classNames))
print(report)

print("[INFO] saving files...")
file_base = "{}_e{}_{}_lrt{}".format(net_cls.__name__, args['epochs'], args['optimizer'], args['lrt'])
if args['suffix'] is not None:
    file_base += args['suffix']

fig_name = os.path.join("tmp", file_base + '.png')
report_name = os.path.join("tmp", file_base + '.txt')

with open(report_name, "w") as text_file:
    text_file.write(report)

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, args['epochs']), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, args['epochs']), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, args['epochs']), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, args['epochs']), H.history["val_acc"], label="val_acc")
plt.title(file_base)
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(fig_name)