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
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import numpy as np
from utils.figure_plot import get_figure


file_base = "{}_{}_{}_{}".format(net_cls.__name__, args['epochs'], args['optimizer'], args['lrt'])
if args['suffix'] is not None:
    file_base += args['suffix']
fig_name = os.path.join("tmp", file_base + '.png')
report_name = os.path.join("tmp", file_base + '.txt')
report_best_name = os.path.join("tmp", file_base + '_best.txt')
model_name = os.path.join("tmp", file_base + '.hdf5')



print("[INFO] loading images...")
imagePaths = []
for (dirpath, dirnames, filenames) in os.walk(args['dataset']):
    for file in filenames:
        imagePaths.append(os.path.join(dirpath, file))
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

if opt_name == 'sgd': opt = SGD(lr=args['lrt'])
elif opt_name == 'adam': opt = Adam(lr=args['lrt'])
else: opt = RMSprop(lr=args['lrt'])




print("[INFO] preparing network...")
model = net_cls.build(width=64, height=64, depth=3, classes=len(classNames))
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])



print("[INFO] training network...")
aug = ImageDataGenerator(rotation_range=10, width_shift_range=0.1,
                         height_shift_range=0.1, horizontal_flip=True, fill_mode="nearest")
msv = ModelCheckpoint(model_name, save_best_only=True, monitor='val_loss', mode='min')
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=64), callbacks=[msv],
                        validation_data=(testX, testY), epochs=args['epochs'],
                        steps_per_epoch=len(trainX) // 64, verbose=1)



print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
report = str(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=classNames))
print(report)
with open(report_name, "w") as text_file:
    text_file.write(report)



print("[INFO] loading best network...")
model = load_model(model_name)



print("[INFO] evaluating best network...")
predictions = model.predict(testX, batch_size=32)
report = str(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=classNames))
print(report)
with open(report_best_name, "w") as text_file:
    text_file.write(report)



print("[INFO] saving figure...")
get_figure(H, args['epochs'], file_base).savefig(fig_name)