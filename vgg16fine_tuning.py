import argparse
import os
from datasets.load_images import load_images

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-g", "--gpu_no", type=int, default=3, help="no of gpu")
ap.add_argument("-e1", "--epochs_1", type=int, default=100, help="number of epochs_1")
ap.add_argument("-e2", "--epochs_2", type=int, default=100, help="number of epochs_2")
opts = ['sgd', 'adam', 'rms']
ap.add_argument("-o1", "--optimizer_1", type=str, default="sgd", help="optimalization method ({})".format(opts))
ap.add_argument("-o2", "--optimizer_2", type=str, default="sgd", help="optimalization method ({})".format(opts))
ap.add_argument("-l1", "--lrt_1", type=float, default=0.05, help="learning rate")
ap.add_argument("-l2", "--lrt_2", type=float, default=0.0001, help="learning rate")
args = vars(ap.parse_args())

os.environ["CUDA_VISIBLE_DEVICES"]=str(args['gpu_no'])
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datasets import DatasetLoader
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.models import save_model
from keras.preprocessing.image import ImageDataGenerator
from keras import Input, Model
from keras.applications import VGG16
from keras.layers import Flatten, Dense, Dropout
from utils.figure_plot import get_figure

class FCHeadNet:
    @staticmethod
    def build(baseModel, classes, D):
        headModel = baseModel.output
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(D, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)
        headModel = Dense(classes, activation="softmax")(headModel)
        return headModel


fc_layers = [256]

file_base = "VGG16_{}_{}_{}_{}__{}_{}_{}".format(fc_layers, args['epochs_1'], args['optimizer_1'], args['lrt_1'],
                                                  args['epochs_2'], args['optimizer_2'], args['lrt_2'])

fig_name_1 = os.path.join("tmp", file_base + '_1.png')
fig_name_2 = os.path.join("tmp", file_base + '_2.png')
report_name_1 = os.path.join("tmp", file_base + '_1.txt')
report_name_2 = os.path.join("tmp", file_base + '_2.txt')
report_name_best = os.path.join("tmp", file_base + '_best.txt')
model_name = os.path.join("tmp", file_base + '.hdf5')


print("[INFO] loading images...")
image_paths, class_names = load_images(args['dataset'])

sdl = DatasetLoader()
(data, labels) = sdl.load(image_paths, verbose=500)
data = data.astype("float") / 255.0
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)


print("[INFO] loading model...")
base_model = VGG16(weights="imagenet", include_top=False,
                  input_tensor=Input(shape=(128, 128, 3)))


print("[INFO] surgering model...")
top_model = FCHeadNet.build(base_model, 17, 256)

for layer in base_model.layers:
    layer.trainable = False
model = Model(inputs=base_model.input, outputs=top_model)


print("[INFO] compiling model...")
opt_name = args['optimizer_1'].lower()
if opt_name not in opts: opt_name = opts[0]
if opt_name == 'sgd': opt = SGD(lr=args['lrt_1'])
elif opt_name == 'adam': opt = Adam(lr=args['lrt_1'])
else: opt = RMSprop(lr=args['lrt_1'])
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])



print("[INFO] training top layers...")
aug = ImageDataGenerator(rotation_range=10, width_shift_range=0.1,
                         height_shift_range=0.1, horizontal_flip=True, fill_mode="nearest")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=64),
                        validation_data=(testX, testY), epochs=args['epochs_1'],
                        steps_per_epoch=len(trainX) // 64, verbose=1)


print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
report = str(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1), target_names=class_names))
print(report)
with open(report_name_1, "w") as text_file:
    text_file.write(report)
get_figure(H, args['epochs_1'], 'first phase').savefig(fig_name_1)

print("[INFO] unfreezing layers...")
for layer in base_model.layers:
    layer.trainable = True

opt_name = args['optimizer_2'].lower()
if opt_name not in opts: opt_name = opts[0]
if opt_name == 'sgd': opt = SGD(lr=args['lrt_2'])
elif opt_name == 'adam': opt = Adam(lr=args['lrt_2'])
else: opt = RMSprop(lr=args['lrt_2'])
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


print("[INFO] fine-tuning model...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=64),
                        validation_data=(testX, testY), epochs=args['epochs_2'],
                        steps_per_epoch=len(trainX) // 64, verbose=1)


print("[INFO] evaluating fine-tuned network...")
predictions = model.predict(testX, batch_size=32)
report = str(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=class_names))
print(report)
with open(report_name_2, "w") as text_file:
    text_file.write(report)
get_figure(H, args['epochs_2'], 'fine-tuned').savefig(fig_name_2)

print("[INFO] saving model...")
save_model(model, model_name)
