import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-t", "--target", required=True, help="path to target")
args = vars(ap.parse_args())

from preprocessing import *
from datasets import *
from datasets.load_images import load_images

aap = AspectAwarePreprocessor(128, 128)
iap = ImageToArrayPreprocessor()
sdl = DatasetLoader(preprocessors=[aap, iap])
sdl.load(image_paths=load_images(args['dataset'])[0], dest_path=args['target'], verbose=200)