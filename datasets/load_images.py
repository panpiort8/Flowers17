import os
import numpy as np

def load_images(path):
    imagePaths = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        for file in filenames:
            imagePaths.append(os.path.join(dirpath, file))
    classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
    return imagePaths, [str(x) for x in np.unique(classNames)]