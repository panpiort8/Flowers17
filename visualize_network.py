import architectures as arch
from keras.utils import plot_model
import argparse
import sys
import os
from utils import *

parser = argparse.ArgumentParser(description='Visualize network')
args, net_cls = arch_semi_parse(parser)

model = net_cls.build(64, 64, 3, 17)
name = os.path.join("images/architectures", net_cls.__name__)
plot_model(model, to_file=name, show_shapes=True)

