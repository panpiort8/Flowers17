from architectures import *
from keras.utils import plot_model

# algo_base = api.Algorithm
# all_algos = [ (name, cls) for name, cls in algorithms.__dict__.items() if
#         isinstance(cls, type) and
#         issubclass(cls, algo_base) and
#         cls != algo_base
#     ]

model = MiniVGGNet.build(64, 64, 1, 17)
plot_model(model, show_shapes=True)

