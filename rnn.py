from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
import numpy as np
mx.random.seed(1)
ctx = mx.gpu(0)

with open("./data/nlp/timemachine.txt") as f:
    time_machine = f.read()

character_list = list(set(time_machine))
vocab_size = len(character_list)
print(character_list)
print("Length of vocab: %s" % vocab_size)