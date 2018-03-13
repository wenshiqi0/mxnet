from __future__ import print_function
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
mx.random.seed(1)
ctx = mx.cpu()

# init minist data
def transform(data, label):
    return data.astype(np.float32) / 255, label.astype(np.float32)
mnist_train = gluon.data.vision.MNIST(train=True, transform=transform)
mnist_test = gluon.data.vision.MNIST(train=False, transform=transform)

batch_size = 64
train_data = mx.gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)
test_data = mx.gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)

# init params
W1 = nd.random_normal(shape=(784,256), ctx=ctx) *.01
b1 = nd.random_normal(shape=256, ctx=ctx) * .01

W2 = nd.random_normal(shape=(256,128), ctx=ctx) *.01
b2 = nd.random_normal(shape=128, ctx=ctx) * .01

W3 = nd.random_normal(shape=(128,10), ctx=ctx) *.01
b3 = nd.random_normal(shape=10, ctx=ctx) *.01

params = [W1, b1, W2, b2, W3, b3]

for param in params:
    param.attach_grad()

# activation
def relu(X):
    return nd.maximum(X, 0)

# dropout layer
def dropout(X, drop_probability):
    keep_probability = 1 - drop_probability
    mask = nd.random_uniform(0, 1.0, X.shape, ctx=X.context) < keep_probability
    #############################
    #  Avoid division by 0 when scaling
    #############################
    if keep_probability > 0.0:
        scale = (1/keep_probability)
    else:
        scale = 0.0
    return mask * X * scale

# softmax
def softmax(y_linear):
    exp = nd.exp(y_linear-nd.max(y_linear))
    partition = nd.nansum(exp, axis=0, exclude=True).reshape((-1,1))
    return exp / partition

def softmax_cross_entropy(yhat_linear, y):
    return - nd.nansum(y * nd.log_softmax(yhat_linear), axis=0, exclude=True)

def net(X, drop_prob=0.0):
    h1_linear = nd.dot(X, W1) + b1
    h1 = relu(h1_linear)
    h1 = dropout(h1, drop_prob)

    h2_linear = nd.dot(h1, W2) + b2
    h2 = relu(h2_linear)
    h2 = dropout(h2, drop_prob)

    yhat_linear = nd.dot(h2, W3) + b3
    return yhat_linear

def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

def evaluate_accuracy(data_iterator, net):
    numerator = 0.
    denominator = 0.
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx).reshape((-1,784))
        label = label.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        numerator += nd.sum(predictions == label)
        denominator += data.shape[0]
    return (numerator / denominator).asscalar()

epochs = 10
moving_loss = 0.
learning_rate = .001

for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx).reshape((-1,784))
        label = label.as_in_context(ctx)
        label_one_hot = nd.one_hot(label, 10)
        with autograd.record():
            output = net(data, drop_prob=.5)
            loss = softmax_cross_entropy(output, label_one_hot)
        loss.backward()
        SGD(params, learning_rate)
        if i == 0:
            moving_loss = nd.mean(loss).asscalar()
        else:
            moving_loss = .99 * moving_loss + .01 * nd.mean(loss).asscalar()
    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))

