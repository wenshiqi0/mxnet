from __future__ import print_function
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
import matplotlib.pyplot as plt

mx.random.seed(1)

data_ctx = mx.cpu()
model_ctx = mx.cpu()

num_inputs = 784
num_outputs = 10
num_examples = 60000


def transform(data, label):
    return data.astype(np.float32) / 255, label.astype(np.float32)
mnist_train = gluon.data.vision.MNIST(train=True, transform=transform)
mnist_test = gluon.data.vision.MNIST(train=False, transform=transform)

batch_size = 64
train_data = mx.gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)
test_data = mx.gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)

W = nd.random_normal(shape=(num_inputs, num_outputs), ctx=model_ctx)
b = nd.random_normal(shape=num_outputs, ctx=model_ctx)
params = [W, b]

for param in params:
    param.attach_grad()


def softmax(y_linear):
    exp = nd.exp(y_linear - nd.max(y_linear, axis=1).reshape((-1, 1)))
    norms = nd.sum(exp, axis=1).reshape((-1, 1))
    return exp / norms


def net(X):
    Y = nd.dot(X, W) + b
    yhat = softmax(Y)
    return yhat


def cross_entropy(yhat, y):
    return - nd.sum(y * nd.log(yhat + 1e-6))


def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad


def evaluate_accuracy(data_iterator, net):
    numerator = 0.
    denominator = 0.
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(model_ctx).reshape((-1, 784))
        label = label.as_in_context(model_ctx)
        label_one_hot = nd.one_hot(label, 10)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        numerator += nd.sum(predictions == label)
        denominator += data.shape[0]
    return (numerator / denominator).asscalar()

epochs = 10
learning_rate = .005

for e in range(epochs):
    cumulative_loss = 0
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx).reshape((-1, 784))
        label = label.as_in_context(model_ctx)
        label_one_hot = nd.one_hot(label, 10)
        with autograd.record():
            output = net(data)
            loss = cross_entropy(output, label_one_hot)
        loss.backward()
        SGD(params, learning_rate)
        cumulative_loss += nd.sum(loss).asscalar()
    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %
          (e, cumulative_loss / num_examples, train_accuracy, test_accuracy))


# Define the function to do prediction
def model_predict(net, data):
    output = net(data)
    return nd.argmax(output, axis=1)

# let's sample 10 random data points from the test set
# last_data = mx.gluon.data.DataLoader(mnist_test, 10, shuffle=True)
sample_data = mx.gluon.data.DataLoader(mnist_test, 10, shuffle=True)

for i, (data, label) in enumerate(sample_data):
    data = data.as_in_context(model_ctx)
    print(data.shape)
    im = nd.transpose(data, (1, 0, 2, 3))
    im = nd.reshape(im, (28, 10 * 28, 1))
    imtiles = nd.tile(im, (1, 1, 3))

    plt.imshow(imtiles.asnumpy())
    pred = model_predict(net, data.reshape((-1, 784)))
    print('model predictions are:', pred)
    plt.show()
    break
