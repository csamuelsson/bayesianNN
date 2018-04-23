##############################################################
# DEPENDENCIES

import collections
import mxnet as mx
import numpy as np
from mxnet import nd, autograd
#from matplotlib import pyplot as plt # funkar ej
from mxnet import gluon

##############################################################
# HYPERPARAMETERS

config = {
    "num_hidden_layers": 4,
    "num_hidden_units": 100,
    "batch_size": 128, # fixed
    "epochs": 10, # fixed
    "learning_rate": 0.001,
    "num_samples": 1,
    "pi": 0.25,
    "sigma_p": 1.0,
    "sigma_p1": 0.75,
    "sigma_p2": 0.01,
    "activation": 'relu',
    "num_principal_components": 200
}

##############################################################
# HARDWARE CONFIGURATIONS
ctx = mx.gpu(0)

##############################################################
# LOAD DATA

def transform(data, label):
    return data.astype(np.float32)/126.0, label.astype(np.float32)

mnist = mx.test_utils.get_mnist()
num_inputs = 784
num_outputs = 10
batch_size = config['batch_size']

train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform),
                                      batch_size, shuffle=True)
test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform),
                                     batch_size, shuffle=False)

num_train = sum([batch_size for i in train_data])
num_batches = num_train / batch_size

##############################################################
# DEFINE NEURAL NETWORK

net = gluon.nn.Sequential()
with net.name_scope():
    for i in range(config['num_hidden_layers']):
        net.add(gluon.nn.Dense(config['num_hidden_units'], activation="relu"))
    net.add(gluon.nn.Dense(num_outputs))

##############################################################
# DEFINE OBJECTIVE FUNCTION

class BBBLoss(gluon.loss.Loss):
    def __init__(self, log_prior="gaussian", log_likelihood="softmax_cross_entropy",
                 sigma_p1=1.0, sigma_p2=0.1, pi=0.5, weight=None, batch_axis=0, **kwargs):
        super(BBBLoss, self).__init__(weight, batch_axis, **kwargs)
        self.log_prior = log_prior
        self.log_likelihood = log_likelihood
        self.sigma_p1 = sigma_p1
        self.sigma_p2 = sigma_p2
        self.pi = pi

    def log_softmax_likelihood(self, yhat_linear, y):
        return nd.nansum(y * nd.log_softmax(yhat_linear), axis=0, exclude=True)

    def log_gaussian(self, x, mu, sigma):
        return -0.5 * np.log(2.0 * np.pi) - nd.log(sigma) - (x - mu) ** 2 / (2 * sigma ** 2)

    def gaussian_prior(self, x):
        sigma_p = nd.array([self.sigma_p1], ctx=ctx)
        return nd.sum(self.log_gaussian(x, 0., sigma_p))

    def gaussian(self, x, mu, sigma):
        scaling = 1.0 / nd.sqrt(2.0 * np.pi * (sigma ** 2))
        bell = nd.exp(- (x - mu) ** 2 / (2.0 * sigma ** 2))

        return scaling * bell

    def scale_mixture_prior(self, x):
        sigma_p1 = nd.array([self.sigma_p1], ctx=ctx)
        sigma_p2 = nd.array([self.sigma_p2], ctx=ctx)
        pi = self.pi

        first_gaussian = pi * self.gaussian(x, 0., sigma_p1)
        second_gaussian = (1 - pi) * self.gaussian(x, 0., sigma_p2)

        return nd.log(first_gaussian + second_gaussian)

    def hybrid_forward(self, F, output, label, params, mus, sigmas, sample_weight=None):
        log_likelihood_sum = nd.sum(self.log_softmax_likelihood(output, label))
        prior = None
        if self.log_prior == "gaussian":
            prior = self.gaussian_prior
        elif self.log_prior == "scale_mixture":
            prior = self.scale_mixture_prior
        log_prior_sum = sum([nd.sum(prior(param)) for param in params])
        log_var_posterior_sum = sum([nd.sum(self.log_gaussian(params[i], mus[i], sigmas[i])) for i in range(len(params))])
        return 1.0 / num_batches * (log_var_posterior_sum - log_prior_sum) - log_likelihood_sum

bbb_loss = BBBLoss(log_prior="scale_mixture", sigma_p1=config['sigma_p1'], sigma_p2=config['sigma_p2'])

##############################################################
# PARAMETER INITIALIZATION

net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)

# one forward propagation to set parameters
for i, (data, label) in enumerate(train_data):
    data = data.as_in_context(ctx).reshape((-1, 784))
    net(data)
    break

weight_scale = .1
rho_offset = -3

# initialize variational parameters; mean and variance for each weight
mus = []
rhos = []

shapes = list(map(lambda x: x.shape, net.collect_params().values()))

for shape in shapes:
    mu = gluon.Parameter('mu', shape=shape, init=mx.init.Normal(weight_scale))
    rho = gluon.Parameter('rho',shape=shape, init=mx.init.Constant(rho_offset))
    mu.initialize(ctx=ctx)
    rho.initialize(ctx=ctx)
    mus.append(mu)
    rhos.append(rho)

variational_params = mus + rhos

raw_mus = list(map(lambda x: x.data(ctx), mus))
raw_rhos = list(map(lambda x: x.data(ctx), rhos))

##############################################################
# OPTIMIZER

trainer = gluon.Trainer(variational_params, 'sgd', {'learning_rate': config['learning_rate']})

##############################################################
# TRAINING LOOP

def sample_epsilons(param_shapes):
    epsilons = [nd.random_normal(shape=shape, loc=0., scale=1.0, ctx=ctx) for shape in param_shapes]
    return epsilons

def softplus(x):
    return nd.log(1. + nd.exp(x))

def transform_rhos(rhos):
    return [softplus(rho) for rho in rhos]

def transform_gaussian_samples(mus, sigmas, epsilons):
    samples = []
    for j in range(len(mus)):
        samples.append(mus[j] + sigmas[j] * epsilons[j])
    return samples

def generate_weight_sample(layer_param_shapes, mus, rhos):
    # sample epsilons from standard normal
    epsilons = sample_epsilons(layer_param_shapes)

    # compute softplus for variance
    sigmas = transform_rhos(rhos)

    # obtain a sample from q(w|theta) by transforming the epsilons
    layer_params = transform_gaussian_samples(mus, sigmas, epsilons)

    return layer_params, sigmas

def evaluate_accuracy(data_iterator, net, layer_params):
    numerator = 0.
    denominator = 0.
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx).reshape((-1, 784))
        label = label.as_in_context(ctx)

        for l_param, param in zip(layer_params, net.collect_params().values()):
            param._data[list(param._data.keys())[0]] = l_param

        output = net(data)
        predictions = nd.argmax(output, axis=1)
        numerator += nd.sum(predictions == label)
        denominator += data.shape[0]
    return (numerator / denominator).asscalar()

epochs = config['epochs']
learning_rate = config['learning_rate']
smoothing_constant = .01
train_acc = []
test_acc = []

for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx).reshape((-1, 784))
        label = label.as_in_context(ctx)
        label_one_hot = nd.one_hot(label, 10)

        with autograd.record():
            # generate sample
            layer_params, sigmas = generate_weight_sample(shapes, raw_mus, raw_rhos)

            # overwrite network parameters with sampled parameters
            for sample, param in zip(layer_params, net.collect_params().values()):
                param._data = sample
                print("param:", param._data.shape)
                print("sample:", sample.shape)
                #print(param._data[0])
                #print(type(param._data[0]))
                #param._data[list(param._data)[0]] = sample

            # forward-propagate the batch
            output = net(data)

            # calculate the loss
            loss = bbb_loss(output, label_one_hot, layer_params, raw_mus, sigmas)

            # backpropagate for gradient calculation
            loss.backward()

        trainer.step(data.shape[0])

        # calculate moving loss for monitoring convergence
        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (curr_loss if ((i == 0) and (e == 0))
                       else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)

    test_accuracy = evaluate_accuracy(test_data, net, raw_mus)
    train_accuracy = evaluate_accuracy(train_data, net, raw_mus)
    train_acc.append(np.asscalar(train_accuracy))
    test_acc.append(np.asscalar(test_accuracy))
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %
          (e, moving_loss, train_accuracy, test_accuracy))

plt.plot(train_acc)
plt.plot(test_acc)
plt.show()