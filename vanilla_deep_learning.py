import typing
import numpy as np
import random
import math
import time
from keras.datasets import mnist
from numpy.random import seed

EPSILON = 1e-7

# Whether to apply batch normalization or not.
USE_BATCHNORM = False


# Forward Propagation ###################################################################################

def initialize_parameters(layer_dims: typing.List):
    """Initialize the parameters of the neural network.

    Returns:
        dict. a dictionary with the layers as values.
        for example:
        {
            'W1': [....
                   ....]
            'b1': [....]
        }
    """
    params = {}

    for i in range(1, len(layer_dims)):
        params[f'W{i}'] = np.random.randn(layer_dims[i], layer_dims[i - 1]) * np.sqrt(2. / layer_dims[i - 1])
        params[f'b{i}'] = np.zeros(([layer_dims[i], 1])) * np.sqrt(1. / layer_dims[i - 1])

    return params


def linear_forward(A_prev: np.array, W: np.array, b: np.array):
    """Process the previous layer with the wights of the next one, only the linear part.

    The result of this computation will go into the activation of the current layer.
    Save the received parameters for the chain rule in backward propgation later
    """
    Z = np.matmul(W, A_prev) + b
    linear_cache = {'A_prev': A_prev, 'W': W, 'b': b}

    return Z, linear_cache


def softmax(Z: np.array):
    return np.exp(Z) / np.sum(np.exp(Z), axis=0), Z


def relu(Z: np.array):
    return np.maximum(Z, 0), Z


ACTIVATION_STR_TO_FUNC = {
    'relu': relu,
    'softmax': softmax

}


def linear_activation_forward(A_prev, W, b, activation_func_str):
    activation_func = ACTIVATION_STR_TO_FUNC[activation_func_str]
    Z, cache = linear_forward(A_prev, W, b)

    # Update the activation value (possibly after batch normalization)
    cache.update({'Z': Z})
    A, _ = activation_func(Z)
    cache.update({'A': A})

    return A, cache


def apply_batchnorm(A):
    """Apply batch normalization on the output of a layer"""
    A = A.T
    m = A.shape[0]
    A_mean = np.sum(A, axis=0) / m
    A_var = np.sum((A - A_mean) ** 2, axis=0) / m
    A_bn = (A - A_mean) / np.sqrt((A_var + EPSILON))
    return A_bn.T


def L_model_forward(X, parameters, use_batchnorm):
    if len(parameters) % 2 != 0:
        raise ValueError(f"Parameters length should be even because there is W and b for every layer, "
                         "got {len(parameters)} instead}")

    caches = []
    layers_count = int(len(parameters) / 2)

    # Perform the forward propagation for all of the middle-hidden layers
    A = X
    for layer_idx in range(1, layers_count):
        W = parameters[f'W{layer_idx}']
        b = parameters[f'b{layer_idx}']
        A, cache = linear_activation_forward(A, W, b, 'relu')

        if use_batchnorm:
            A = apply_batchnorm(A)
            cache['A'] = A

        caches.append(cache)

    # Perform the output layer - softmax
    W = parameters[f'W{layers_count}']
    b = parameters[f'b{layers_count}']
    # Don't apply batchnorm on the output of the last layer
    AL, cache = linear_activation_forward(A, W, b, 'softmax')
    cache.update({'A': AL})
    caches.append(cache)

    return AL, caches


def compute_cost(AL, Y):
    """Compute cross-entropy cost

    Args:
        AL(n_classes,n_examples)
        Y(n_classes, n_examples)
    """
    m = Y.shape[1]

    # Select the likelihoods predicted for the true labels for each sample
    log_likelihood = np.multiply(np.log(AL), Y)

    return (-1. / m) * np.sum(log_likelihood)


# Backward Propagation ##################################################################################

def linear_backward(dZ, cache):
    A_prev = cache['A_prev']
    W = cache['W']

    m = dZ.shape[1]
    dA_prev = np.matmul(W.T, dZ)
    dW = (1. / m) * np.matmul(dZ, A_prev.T)
    db = (1. / m) * np.sum(dZ, axis=1, keepdims=True)

    return {
        'dA_prev': dA_prev,
        'dW': dW,
        'db': db,
    }


def relu_backward(dA, cache):
    """dA is basically W*dZ"""
    Z = cache['Z']

    dg = Z[:]
    dg[Z <= 0] = 0
    dg[Z > 0] = 1

    dZ = np.multiply(dA, dg)
    return dZ


def only_softmax_backward(dA, cache):
    """Compute the derivative of softmax only - independently from Cross Entropy loss function

    NOTE: this function is not used because the combination softmax and CE is more efficient.
    """
    A = cache['A']

    # Calculate the partial derivatives of each of the softmax output with respect
    # to each of the z_i variables
    m = A.shape[1]

    # for each instance, compute its jacobian and then its dZ
    dZ = np.zeros((A.shape[0], m))
    for m_i in range(m):
        # Compute the jacobian of the instance softmax
        jacobian = np.zeros((A.shape[0], A.shape[0]))
        a = A[:, m_i]
        for i in range(len(jacobian)):
            for j in range(len(jacobian)):
                if i == j:
                    jacobian[i][j] = a[i] * (1 - a[i])
                else:
                    jacobian[i][j] = -a[i] * a[j]

        # Compute dZ of the instance
        for j in range(len(jacobian)):
            dZ[j][m_i] = sum([dA[k, m_i] * jacobian[k, j]
                              for k in range(len(jacobian))])

    return dZ


def softmax_ce_backward(dA, cache):
    """Softmax and Cross Entropy backward propagation

    This function combines softmax and CE in order to compute the dZ of the last layer of the network and the
    loss 'layer'.
    This function assumes that dA contains Y (the ground truth labels) in order to compute the A-Y function.
    """
    # Z = cache['Z']
    # TODO: maybe save A in cache instead of re-computing?
    # A = softmax(Z)[0]
    A = cache['A']

    # This function assumes that dA contains Y (the ground truth labels)
    Y = dA

    # The derivative of CE loss with respect to Z of the softmax layer is A-Y
    # (where A is the post-softmax activation values and Y is the true labels)
    dZ = A - Y

    return dZ


def softmax_backward(dA, cache):
    return softmax_ce_backward(dA, cache)


ACTIVATION_BACKWARD_STR_TO_FUNC = {
    'relu': relu_backward,
    'softmax': softmax_backward
}


def linear_activation_backward(dA, cache, activation):
    dZ = ACTIVATION_BACKWARD_STR_TO_FUNC[activation](dA, cache)

    return linear_backward(dZ, cache)


def L_model_backward(AL, Y, caches):
    """Compute the gradients for each layer"""
    layer_count = len(caches)
    grads = {}

    # Calculate the derivatives of the loss function in terms of AL - the output of the final layer
    # This is commented out because we are using the combined derivative for both cross entropy and softmax
    # dA = -np.divide(Y, AL)
    # Put Y in dA in order to efficiently combine the derivatives of softmax and Cross Entropy.
    dA = Y

    # Perform the back propagation for the final, softmax layer
    grads["dA{layer_count}"] = dA
    curr_grads = linear_activation_backward(dA, caches[-1], 'softmax')
    grads[f'dW{layer_count}'] = curr_grads['dW']
    grads[f'db{layer_count}'] = curr_grads['db']

    # Perform the backpropagation for all of the middle, RELU layers
    for i in range(layer_count - 1, 0, -1):
        grads["dA{i}"] = curr_grads['dA_prev']
        curr_grads = linear_activation_backward(grads["dA{i}"], caches[i - 1], 'relu')
        grads[f'dW{i}'] = curr_grads['dW']
        grads[f'db{i}'] = curr_grads['db']

    return grads


def update_parameters(parameters, grads, learning_rate):
    if len(parameters) % 2 != 0:
        raise ValueError(f'parameters length should be even, got {len(parameters)} instead')

    n_layers = int(len(parameters) / 2)
    for i in range(1, n_layers + 1):
        parameters[f'W{i}'] -= learning_rate * grads[f'dW{i}']
        parameters[f'b{i}'] -= learning_rate * grads[f'db{i}']

    # NOTE: this is actually redundant since the given parameters is an object which we are changing anyway
    return parameters


# Model #####################################################################################################

def generate_batches(X, Y, batch_size):
    """Generate random batches of the given size from the given instances.

    Args:
        X(n_features, n_samples): The input of the neural network
        batch_size (int): the desired batch size

    Return:
        generator. The batches in an iterable manner (one can execute a for loop on it).
        each element will contain batch from X and its matching elements from Y
    """
    m = X.shape[1]
    indices = list(range(m))
    random.shuffle(indices)
    n_batches = int(math.ceil(m / batch_size))
    for i in range(n_batches):
        X_batch = X[:, indices[i * batch_size:(i + 1) * batch_size]]
        Y_batch = Y[:, indices[i * batch_size:(i + 1) * batch_size]]

        yield X_batch, Y_batch


def train_model(params, X, Y, learning_rate, num_iterations, batch_size):
    """Train the given parameters more times (assuming RELU for all middle layers and softmax for the last one)"""
    iters_count = 0
    costs = []
    while iters_count < num_iterations:
        batches = generate_batches(X, Y, batch_size)
        for X_batch, Y_batch in batches:
            iters_count += 1
            AL, caches = L_model_forward(X_batch, params, use_batchnorm=USE_BATCHNORM)
            grads = L_model_backward(AL, Y_batch, caches)
            params = update_parameters(params, grads, learning_rate)

            if iters_count % 100 == 0:
                iter_cost = compute_cost(AL, Y_batch)
                costs.append(iter_cost)

            if iters_count == num_iterations:
                return params, costs

    return params, costs


def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, batch_size):
    params = initialize_parameters(layers_dims)

    return train_model(params, X, Y, learning_rate, num_iterations, batch_size)


def predict(X, Y, parameters):
    AL, _ = L_model_forward(X, parameters, use_batchnorm=USE_BATCHNORM)
    outputs = np.argmax(AL, axis=0)
    y = np.argmax(Y, axis=0)

    all = len(outputs)
    correct = len(outputs[outputs == y])

    return correct / all


def get_mnist_data():
    val_size = 0.2

    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    # float 0-1
    train_X = train_X.reshape(train_X.shape[0], 784).astype('float32') / 255.
    test_X = test_X.reshape(test_X.shape[0], 784).astype('float32') / 255.

    # 1hot encoding
    y_tr_enc = np.zeros((len(train_y), 10), np.float32)
    y_tr_enc[range(y_tr_enc.shape[0]), train_y] = 1
    y_te_enc = np.zeros((len(test_y), 10), np.float32)
    y_te_enc[range(y_te_enc.shape[0]), test_y] = 1

    # train_val_split
    def train_val_split(train_X, y_enc, val_size):
        tr_sz = np.int(train_X.shape[0] * (1 - val_size))
        X_train, X_val = train_X[:tr_sz], train_X[tr_sz:]
        y_train, y_val = y_enc[:tr_sz], y_enc[tr_sz:]
        return X_train, X_val, y_train, y_val

    X_tr, X_val, y_tr, y_val = train_val_split(train_X, y_tr_enc, val_size)

    # transpose
    X_tr = X_tr.T
    X_val = X_val.T
    y_tr = y_tr.T
    y_val = y_val.T
    X_te = test_X.T
    y_te = y_te_enc.T

    return X_tr, X_val, X_te, y_tr, y_val, y_te


def main():
    # Set the random seed to be a constant
    seed(1)

    # Experiment variables
    layers_dims = [784, 20, 7, 5, 10]
    learning_rate = 0.009
    batch_size = 512
    improvement_threshold = 0.001

    X_tr, X_val, X_te, y_tr, y_val, y_te = get_mnist_data()

    # Train until there is no significant improvement on the validation set ##############################

    # Initialize the model
    model_params, costs = L_layer_model(X_tr, y_tr, layers_dims, learning_rate, 0, batch_size)
    iters_count = 0
    acc = predict(X_val, y_val, model_params)
    print(f'initial accuracy is {round(acc * 100, 1)}')

    # This is just a dummy value for the first time, we need to get into the while loop
    acc_improvement = improvement_threshold
    num_iterations = 100

    start_time = time.time()
    # Train the model until the improvement is less than 1 percent
    while acc_improvement >= improvement_threshold:
        model_params, new_costs = train_model(model_params, X_tr, y_tr, learning_rate, num_iterations, batch_size)
        iters_count += 1
        costs += new_costs

        new_acc = predict(X_val, y_val, model_params)
        print(f'iter {iters_count * num_iterations}, accuracy is {round(new_acc * 100, 1)}, cost is {costs[-1]}')
        if new_acc == 0 or acc == 0:
            # If the results are bad keep training the model
            acc_improvement = improvement_threshold
        else:
            acc_improvement = new_acc - acc

        acc = new_acc

    print(f'training took {time.time() - start_time} seconds')

    # Test the model on the test set
    test_acc = predict(X_te, y_te, model_params)

    # Test the model on the train set
    train_acc = predict(X_tr, y_tr, model_params)

    print(f'final accuracy on train set is {round(train_acc * 100, 1)}')
    print(f'final accuracy on validation set is {round(acc * 100, 1)}')
    print(f'final accuracy on test set is {round(test_acc * 100, 1)}')

    print(f'batch size is {batch_size}')
    print(f'ran {iters_count * num_iterations} iterations')
    print(f'ran {int(iters_count * num_iterations / batch_size)} epochs')


if __name__ == '__main__':
    main()
