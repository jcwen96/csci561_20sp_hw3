import csv_loader
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Use cross-entropy for cost function
def compute_loss(Y, Y_hat):
    L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
    m = Y.shape[1]
    return -(1 / m) * L_sum


def feed_forward(X, params):
    cache = {}
    cache["Z1"] = np.matmul(params["W1"], X) + params["b1"]
    cache["A1"] = sigmoid(cache["Z1"])
    cache["Z2"] = np.matmul(params["W2"], cache["A1"]) + params["b2"]
    cache["A2"] = np.exp(cache["Z2"]) / np.sum(np.exp(cache["Z2"]), axis=0)
    return cache


def back_propagate(X, Y, params, cache, m_batch):
    dZ2 = cache["A2"] - Y  # error at last layer

    # gradients at last layer
    dW2 = (1 / m_batch) * np.matmul(dZ2, cache["A1"].T)
    db2 = (1. / m_batch) * np.sum(dZ2, axis=1, keepdims=True)

    # back propagate trough first layer
    dA1 = np.matmul(params["W2"].T, dZ2)
    dZ1 = dA1 * sigmoid(cache["Z1"]) * (1 - sigmoid(cache["Z1"]))

    # gradients at first layer
    dW1 = (1 / m_batch) * np.matmul(dZ1, X.T)
    db1 = (1 / m_batch) * np.sum(dZ1, axis=1, keepdims=True)

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}


train_images, train_labels, test_images = csv_loader.load_csv()

X_train, X_test = train_images.T / 255, test_images.T / 255  # (784, 60000), (784, 10000)
# one-hot encoding
digits = 10
Y_train = np.eye(digits)[train_labels.T.astype('int32')]
Y_train = Y_train.T.reshape(digits, train_labels.shape[0])  # (10, 60000)

# Shuffle the training set
# np.random.seed(138)
# shuffle_index = np.random.permutation(X_train.shape[1])
# X_train, Y_train = X_train[:, shuffle_index], Y_train[:, shuffle_index]

# hyperparameters
n_1 = X_train.shape[0]  # 784
n_2 = 64
learning_rate = 1
beta = .5
batch_size = 100
batches = -(-X_train.shape[1] // batch_size)

# initialization
params = {
    "W1": np.random.randn(n_2, n_1) * np.sqrt(1 / n_1),
    "b1": np.zeros((n_2, 1)) * np.sqrt(1 / n_1),
    "W2": np.random.randn(digits, n_2) * np.sqrt(1 / n_2),
    "b2": np.zeros((digits, 1)) * np.sqrt(1 / n_2)
}

# initialize momentum
V_dW1 = np.zeros(params["W1"].shape)
V_db1 = np.zeros(params["b1"].shape)
V_dW2 = np.zeros(params["W2"].shape)
V_db2 = np.zeros(params["b2"].shape)

# train
for i in range(20):

    permutation = np.random.permutation(X_train.shape[1])
    X_train_shuffled = X_train[:, permutation]
    Y_train_shuffled = Y_train[:, permutation]

    for j in range(batches):
        # get mini-batch
        begin = j * batch_size
        end = min(begin + batch_size, X_train.shape[1] - 1)
        X = X_train_shuffled[:, begin:end]
        Y = Y_train_shuffled[:, begin:end]
        m_batch = end - begin

        cache = feed_forward(X, params)
        grads = back_propagate(X, Y, params, cache, m_batch)

        V_dW1 = (beta * V_dW1 + (1. - beta) * grads["dW1"])
        V_db1 = (beta * V_db1 + (1. - beta) * grads["db1"])
        V_dW2 = (beta * V_dW2 + (1. - beta) * grads["dW2"])
        V_db2 = (beta * V_db2 + (1. - beta) * grads["db2"])

        params["W1"] = params["W1"] - learning_rate * V_dW1
        params["b1"] = params["b1"] - learning_rate * V_db1
        params["W2"] = params["W2"] - learning_rate * V_dW2
        params["b2"] = params["b2"] - learning_rate * V_db2

    # cache = feed_forward(X_train, params)
    # train_results = np.argmax(cache['A2'], axis=0).reshape(train_labels.shape)
    # train_correct = sum(int(x == y) for x, y in zip(train_results, train_labels))
    # cache = feed_forward(X_test, params)
    # test_results = np.argmax(cache['A2'], axis=0).reshape(test_labels.shape)
    # test_correct = sum(int(x == y) for x, y in zip(test_results, test_labels))
    # print("Epoch {}: train set: {} / {} test set: {} / {}"
    #       .format(i + 1, train_correct, train_labels.shape[0], test_correct, test_labels.shape[0]))

cache = feed_forward(X_test, params)
test_results = np.argmax(cache['A2'], axis=0).reshape(X_test.shape[1], 1)
np.savetxt('test_predictions.csv', test_results, delimiter=',', fmt='%d')
print("Done.")
