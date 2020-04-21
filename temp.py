import csv_loader
import numpy as np

train_images, train_labels, test_images, test_labels = csv_loader.load_csv()

x_train, x_test = train_images / 255, test_images / 255  # (60000, 784), (10000, 784)
y_train, y_test = train_labels, test_labels  # (60000, 1), (10000, 1)

# stack together
X = np.vstack((x_train, x_test))  # (70000, 784)
y = np.vstack((y_train, y_test))  # (70000, 1)

# one-hot encoding
digits = 10
examples = y.shape[0]  # 70000
y = y.T  # (1, 70000)
Y_new = np.eye(digits)[y.astype('int32')]
Y_new = Y_new.T.reshape(digits, examples)  # (10, 70000)

# split the training and testing sets
m = 60000  # number of training data
m_test = X.shape[0] - m  # number of testing data, 10000
X_train, X_test = X[:m].T, X[m:].T  # (784, 60000), (784, 10000)
Y_train, Y_test = Y_new[:, :m], Y_new[:, m:]  # (10, 60000), (10, 10000)

# Shuffle the training set
shuffle_index = np.random.permutation(m)
X_train, Y_train = X_train[:, shuffle_index], Y_train[:, shuffle_index]


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


np.random.seed(138)

# hyperparameters
n_x = X_train.shape[0]  # 784
n_h = 64
learning_rate = 4
beta = .9
batch_size = 128
batches = -(-m // batch_size)

# initialization
params = {
    "W1": np.random.randn(n_h, n_x) * np.sqrt(1 / n_x),
    "b1": np.zeros((n_h, 1)) * np.sqrt(1 / n_x),
    "W2": np.random.randn(digits, n_h) * np.sqrt(1 / n_h),
    "b2": np.zeros((digits, 1)) * np.sqrt(1 / n_h)
}

# initialize momentum
V_dW1 = np.zeros(params["W1"].shape)
V_db1 = np.zeros(params["b1"].shape)
V_dW2 = np.zeros(params["W2"].shape)
V_db2 = np.zeros(params["b2"].shape)

# train
for i in range(9):

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

    cache = feed_forward(X_train, params)
    train_cost = compute_loss(Y_train, cache["A2"])
    cache = feed_forward(X_test, params)
    test_cost = compute_loss(Y_test, cache["A2"])
    print("Epoch {}: training cost = {}, test cost = {}".format(i + 1, train_cost, test_cost))

print("Done.")
