# import numpy as np
#
#
# class Network(object):
#
#     def __init__(self, sizes):
#         self.num_layers = len(sizes)
#         self.sizes = sizes
#         self.biases = [np.zeros((y, 1)) for y in sizes[1:]]
#         self.weights = [np.random.randn(y, x) * np.sqrt(1 / x) for x, y in zip(sizes[:-1], sizes[1:])]
#
#     def feedforward(self, a):
#         for b, w in zip(self.biases, self.weights):
#             a = sigmoid(np.dot(w, a) + b)
#         return a
#
#     def train(self, X_train, Y_train, epochs, batch_size, learning_rate, beta, test_data=None):
#         train_size = X_train.shape[1]    # 60000
#         if test_data:
#             X_test = test_data[0]
#             Y_test = test_data[1]
#             test_size = X_test.shape[1]  # 10000
#         for i in range(epochs):
#
#             permutation = np.random.permutation(X_train.shape[1])
#             X_train_shuffled = X_train[:, permutation]
#             Y_train_shuffled = Y_train[:, permutation]
#
#             mini_batches = [(X_train_shuffled[:, k:k+batch_size], Y_train_shuffled[:, k:k+batch_size])
#                             for k in range(0, train_size, batch_size)]
#             for mini_batch in mini_batches
#                 self.update_mini_batch(mini_batch, learning_rate, beta)
#
#             if test_data:
#                 print("Result in test set: Epoch {} : {} / {}".format(i, self.evaluate(X_test, Y_test), test_size))
#             else:
#                 print("Result in train set: Epoch {} : {} / {}".format(i, self.evaluate(X_train_shuffled, Y_train_shuffled), train_size))
#
#
#
#
#
#
#
# def sigmoid(z):
#     return 1 / (1 + np.exp(-z))
#
#
# # Use cross-entropy for cost function
# def compute_loss(Y, Y_hat):
#     L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
#     m = Y.shape[1]
#     return -(1 / m) * L_sum