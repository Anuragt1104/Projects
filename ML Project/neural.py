import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import timedelta
from contextlib import redirect_stdout

from sklearn.neural_network import MLPClassifier


class NeuralNetwork():
    def __init__(self, loss, activation, batch_size, n, r, hidden_layers) -> None:
        self.loss_type = loss
        self.batch_size = batch_size
        self.n = n
        self.r = r
        self.hidden_layers = hidden_layers

        if loss == 'MSE':
            self.loss = self.MSE
        else:
            self.loss = self.BCE

        if activation == 'sigmoid':
            self.activation = self.sigmoid
            self.activation_grad = self.sigmoid_grad
        else:
            self.activation = self.ReLU
            self.activation_grad = self.ReLU_grad

        self.weights = []
        self.bias = []
        self.net = list([0 for _ in range(len(hidden_layers)+1)])
        self.out = list([0 for _ in range(len(hidden_layers)+1)])
        self.initialize_weights()

        self.grads = list([0 for _ in range((len(hidden_layers))+1)])
        self.bias_grads = list([0 for _ in range((len(hidden_layers))+1)])

    ## Activation functions
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def sigmoid_grad(self, y):
        '''
        Assuming y = sigmoid(x)
        This functions dy/dx
        '''
        return y*(1-y)

    def ReLU(self, z):
        z[z<0] = 0
        return z

    def ReLU_grad(self, y):
        '''
        Assuming y = ReLU(x)
        This returns dy/dx with subgradient at x=0 being taken as 0.
        '''
        y[y>0] = 1
        return y

    ## Loss functions
    def MSE(self, y, o):
        m = y.shape[0]

        j = np.sum((y-o)**2)/(2*m)
        return j

    def BCE(self, y, o):
        m = y.shape[0]

        j = -(np.sum(np.log(o[y==1]))+np.sum(np.log(np.ones((o[y==0]).shape)-o[y==0])))/m
        return j
    
    ## NN code
    def initialize_weights(self):
        self.weights = []
        
        self.weights.append(np.random.randn(self.n, self.hidden_layers[0])*0.1)
        self.bias.append(np.random.randn(1, self.hidden_layers[0])*0.1)
        for i in range(len(self.hidden_layers)-1):
            self.weights.append(np.random.randn(self.hidden_layers[i], self.hidden_layers[i+1])*0.1)
            self.bias.append(np.random.randn(1, self.hidden_layers[i+1])*0.1)
        self.weights.append(np.random.randn(self.hidden_layers[-1], self.r)*0.1)
        self.bias.append(np.random.randn(1, self.r)*0.1)


    def forward_prop(self, X):
        self.net[0] = np.dot(X, self.weights[0]) + self.bias[0]
        self.out[0] = self.activation(self.net[0])
        # self.out[0] = np.concatenate((np.ones((self.out[0].shape[0], 1)), self.out[0]), axis=1)
        for i in range(1, len(self.hidden_layers)+1):
            self.net[i] = np.dot(self.out[i-1], self.weights[i]) + self.bias[i]
            if i!=len(self.hidden_layers):
                self.out[i] = self.activation(self.net[i])
                # self.out[i] = np.concatenate((np.ones((self.out[i].shape[0], 1)), self.out[i]), axis=1)
            else:
                self.out[i] = self.sigmoid(self.net[i])

    def back_prop(self, X, y):
        m = y.shape[0]

        if self.loss_type=='MSE':
            delta = (y-self.out[-1])*self.sigmoid_grad(self.out[-1])/m
        else:
            delta = (y-self.out[-1])/m

        self.grads[-1] = -np.dot(np.transpose(self.out[-2]), delta)
        self.bias_grads[-1] = -np.sum(delta, axis=0, keepdims=True)
        for i in range(len(self.hidden_layers)-1, -1, -1):
            delta = np.dot(delta, np.transpose(self.weights[i+1]))*self.activation_grad(self.out[i])
            if i!=0:
                self.grads[i] = -np.dot(np.transpose(self.out[i-1]), delta)
            else:
                self.grads[i] = -np.dot(np.transpose(X), delta)
            self.bias_grads[i] = -np.sum(delta, axis=0, keepdims=True)

    def fit(self, X, y, lr=0.01, lr0=0, eps=0.0001):
        self.initialize_weights()
        # X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        m = y.shape[0]

        r = np.random.permutation(m)
        X, y = X[r], y[r]
    
        l = 0
        epochs = 1
        J = 1
        converged = False

        while not converged:
            costs = []
            while l < m:
                self.forward_prop(X[l:l+self.batch_size, :])
                costs.append(self.loss(y[l:l+self.batch_size, :], self.out[-1]))
                self.back_prop(X[l:l+self.batch_size, :], y[l:l+self.batch_size, :])
                for i in range(len(self.weights)):
                    self.weights[i] = self.weights[i] - (lr+lr0/epochs)*self.grads[i]
                    self.bias[i] = self.bias[i] - (lr+lr0/epochs)*self.bias_grads[i]
                l += self.batch_size
        
            J_new = (np.sum(costs)*self.batch_size/m)
            l = 0
            epochs += 1

            if epochs >= 50:
                # if abs(1 - J_new/J) < eps:
                if abs(J-J_new) < eps:
                    converged = True
            J = J_new
            r = np.random.permutation(m)
            X, y = X[r], y[r]

    def predict(self, X):
        # X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        self.forward_prop(X)
        y_pred = np.zeros(self.out[-1].shape)
        y_pred[np.arange(self.out[-1].shape[0]), np.argmax(self.out[-1], axis=1)] = 1
        
        return y_pred


    def calc_accuracy(self, X, y):
        y_pred = self.predict(X)
        correct = np.count_nonzero(np.argmax(y, axis=1)==np.argmax(y_pred, axis=1))
        total = y_pred.shape[0]

        return correct/total

def normalize(A):
    max_vals = np.max(A, axis=1)
    A = A/(np.outer(np.transpose(max_vals), np.ones((1, A.shape[1]))))

    return A

def one_hot_encode(y, r):
    y_encode = np.zeros((y.shape[0], r))
    y_encode[np.arange(y.shape[0]), y] = 1

    return y_encode


def get_data(file_path):
    dataset = pd.read_csv(file_path, header=None).to_numpy()
    X, y = dataset[:, :-1], dataset[:, -1]
    X = normalize(X)
    y = one_hot_encode(y, 10)

    return (X, y)


def train_basic_model(X_train, y_train, X_test, y_test, filename, OUTPUT_FOLDER_PATH, lr=0.1, lr0=0, eps=0.001, activation='sigmoid', subpart='a'):
    hidden_layers = [5, 10, 15, 20, 25]

    train_scores = []
    test_scores = []
    exec_times = []
    confusion_matrices = []

    for n_hl in hidden_layers:
        start_time = time.monotonic()
        nn_model = NeuralNetwork('MSE', activation, 100, X_train.shape[1], 10, [n_hl])
        nn_model.fit(X_train, y_train, lr, lr0, eps)
        end_time = time.monotonic()

        train_scores.append(nn_model.calc_accuracy(X_train, y_train))
        test_scores.append(nn_model.calc_accuracy(X_test, y_test))
        exec_times.append(timedelta(seconds=(end_time-start_time)).seconds)

        cm = np.zeros((10, 10))
        y_pred = nn_model.predict(X_test)
        for i in range(len(y_test)):
            cm[y_test[i]==1, y_pred[i]==1] += 1
        confusion_matrices.append(cm)

    fig, ax = plt.subplots()
    ax.set_xlabel('Number of units in hidden layer')
    ax.set_ylabel('accuracy')
    ax.set_title('Accuracy vs number of neurons for training and testing sets')
    ax.plot(hidden_layers, train_scores, marker='o', label='train')
    ax.plot(hidden_layers, test_scores, marker='o', label='test')
    ax.legend()
    fig.savefig(os.path.join(OUTPUT_FOLDER_PATH, subpart+'_accuracy_vs_num_units.png'), dpi=300)

    fig, ax = plt.subplots()
    ax.set_xlabel('Number of units in hidden layer')
    ax.set_ylabel('time to train')
    ax.set_title('Time taken to train vs number of hidden units for training and testing sets')
    ax.plot(hidden_layers, exec_times, marker='o', label='time taken')
    # ax.legend()
    fig.savefig(os.path.join(OUTPUT_FOLDER_PATH, subpart+'_time_vs_num_units.png'), dpi=300)

    with open(filename, 'w') as f:
        with redirect_stdout(f):
            for i in range(len(hidden_layers)):
                print('Number of units in hidden layer: ', hidden_layers[i])
                print('Train Accuracy: ', train_scores[i])
                print('Test Accuracy: ', test_scores[i])
                print('Training Time: ', exec_times[i])
                print('Confusion Matrix: \n', confusion_matrices[i])
    

def train_basic_multilayer_model(X_train, y_train, X_test, y_test, filename, OUTPUT_FOLDER_PATH, lr=0.1, lr0=0, eps=0.001, activation='sigmoid', subpart='c'):
    train_scores = []
    test_scores = []
    exec_times = []
    confusion_matrices = []

    start_time = time.monotonic()
    nn_model = NeuralNetwork('MSE', activation, 100, X_train.shape[1], 10, [100, 100])
    nn_model.fit(X_train, y_train, lr, lr0, eps)
    end_time = time.monotonic()

    train_scores.append(nn_model.calc_accuracy(X_train, y_train))
    test_scores.append(nn_model.calc_accuracy(X_test, y_test))
    exec_times.append(timedelta(seconds=(end_time-start_time)).seconds)

    cm = np.zeros((10, 10))
    y_pred = nn_model.predict(X_test)
    for i in range(len(y_test)):
        cm[y_test[i]==1, y_pred[i]==1] += 1
    confusion_matrices.append(cm)

    with open(filename, 'a') as f:
        with redirect_stdout(f):
            for i in range(len(train_scores)):
                print('Activation: ', activation)
                print('Train Accuracy: ', train_scores[i])
                print('Test Accuracy: ', test_scores[i])
                print('Training Time: ', exec_times[i])
                print('Confusion Matrix: \n', confusion_matrices[i])


def train_multilayer_model(X_train, y_train, X_test, y_test, filename, OUTPUT_FOLDER_PATH, lr=0.1, lr0=0, eps=0.001, activation='sigmoid', subpart='a'):
    hidden_layers = [2, 3, 4, 5]
    train_scores = []
    test_scores = []
    exec_times = []
    
    for hl in hidden_layers:
        start_time = time.monotonic()
        nn_model = NeuralNetwork('MSE', activation, 100, X_train.shape[1], 10, [50 for _ in range(hl)])
        nn_model.fit(X_train, y_train, lr, lr0, eps)
        end_time = time.monotonic()

        train_scores.append(nn_model.calc_accuracy(X_train, y_train))
        test_scores.append(nn_model.calc_accuracy(X_test, y_test))
        exec_times.append(timedelta(seconds=(end_time-start_time)).seconds)

    fig, ax = plt.subplots()
    ax.set_xlabel('Number of hidden layers')
    ax.set_ylabel('accuracy')
    ax.set_title('Accuracy vs number of hidden layers for training and testing sets')
    ax.plot(hidden_layers, train_scores, marker='o', label='train')
    ax.plot(hidden_layers, test_scores, marker='o', label='test')
    ax.legend()
    fig.savefig(os.path.join(OUTPUT_FOLDER_PATH, subpart+'_accuracy_vs_num_layers.png'), dpi=300)

    fig, ax = plt.subplots()
    ax.set_xlabel('Number of hidden layers')
    ax.set_ylabel('time to train')
    ax.set_title('Time taken to train vs number of hidden layers for training and testing sets')
    ax.plot(hidden_layers, exec_times, marker='o', label='time taken')
    # ax.legend()
    fig.savefig(os.path.join(OUTPUT_FOLDER_PATH, subpart+'_time_vs_num_layers.png'), dpi=300)

    with open(filename, 'a') as f:
        with redirect_stdout(f):
            print('Activation function: ', activation)
            for i in range(len(hidden_layers)):
                print('Number of hidden layers: ', hidden_layers[i])
                print('Train Accuracy: ', train_scores[i])
                print('Test Accuracy: ', test_scores[i])
                print('Training Time: ', exec_times[i])

            print('Best accuracy is achieved with number of hidden layers = ', hidden_layers[np.argmax(test_scores)])




if __name__=='__main__':
    TRAIN_DATA_PATH = sys.argv[1]
    TEST_DATA_PATH = sys.argv[2]
    OUTPUT_FOLDER_PATH = sys.argv[3]
    QUESTION_PART = sys.argv[4]

    X_train, y_train = get_data(TRAIN_DATA_PATH)
    X_test, y_test = get_data(TEST_DATA_PATH)

    if QUESTION_PART == 'b':
        train_basic_model(X_train, y_train, X_test, y_test, os.path.join(OUTPUT_FOLDER_PATH, 'b.txt'), OUTPUT_FOLDER_PATH, 0.1, 0, 0.001, subpart='b')
    
    if QUESTION_PART == 'c':
        train_basic_model(X_train, y_train, X_test, y_test, os.path.join(OUTPUT_FOLDER_PATH, 'c.txt'), OUTPUT_FOLDER_PATH, 0, 1, 0.0001, subpart='c')
    
    if QUESTION_PART == 'd':
        with open(os.path.join(OUTPUT_FOLDER_PATH, 'd.txt'), 'w') as f:
            f.write('')
        train_basic_multilayer_model(X_train, y_train, X_test, y_test, os.path.join(OUTPUT_FOLDER_PATH, 'd.txt'), OUTPUT_FOLDER_PATH, 0, 0.1, 0.001, activation='relu', subpart='d_relu')
        train_basic_multilayer_model(X_train, y_train, X_test, y_test, os.path.join(OUTPUT_FOLDER_PATH, 'd.txt'), OUTPUT_FOLDER_PATH, 0, 0.1, 0.001, activation='sigmoid', subpart='d_sig')

    if QUESTION_PART == 'e':
        with open(os.path.join(OUTPUT_FOLDER_PATH, 'e.txt'), 'w') as f:
            f.write('')
        train_multilayer_model(X_train, y_train, X_test, y_test, os.path.join(OUTPUT_FOLDER_PATH, 'e.txt'), OUTPUT_FOLDER_PATH, 0, 1.5, 0.0001, activation='sigmoid', subpart='e_sigmoid')
        train_multilayer_model(X_train, y_train, X_test, y_test, os.path.join(OUTPUT_FOLDER_PATH, 'e.txt'), OUTPUT_FOLDER_PATH, 0, 1, 0.0001, activation='relu', subpart='e_relu')

    if QUESTION_PART == 'f':
        nn_model = NeuralNetwork('BCE', 'relu', 100, X_train.shape[1], 10, [50, 50, 50])
        nn_model.fit(X_train, y_train, 0, 1, 0.0001)

        with open(os.path.join(OUTPUT_FOLDER_PATH, 'f.txt'), 'w') as f:
            with redirect_stdout(f):
                print('Using BCE loss:')
                print('Train Accuracy: ', nn_model.calc_accuracy(X_train, y_train))
                print('Test Accuracy: ', nn_model.calc_accuracy(X_test, y_test))

    if QUESTION_PART == 'g':
        sk_nn_model = MLPClassifier(hidden_layer_sizes=(50, 50, 50), solver='sgd', batch_size=100, learning_rate='constant', learning_rate_init=0.1, random_state=0)
        sk_nn_model.fit(X_train, y_train)

        with open(os.path.join(OUTPUT_FOLDER_PATH, 'g.txt'), 'w') as f:
            with redirect_stdout(f):
                # print(sk_nn_model.loss_)
                print('Train Accuracy: ', sk_nn_model.score(X_train, y_train))
                print('Test Accuracy: ', sk_nn_model.score(X_test, y_test))