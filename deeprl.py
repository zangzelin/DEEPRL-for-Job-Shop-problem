# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
from keras.utils import np_utils
from keras.layers import Dropout, Conv2D, MaxPooling2D, Input, Dense, Flatten, concatenate
from keras.models import Model
import keras
import TestTheOutput
import tensorflow as tf

EPISODES = 5000


def importdata(dataset):
    # ?his function import the data from the database(csv file)
    # And return the feature and the label of the problem
    # 'dataset' is the name of the file
    # 'train_feature, train_label, test_feature, test_label'
    # is training data
    # 'inputnum' is number of input data
    # 'nb_classes' is the number of class

    # load the data
    data = np.loadtxt('./data/'+dataset,
                      dtype=float, delimiter=',')
    features = data[:, 1:-2]
    labels = data[:, -2]
    nb_classes = int(max(labels))+1
    inputnum = features.shape[1]

    # devide the input data, devide it to train data, test_data and left data
    datashape = data.shape
    num_train = int(datashape[0]*0.6)
    num_test = int(datashape[0]*0.2)

    train_feature = features[:num_train, :]*0.999
    train_label = labels[:num_train]
    test_feature = features[num_train:num_test+num_train, :]*0.999
    test_label = labels[num_train:num_test+num_train]

    # reshape the data
    train_feature = train_feature.reshape(train_feature.shape[0], inputnum)
    test_feature = test_feature.reshape(test_feature.shape[0], inputnum)
    train_label = train_label.reshape(train_label.shape[0], 1)
    test_label = test_label.reshape(test_label.shape[0], 1)

    # translate the label data into onehot shape
    train_label = np_utils.to_categorical(train_label, nb_classes)
    test_label = np_utils.to_categorical(test_label, nb_classes)

    return train_feature, train_label, test_feature, test_label, inputnum, nb_classes


def creatmodel_ann(inputnum, nb_classes, layer=15):
    # This function is used to create the ann model in keras
    # 'inputnum' is number of input data
    # 'nb_classes' is the number of class
    # Layer is the number of the hidden ann layers
    # The output model is the created model used to train
    # See http://keras-cn.readthedocs.io/en/latest/models/model/

    sess = tf.InteractiveSession()
    input1 = Input(shape=(inputnum, ), name='i1')
    mid = Dense(200, activation='relu', use_bias=True)(input1)
    for i in range(layer):
        mid = Dense(150, activation='relu', use_bias=True)(mid)
    mid = Dense(150, activation='relu', use_bias=True)(mid)
    mid = Dense(150, activation='relu', use_bias=True)(mid)
    mid = Dense(100, activation='relu', use_bias=True)(mid)
    mid = Dense(80, activation='relu', use_bias=True)(mid)
    mid = Dropout(0.25)(mid)
    mid = Dense(50, activation='relu', use_bias=True)(mid)
    mid = Dropout(0.25)(mid)
    mid = Dense(20, activation='relu', use_bias=True)(mid)
    out = Dense(nb_classes, activation='softmax',
                name='out', use_bias=True)(mid)

    model = Model(input1, out)
    sgd = keras.optimizers.SGD(
        lr=0.11, momentum=0.0, decay=0.0, nesterov=False)

    model.compile(
        optimizer=sgd,
        loss='mean_squared_error',
        metrics=['accuracy'],
    )

    return model


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        sess = tf.InteractiveSession()
        input1 = Input(shape=(136, ), name='i1')
        mid = Dense(200, activation='relu', use_bias=True)(input1)
        for i in range(15):
            mid = Dense(150, activation='relu', use_bias=True)(mid)
        mid = Dense(150, activation='relu', use_bias=True)(mid)
        mid = Dense(150, activation='relu', use_bias=True)(mid)
        mid = Dense(100, activation='relu', use_bias=True)(mid)
        mid = Dense(80, activation='relu', use_bias=True)(mid)
        mid = Dropout(0.25)(mid)
        mid = Dense(50, activation='relu', use_bias=True)(mid)
        mid = Dropout(0.25)(mid)
        mid = Dense(20, activation='relu', use_bias=True)(mid)
        out = Dense(nb_classes, activation='softmax',
                    name='out', use_bias=True)(mid)

        model = Model(input1, out)
        sgd = keras.optimizers.SGD(
            lr=0.11, momentum=0.0, decay=0.0, nesterov=False)

        model.compile(
            optimizer=sgd,
            loss='mean_squared_error',
            metrics=['accuracy'],
        )
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                # a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
                # target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def GetState(train_feature, train_label):

    randint = np.random.randint(38400//64)
    stateout = train_feature[randint*64:randint*64+64]
    keyout = train_label[randint*64:randint*64+64]
    return stateout, keyout


if __name__ == "__main__":

    layerofann = 15

    # import the the data
    path_of_machine = './data/machineco_traindata_m=8_n=8_' + \
        'timelow=6_timehight=30_numofloop=1000.csv'
    dataset = 'featureandlable_traindata_m=8_n=8_timelow=6_timehight=30_numofloop=1000.csv'
    T = np.loadtxt('./data/pssave_traindata_m=8_n=8_' +
                   'timelow=6_timehight=30_numofloop=1000.csv', delimiter=',')
    m = 8
    n = 8

    train_feature, train_label, test_feature, test_label, inputnum, nb_classes = importdata(
        dataset)
    model = creatmodel_ann(inputnum, nb_classes, layer=15)

    # env = gym.make('CartPole-v1')
    # state_size = env.observation_space.shape[0]
    # action_size = env.action_space.n
    # agent = DQNAgent(136, 8)
    # agent.load("./save/cartpole-ddqn.h5")
    done = False
    batch_size = 32

    for e in range(EPISODES): 
        # state = env.reset()
        
        stateout1, keyout1 = GetState(train_feature, train_label)
        predictout1 = model.predict(stateout1)
        Fit1 = TestTheOutput.LineUpTheSolution(
            predictout1, path_of_machine, m, n, T, 0)

        stateout2, keyout2 = GetState(train_feature, train_label)
        predictout2 = model.predict(stateout2)
        Fit2 = TestTheOutput.LineUpTheSolution(
            predictout2, path_of_machine, m, n, T, 0)
        
        if Fit1 < Fit2:
            model.fit(stateout1,keyout1)
        else:
            model.fit(stateout2,keyout2)

        
        
        print(Fit1)
