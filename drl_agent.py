import gc
import random
import numpy as np
import tensorflow as tf
from collections import deque

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)
        self.batch_size = 64
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001

        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(self.state_size, input_dim=self.state_size))
        model.add(tf.keras.layers.Dense(self.state_size*2, activation='sigmoid'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), run_eagerly=True)
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        actionBy = None
        action = [0] * self.action_size
        if np.random.rand() <= self.epsilon:
            actionBy = "Random"
            actionIndex = random.randint(0, 3)
        else:
            actionBy = "Agent"
            actionIndex = np.argmax(self.model.predict(state))
        action[actionIndex] = 1
        return action, actionBy

    def train_short_memory(self, state, action, reward, next_state, done):
        sample = [(state, action, reward, next_state, done)]
        self.train(sample)

    def train_long_memory(self):
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)
        else:
            mini_sample = self.memory
        self.train(mini_sample)

    def train(self, mini_sample):
        for state, action, reward, next_state, done in mini_sample:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0, callbacks=DQNAgentCallback())
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class DQNAgentCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        tf.keras.backend.clear_session()