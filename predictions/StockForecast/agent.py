from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam

import numpy as np
import random
from collections import deque

class Agent:
	def __init__(self, state_size, is_eval=False, model_name=""):
		"""
		Initialization of the Agent
		:param state_size: WINDOW_LENGTH, used for the input dimension of the model
		:param is_eval: bool variable to determine if we are using a saved model or not
		:param model_name: name of the model in the directory models/
		"""
		self.state_size = state_size  # normalized previous days
		self.action_size = 3  # sit, buy, sell
		self.memory = deque(maxlen=1000)
		self.inventory = []
		self.model_name = model_name
		self.is_eval = is_eval

		# RL variables
		self.gamma = 0.95
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995

		self.model = load_model("amd_models/" + model_name) if is_eval else self.model()

	def model(self):
		"""
		Creation of the ANN using Tensorflow
		"""

		model = Sequential()
		model.add(Dense(units=64, input_dim=self.state_size, activation="relu"))
		model.add(Dense(units=32, activation="relu"))
		model.add(Dense(units=8, activation="relu"))
		model.add(Dense(self.action_size, activation="linear"))
		model.compile(loss="mse", optimizer=Adam(lr=0.001))

		return model

	def act(self, state):
		"""
		Function where the model outputs a prediction given the state
		:param state: numpy array representation of the current stae to be fed into the network
		:return: Value of 	0 : Hold		1 : Buy 	2 : Sell
		"""

		if not self.is_eval and np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)

		options = self.model.predict(state)
		return np.argmax(options[0])


	def replay(self, batch_size):
		"""
		Function where the training takes place
		Uses accumulated data from the memory to fit the agent's model
		:param batch_size:
		"""

		mini_batch = []
		l = len(self.memory)

		for i in range(l - batch_size + 1, l):
			mini_batch.append(self.memory[i])

		for state, action, reward, next_state, done in mini_batch:
			target = reward
			if not done:
				target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

			target_f = self.model.predict(state)
			target_f[0][action] = target
			self.model.fit(state, target_f, epochs=1, verbose=0)

		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

	def memorize(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))
