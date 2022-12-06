import collections
import typing

from keras.models import Sequential, save_model, load_model
from keras.layers import *
from collections import deque
import numpy as np
import random


# Deep Q Learning Agent + Maximin
#
# This version only provides only value per input,
# that indicates the score expected in that state.
# This is because the algorithm will try to find the
# best final state for the combinations of possible states,
# in constrast to the traditional way of finding the best
# action for a particular state.
from matris import VISIBLE_MATRIX_HEIGHT, MATRIX_WIDTH

_field_names = [
    "state",
    "action",
    "reward",
    "next_state",
    "done"
]
Experience = collections.namedtuple("Experience", field_names=_field_names)


class PrioritizedExperienceReplayBuffer:
    """Fixed-size buffer to store priority, Experience tuples."""

    def __init__(self,
                 buffer_size: int,
                 alpha: float = 0.0,
                 random_state: np.random.RandomState = None) -> None:
        """
        Initialize an ExperienceReplayBuffer object.

        Parameters:
        -----------
        buffer_size (int): maximum size of buffer
        batch_size (int): size of each training batch
        alpha (float): Strength of prioritized sampling. Default to 0.0 (i.e., uniform sampling).
        random_state (np.random.RandomState): random number generator.

        """
        self._buffer_size = buffer_size
        self._buffer_length = 0  # current number of prioritized experience tuples in buffer
        self._buffer = np.empty(self._buffer_size, dtype=[("priority", np.float32), ("experience", Experience)])
        self._alpha = alpha
        self._random_state = np.random.RandomState() if random_state is None else random_state

    def __len__(self) -> int:
        """Current number of prioritized experience tuple stored in buffer."""
        return self._buffer_length

    @property
    def alpha(self):
        """Strength of prioritized sampling."""
        return self._alpha

    @property
    def buffer_size(self) -> int:
        """Maximum number of prioritized experience tuples stored in buffer."""
        return self._buffer_size

    def add(self, experience: Experience) -> None:
        """Add a new experience to memory."""
        priority = 1.0 if self.is_empty() else self._buffer["priority"].max()
        if self.is_full():
            if priority > self._buffer["priority"].min():
                idx = self._buffer["priority"].argmin()
                self._buffer[idx] = (priority, experience)
            else:
                pass  # low priority experiences should not be included in buffer
        else:
            self._buffer[self._buffer_length] = (priority, experience)
            self._buffer_length += 1

    def is_empty(self) -> bool:
        """True if the buffer is empty; False otherwise."""
        return self._buffer_length == 0

    def is_full(self) -> bool:
        """True if the buffer is full; False otherwise."""
        return self._buffer_length == self._buffer_size

    def sample(self, batch_size: int, beta: float) -> typing.Tuple[np.array, np.array, np.array]:
        """Sample a batch of experiences from memory."""
        # use sampling scheme to determine which experiences to use for learning
        ps = self._buffer[:self._buffer_length]["priority"]
        sampling_probs = ps ** self._alpha / np.sum(ps ** self._alpha)
        idxs = self._random_state.choice(np.arange(ps.size),
                                         size=batch_size,
                                         replace=True,
                                         p=sampling_probs)

        # select the experiences and compute sampling weights
        experiences = self._buffer["experience"][idxs]
        weights = (self._buffer_length * sampling_probs[idxs]) ** -beta
        normalized_weights = weights / weights.max()

        return idxs, experiences, normalized_weights

    def update_priorities(self, idxs: np.array, priorities: np.array) -> None:
        """Update the priorities associated with particular experiences."""
        self._buffer["priority"][idxs] = priorities


class DQNAgent:
    '''Deep Q Learning Agent + Maximin
    Args:
        state_size (int): Size of the input domain
        mem_size (int): Size of the replay buffer
        discount (float): How important is the future rewards compared to the immediate ones [0,1]
        epsilon (float): Exploration (probability of random values given) value at the start
        epsilon_min (float): At what epsilon value the agent stops decrementing it
        epsilon_stop_episode (int): At what episode the agent stops decreasing the exploration variable
        n_neurons (list(int)): List with the number of neurons in each inner layer
        activations (list): List with the activations used in each inner layer, as well as the output
        loss (obj): Loss function
        optimizer (obj): Otimizer used
        replay_start_size: Minimum size needed to train
    '''

    def __init__(self, state_size, mem_size=10000, discount=0.95,
                 epsilon=1, epsilon_min=0, epsilon_stop_episode=500, replace_every=10,
                 n_neurons=[32, 32], activations=['relu', 'relu', 'linear'],
                 loss='mse', optimizer='adam', replay_start_size=None):

        assert len(activations) == len(n_neurons) + 1

        self.state_size = state_size
        # self.memory = deque(maxlen=mem_size)
        self.memory = PrioritizedExperienceReplayBuffer(buffer_size=mem_size, alpha=0.0)
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / epsilon_stop_episode
        self.n_neurons = n_neurons
        self.activations = activations
        self.loss = loss
        self.optimizer = optimizer
        if not replay_start_size:
            replay_start_size = mem_size / 2
        self.replay_start_size = replay_start_size
        self.replace_every = replace_every
        self.model = self._build_model()
        self.baseline = self._build_model()

    def _build_model(self):
        '''Builds a Keras deep neural network model'''
        model = Sequential()
        model.add(Dense(self.n_neurons[0], input_dim=self.state_size, activation=self.activations[0], kernel_initializer='random_normal'))

        for i in range(1, len(self.n_neurons)):
            model.add(Dense(self.n_neurons[i], activation=self.activations[i], kernel_initializer='random_normal'))

        model.add(Dense(1, activation=self.activations[-1], kernel_initializer='random_normal'))

        model.compile(loss=self.loss, optimizer=self.optimizer)

        return model

    def add_to_memory(self, current_state, best_action, next_state, reward, done):
        '''Adds a play to the replay memory buffer'''
        self.memory.add(Experience(current_state, best_action, reward, next_state, done))

    def random_value(self):
        '''Random score for a certain action'''
        return random.random()

    def predict_value(self, state):
        '''Predicts the score for a certain state'''
        return self.model.predict(state, verbose=0)[0]

    def act(self, state):
        '''Returns the expected score of a certain state'''
        # state = np.reshape(state, [1, self.state_size])
        state = np.reshape(state, [1, 4])

        if random.random() <= self.epsilon:
            return self.random_value()
        else:
            return self.predict_value(state)

    def best_state(self, states):
        '''Returns the best state for a given collection of states'''
        max_value = None
        best_state = None

        if random.random() <= self.epsilon:
            return random.choice(list(states))
        else:
            for state in states:
                value = self.predict_value(np.reshape(state, [1, 4]))
                if not max_value or value > max_value:
                    max_value = value
                    best_state = state

        return best_state

    def train(self, batch_size=32, epochs=1, episode=0):
        '''Trains the agent'''
        n = len(self.memory)

        if n >= self.replay_start_size and n >= batch_size:

            # batch = random.sample(self.memory, batch_size)
            idx, batch, _ = self.memory.sample(batch_size=batch_size, beta=1.0)
            states = np.array([x[0] for x in batch])
            next_states = np.array([x[3] for x in batch])
            qs = [x[0] for x in self.model.predict(states)]
            next_qs = [x[0] for x in self.baseline.predict(next_states)]

            errors = [abs(qs[i] - next_qs[i]) for i in range(len(qs))]
            self.memory.update_priorities(idx, errors)

            x = []
            y = []

            for i, (state, _, reward, _, done) in enumerate(batch):
                if not done:
                    new_q = reward + self.discount * next_qs[i]
                else:
                    new_q = reward

                x.append(state)
                y.append(new_q)

            # Fit the model to the given values
            self.model.fit(np.array(x), np.array(y), batch_size=batch_size, epochs=epochs, verbose=0)
            if episode % self.replace_every == 0:
                self.baseline.set_weights(self.model.get_weights())
                self.model.save('checkpoint.pt')

            # Update the exploration variable
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay
