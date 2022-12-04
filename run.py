import pygame

from agent import DQNAgent
from matris import Game, WIDTH, HEIGHT, GameOver
from datetime import datetime
from statistics import mean, median
import random
from tqdm import tqdm

import numpy as np
from keras.callbacks import TensorBoard
# from tensorflow.summary import FileWriter

class CustomTensorBoard(TensorBoard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_model(self, model):
        pass

    def log(self, step, **stats):
        print(stats)

# Run dqn with Tetris
def DQN():
    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("MaTris")
    env = Game().env(screen)

    episodes = 2000
    max_steps = None
    epsilon_stop_episode = 100
    mem_size = 1000000
    discount = 0.9
    batch_size = 512
    epochs = 1
    render_every = 1
    log_every = 50
    replay_start_size = 2000
    train_every = 1
    n_neurons = [128, 128]
    activations = ['relu', 'relu', 'linear']

    agent = DQNAgent(state_size=6,
                     n_neurons=n_neurons, activations=activations,
                     epsilon_stop_episode=epsilon_stop_episode, mem_size=mem_size,
                     discount=discount, replay_start_size=replay_start_size)

    log_dir = f'logs/tetris-nn={str(n_neurons)}-mem={mem_size}-bs={batch_size}-e={epochs}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    log = CustomTensorBoard(log_dir=log_dir)

    scores = []

    for episode in tqdm(range(episodes)):
        current_state = env.get_current_state()
        done = False
        steps = 0

        render = False
        if episode % render_every == 0:
            render = True

        # Game
        while not done and (not max_steps or steps < max_steps):
            env.set_step(steps)
            next_states, punishment = env.get_next_states(steps=steps)
            best_state = agent.best_state(next_states.values())

            best_action = None
            for action, state in next_states.items():
                if (np.array(state) == np.array(best_state)).all():
                    best_action = action
                    break
            reward = punishment[best_action]
            try:
                reward += env.play(best_action[0], best_action[1], render=render)
                # env.play(best_action[0], best_action[1], render=render)
            except GameOver:
                done = True
                reward += -1000000
            if steps > 0:
                agent.add_to_memory(current_state, next_states[best_action], reward, done)
            current_state = next_states[best_action]
            steps += 1

        scores.append(env.get_current_score())
        if done:
            env.reset(screen, render=render)
        # Train
        if episode % train_every == 0:
            agent.train(batch_size=batch_size, epochs=epochs)

        # Logs
        if log_every and episode and episode % log_every == 0:
            avg_score = mean(scores[-log_every:])
            min_score = min(scores[-log_every:])
            max_score = max(scores[-log_every:])

            log.log(episode, avg_score=avg_score, min_score=min_score,
                    max_score=max_score)


if __name__ == "__main__":
    DQN()