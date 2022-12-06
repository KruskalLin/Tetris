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
from matplotlib import pyplot as plt
import pandas as pd

def plot_episode_stats(episode_lengths, episode_rewards, smoothing_window=10, noshow=False):

    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10,5))
    plt.plot(episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    if noshow:
        plt.close()
    else:
        plt.show()

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10,5))
    rewards_smoothed = pd.Series(episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Score (Smoothed)")
    plt.title("Episode Score over Time (Smoothed over window size {})".format(smoothing_window))
    if noshow:
        plt.close()
    else:
        plt.show()

    return fig1, fig2

# Run dqn with Tetris
def DQN():
    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("MaTris")
    env = Game().env(screen)

    episodes = 500
    max_steps = None
    epsilon_stop_episode = 100
    mem_size = 1000000
    discount = 0.1
    batch_size = 512
    epochs = 1
    render_every = 1
    log_every = 50
    replay_start_size = 2000
    train_every = 1
    n_neurons = [32, 32]
    activations = ['relu', 'relu', 'linear']

    agent = DQNAgent(state_size=4,
                     n_neurons=n_neurons, activations=activations,
                     epsilon_stop_episode=epsilon_stop_episode, mem_size=mem_size,
                     discount=discount, replay_start_size=replay_start_size)

    # log_dir = f'logs/tetris-nn={str(n_neurons)}-mem={mem_size}-bs={batch_size}-e={epochs}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    # log = CustomTensorBoard(log_dir=log_dir)

    scores = []
    episode_lengths = []
    episode_rewards = []

    for episode in tqdm(range(episodes)):
        current_state = env.get_current_state()
        done = False
        steps = 0
        rewards = []
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
                rewards.append(reward)
            except GameOver:
                done = True
                # reward += -10000
            if steps > 0:
                agent.add_to_memory(current_state, best_action, next_states[best_action], reward, done)
            current_state = next_states[best_action]

            steps += 1

        total_reward = sum(rewards)

        episode_lengths.append(steps)
        episode_rewards.append(total_reward)
        scores.append(env.get_current_score())
        if done:
            env.reset(screen, render=render)
        # Train
        if episode % train_every == 0:
            agent.train(batch_size=batch_size, epochs=epochs, episode=episode)

        # Logs
        if log_every and episode and episode % log_every == 0:
            avg_score = mean(scores[-log_every:])
            min_score = min(scores[-log_every:])
            max_score = max(scores[-log_every:])

            print("episode: " + str(episode), "avg_score: " + str(avg_score),
                  "min_score: " + str(min_score), "max_score: " + str(max_score))

    plot_episode_stats(episode_lengths, scores)
    # np.save('episode_lengths_10505025.npy', np.array(episode_lengths))
    # np.save('scores_10505025.npy', np.array(scores))


if __name__ == "__main__":
    DQN()