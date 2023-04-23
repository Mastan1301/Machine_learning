# Training a perceptron to play the Cart-Pole game

import gym
import random
import tensorflow as tf
import numpy as np
from statistics import mean, median
from collections import Counter

LR = 1e-3
env = gym.make('CartPole-v1')
env.reset()
goal_steps = 500
score_requirement = 50
initial_games = 10000

# Make some random moves to create the dataset
def some_random_games_first():
    for episode in range(5):
        env.reset()
        for t in range(goal_steps):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                break

def initial_population():
    training_data = []
    scores = []
    accepted_scores = []
    for _ in range(initial_games):
        score = 0
        game_memory = []
        prev_observation = []
        for _ in range(goal_steps):
            action = random.randrange(0, 2)
            observation, reward, done, info = env.step(action)

            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])
            
            prev_observation = observation
            score += reward
            if done:
                break

        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                # convert to one-hot
                if data[1] == 1:
                    output = [0, 1]
                elif data[1] == 0:
                    output = [1, 0]

                training_data.append([data[0], output])

        env.reset()
        scores.append(score)

    # training_data_save = np.array(training_data)
    # np.save('../data/saved.npy', training_data_save)

    # print('Average accepted score:', mean(accepted_scores))
    # print('Median accepted score:', median(accepted_scores))
    # print(Counter(accepted_scores))

    return training_data

def neural_network_model(input_size):
    model = tf.keras.models.Sequential([
                tf.keras.layers.InputLayer(input_shape=(input_size)),

                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.2),

                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dropout(0.2),

                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dropout(0.2),

                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.2),

                tf.keras.layers.Dense(2, activation='softmax'),
            ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )

    return model

def train_model(training_data, model=None):
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]))
    y = np.array([i[1] for i in training_data])

    if model is None:
        model = neural_network_model(input_size=len(X[0]))
    
    model.fit(x=X, y=y, epochs=3,)

    return model

training_data = initial_population()
model = train_model(training_data)

scores = []
choices = []
for each_game in range(10):
    score = 0
    game_memory = []
    prev_observation = []
    env.reset()
    for _ in range(goal_steps):
        env.render()
        if len(prev_observation) == 0:
            action = random.randrange(0, 2)
        else:
            action = np.argmax(model.predict(prev_observation.reshape(-1, len(prev_observation)))[0])

        choices.append(action)
        new_observation, reward, done, info = env.step(action)
        prev_observation = new_observation
        game_memory.append([new_observation, action])
        score += reward

        if done:
            break

    scores.append(score)

print('Average score', mean(scores))
print('Choice 0: {}, Choice 1: {}'.format(choices.count(0) / len(choices), choices.count(1) / len(choices)))
