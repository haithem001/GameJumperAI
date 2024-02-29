from matplotlib import pyplot
import torch
import random
import numpy as np
from collections import deque
from GUI1 import Game, Direction
from model import Linear_QNet, QTrainer


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


def get_state(game):
    dir_r = game.direction == Direction.RIGHT
    dir_u = game.direction == Direction.UP
    dir_l = game.direction == Direction.LEFT

    state = [
        dir_l,
        dir_r,
        dir_u,
    ]

    return np.array(state, dtype=int)


class Agent:
    def getIterations(self):
        return self.iter
    def __init__(self):
        self.iter =0
        self.n_ts = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(3, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        '''for state, action, reward, next_state, done in mini_sample:
            self.trainer.train_step(state, action, reward, next_state, done)'''

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_ts
        final_move = [0, 0, 0]
        if random.randint(0, 300) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    highestscore=0
    saved=False
    agent = Agent()
    game = Game()
    while True:

        # get old state
        state_old = get_state(game)

        # get move
        final_move = agent.get_action(state_old)
        # perform move and get new state
        reward, done, score = game.play_step(final_move)

        state_new = get_state(game)

        if record < score and not saved:
            saved=True
            record += score


            agent.model.save()



        if reward >= 10:
            print(reward)


        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:

            # train long memory, plot result
            game.reset()
            agent.n_ts += 1
            agent.model.load()
            agent.train_long_memory()


            print('Game', agent.n_ts, 'Score', score, 'Record:', record,'TotalScore:', total_score)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_ts
            plot_mean_scores.append(mean_score)


            saved=False




if __name__ == '__main__':
    train()
