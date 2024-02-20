import torch
import torch.nn as nn
import numpy as np
import random
from enum import Enum
import pygame

# Constants
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    FIX = 4


class DDQN(object):
    def __init__(self):
        self.target_net = NETWORK(2, len(Direction), 32)  # Adjusted output_dim
        self.eval_net = NETWORK(2, len(Direction), 32)  # Adjusted output_dim

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        self.memory_counter = 0
        self.memory_size = 50000
        self.memory = np.zeros((self.memory_size, 7))  # Adjusted memory size

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995  # Adjusted epsilon decay

        self.batch_size = 64
        self.episode_counter = 0
        self.target_update_freq = 1000  # Adjusted target update frequency

        self.target_net.load_state_dict(self.eval_net.state_dict())

    def memory_store(self, s0, a0, r, s1, done):
        transition = np.concatenate((s0, [a0, r], s1, [done]))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def select_action(self, state):

        if np.random.rand() <= self.epsilon:

            return np.random.choice(len(Direction))
        else:
            state = torch.tensor([state], dtype=torch.float32)
            q_values = self.eval_net(state)
            return torch.argmax(q_values).item()

    def train(self):
        if self.memory_counter < self.batch_size:
            return

        batch_index = np.random.choice(
            min(self.memory_counter, self.memory_size), size=self.batch_size)
        batch_memory = self.memory[batch_index, :]

        batch_s0 = torch.tensor(batch_memory[:, :2], dtype=torch.float32)
        batch_a0 = torch.tensor(batch_memory[:, 2], dtype=torch.long)
        batch_r = torch.tensor(batch_memory[:, 3], dtype=torch.float32)
        batch_s1 = torch.tensor(batch_memory[:, 4:6], dtype=torch.float32)
        batch_done = torch.tensor(batch_memory[:, 6], dtype=torch.float32)

        q_eval = self.eval_net(batch_s0).gather(1, batch_a0.unsqueeze(1))

        with torch.no_grad():
            q_next = self.target_net(batch_s1).detach()
            q_target = batch_r.unsqueeze(1) + (1 - batch_done.unsqueeze(1)) * q_next.max(1)[0].unsqueeze(1)

        loss = self.criterion(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Update target network
        if self.episode_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

    def update_episode_counter(self):
        self.episode_counter += 1


class NETWORK(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(NETWORK, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)


class Game:
    def __init__(self, retries=float('inf')):
        self.retries = retries
        self.step_counter = 0

        pygame.init()
        self.w = 900
        self.h = 900
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('JumpGame')
        self.clock = pygame.time.Clock()
        self.clock.tick(30)
        self.direction = Direction.FIX
        self.visited = {}
        self.D = pygame.Rect(self.w / 2, self.h - 200, 60, 60)
        self.on_tile=False
        self.ListOfThem = [pygame.Rect(self.w / 2 + 100, self.h - 300, 100, 10),
                           pygame.Rect(self.w / 2 - 200, self.h - 400, 100, 10),
                           pygame.Rect(self.w - 30, self.h - 400, 100, 10),
                           pygame.Rect(self.w / 2 + 100, self.h - 595, 100, 10),
                           pygame.Rect(self.w / 2 - 200, self.h - 530, 100, 10),
                           pygame.Rect(600, self.h - 700, 100, 10),
                           pygame.Rect(400, self.h - 700, 100, 10),
                           pygame.Rect(self.w - 30, self.h - 800, 100, 10),
                           pygame.Rect(0, self.h - 700, 100, 10),
                           pygame.Rect(0, self.h - 450, 100, 10),
                           pygame.Rect(self.w / 2 - 20, self.h - 140, 100, 10)]

        self.velocity = 0
        self.on_ground = False
        self.is_jumping = False
        self.jump_height = 15

    def step(self):
        old_y = self.D.y

        self._update_ui()


        if not self.is_jumping:
            self.step_counter += 1
            state = [self.D.x, self.D.y]
            if self.D.y < old_y:
                reward = 0

            else:
                self.visited[self.D.y] = self.visited.get(self.D.y, 0) + 1
                if self.visited[self.D.y] < self.visited[old_y]:
                    self.visited[self.D.y] = self.visited[old_y] + 1
                reward = -self.visited[self.D.y]

            done = True if self.step_counter > self.retries else False
            return state, reward, done

    '''def play_step(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP and not self.is_jumping:
                    self.is_jumping = True
                    self.velocity = -self.jump_height
                    self.on_ground = False

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                    self.direction = Direction.FIX'''

    def reset(self):
        self.D.x = self.w / 2
        self.D.y = self.h - 200 - self.D.h
        self.step_counter = 0
        self.visited = {}
        self.visited[self.D.y] = 1
        state = [self.D.x, self.D.y-20]
        done = False
        return done, state

    def _move(self,action):
        while True:
            self.clock.tick(70)

            pygame.display.update()
            self._update_ui()

            '''Hethi lekhra'''
            old_y = self.D.y
            self.step_counter += 1

            if action == 1 :
                self.D.x += 3
            elif action == 2 :
                self.D.x -= 3
            elif action == 3 and not self.is_jumping:

                self.is_jumping = True
                self.velocity = -self.jump_height
                self.on_ground = False

            if not self.on_ground:
                self.velocity += 0.5
                self.D.y += self.velocity
                for i in self.ListOfThem:
                    self.Collisions(i)

            if old_y not in self.visited:
                self.visited[old_y] = 0

            if self.D.y > old_y:
                reward = 0
            elif(self.D.x < self.ListOfThem[0].x + self.ListOfThem[0].w and
                    self.D.x + self.D.w > self.ListOfThem[0].x and
                    self.D.y < self.ListOfThem[0].y + 1.5 * self.ListOfThem[0].h and
                    self.D.y + self.D.h == self.ListOfThem[0].y) :
                self.visited[self.D.y] = self.visited[old_y] + 1
                reward=-self.visited[self.D.y]
            else :

                reward = 0

            if self.D.x <= 0:
                self.D.x = 0
            elif self.D.x >= self.w - 60:
                self.D.x = self.w - 60

            state = [self.D.x, self.D.y]
            done = True if self.step_counter > self.retries else False

            return state, reward, done

    def _update_ui(self):
        pygame.display.update()

        self.display.fill(BLACK)
        for i in self.ListOfThem:
            pygame.draw.rect(self.display, WHITE, i)
        Show = self.set_text("MOTHER FUCKER", self.w / 2, 40, 20)
        self.display.blit(Show[0], Show[1])

        pygame.draw.rect(self.display, WHITE, self.D)

        pygame.display.flip()

    @staticmethod
    def set_text(string, coordx, coordy, fontSize):
        font = pygame.font.Font('04B_30__.TTF', fontSize)
        text = font.render(string, True, RED)
        textRect = text.get_rect()
        textRect.center = (coordx, coordy)
        return text, textRect

    def Collisions(self, Tile):
        if (self.D.x < Tile.x + Tile.w and
                self.D.x + self.D.w > Tile.x and
                self.D.y < Tile.y + 1.5 * Tile.h and
                self.D.y + self.D.h > Tile.y):
            self.velocity = 0
            self.is_jumping = True
            self.on_tile=True

            if (Tile.y - self.D.y < self.D.h) and not (Tile.x - 55 < self.D.x):
                self.D.x = Tile.x - self.D.w
                self.on_tile=False

            elif (Tile.y - self.D.y < self.D.h) and not (Tile.x > self.D.x - 95):
                self.D.x = Tile.x + Tile.w
                self.on_tile=False

            elif (self.D.y + self.D.h >= Tile.y) and (self.D.y < Tile.y):
                self.D.y = Tile.y - self.D.h
                self.is_jumping = False
                self.on_tile=False
            else:
                self.on_ground = False
                if self.D.y > self.h - 200:
                    self.velocity += 0.5
                    self.D.y += self.velocity


if __name__ == '__main__':
    agent = DDQN()
    env = Game(retries=200)
    num_episode = 100000

    running = True

    for i in range(num_episode):
        if  env.D.y < env.h:
            done, state = env.reset()

            running_reward = 0
            while not done:
                action = agent.select_action(state)
                print(action)
                next_state, reward, done = env._move(action)

                running_reward += reward
                agent.memory_store(state, action, reward, next_state, done)
                agent.train()
                state = next_state

                agent.update_episode_counter()
                print(f'episode: {i}, reward: {running_reward}')

        else:
            done, state = env.reset()
