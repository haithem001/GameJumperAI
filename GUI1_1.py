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
        self.transition = []
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def memory_store(self, s0, a0, r, s1, done):
        self.transition = np.concatenate((s0, [a0, r], s1, [done]))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = self.transition
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
            batch_index = np.random.choice(
                min(self.memory_counter, self.memory_size), size=self.batch_size)
        else:
            batch_index = np.random.choice(self.memory_counter, size=self.batch_size)

        batch_memory = self.memory[batch_index, :]

        batch_s0 = torch.tensor(batch_memory[:, :2], dtype=torch.float32)
        batch_a0 = torch.tensor(batch_memory[:, 2], dtype=torch.long)
        batch_r = torch.tensor(batch_memory[:, 3], dtype=torch.float32)
        batch_s1 = torch.tensor(batch_memory[:, 4:6], dtype=torch.float32)
        batch_done = torch.tensor(batch_memory[:, 6], dtype=torch.float32)

        q_eval = self.eval_net(batch_s0).gather(1, batch_a0.unsqueeze(1))

        with torch.no_grad():
            q_next = self.target_net(batch_s1)
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
        self.jumpCount = 0
        self.retries = retries
        self.step_counter = 0
        self.reward = 0
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
        self.on_tile = False
        self.ListOfThem = [pygame.Rect(self.w / 2 - 20, self.h - 140, 100, 10),

                           pygame.Rect(self.w / 2 + 170, self.h - 300, 300, 300),
                           pygame.Rect(self.w / 2 + - 50, self.h - 450, 100, 10)

                           ]
        '''self.ListOfThem = [pygame.Rect(self.w / 2 + 150, self.h - 300, 100, 10),
                           pygame.Rect(self.w / 2 - 200, self.h - 470, 100, 10),
                           pygame.Rect(self.w - 100, self.h - 400, 100, 10),
                           pygame.Rect(self.w / 2 + 100, self.h - 595, 100, 10),
                           pygame.Rect(self.w / 2 - 200, self.h - 580, 100, 10),
                           pygame.Rect(600, self.h - 700, 100, 10),
                           pygame.Rect(400, self.h - 700, 100, 10),
                           pygame.Rect(self.w - 30, self.h - 800, 100, 10),
                           pygame.Rect(0, self.h - 700, 100, 10),
                           pygame.Rect(0, self.h - 450, 100, 10),
                           pygame.Rect(self.w / 2 - 20, self.h - 140, 100, 10)]'''

        self.velocity = 0
        self.on_ground = False
        self.is_jumping = False
        self.jump_height = 15
        self.REWARD = 0
        self.Left = False
        self.Right = False

    '''def step(self):
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

        def play_step(self):
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

    def _move(self, action):

        self.reward = 0
        while True:

            self.clock.tick(30)

            pygame.display.update()
            self._update_ui()

            '''Hethi lekhra'''
            self.step_counter += 1

            if action == 1:
                if self.is_jumping :
                    self.Right = True
                    self.Left = False

            elif action == 2:
                if self.is_jumping :
                    self.Right = False
                    self.Left = True
            elif action == 3 and not self.is_jumping:

                self.is_jumping = True
                self.velocity = -self.jump_height
                self.on_ground = False

            if not self.on_ground:
                if self.Left and not self.Right:
                        self.D.x -= self.jump_height/3
                else:
                        self.D.x += self.jump_height/3

                self.velocity += 0.5
                self.D.y += self.velocity
                for i in self.ListOfThem:
                    self.Collisions(i)

            if self.D.left <= 0:
                self.D.left = 0
                self.Left = False

            elif self.D.right >= self.w:
                self.D.right = self.w
                self.Right = False

            if self.D.bottom > env.h - 20:
                self.step_counter = self.retries + 1
                self.reward -= 5 * env.h - 20

            state = [self.D.x, self.D.bottom]

            done = True if self.step_counter > self.retries else False

            return state, self.reward, done

    def Collisions(self, Tile):

        if (self.D.left < Tile.right and self.D.right > Tile.left and
                self.D.top < Tile.bottom and self.D.bottom > Tile.top):

            if self.D.bottom < self.h - 200:
                if self.D.bottom not in self.visited:
                    self.visited[self.D.bottom] = 0
                    self.jumpCount += 1
                    self.reward = 2 * Tile.top

            print(self.REWARD)

            self.velocity = 0
            self.is_jumping = True
            print("heyyyyyy")

            print(self.on_tile)

            if (self.D.bottom > Tile.bottom) and (self.D.top < Tile.top) and (self.D.left < Tile.right - 5 ):
                self.D.left = Tile.right

            if  (self.D.bottom < Tile.top) and  (self.D.right > Tile.left ):
                self.D.right = Tile.left

            if (self.D.bottom > Tile.top) and (self.D.y < Tile.y):
                self.D.bottom = Tile.top
                self.is_jumping = False

            else:
                self.on_ground = False
                if self.D.top > self.h - 200:

                    self.D.y += self.velocity

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

    def set_Record(self):
        self.REWARD = 10

    def reset(self):

        self.visited = {self.D.y: 1}
        self.D.x = self.w / 2 -20
        self.D.y = self.h - 200  # set y-coordinate to a valid ground position
        self.step_counter = 0
        state = [self.D.x, self.D.y]
        done = False
        return done, state


if __name__ == '__main__':
    agent = DDQN()
    R = 600
    env = Game(retries=R)
    num_episode = 1000
    running = True
    env.reward = 0
    reward_saved = 0
    prev_state = [0, 0]
    saved_state = [0, 0]
    agent.memory_store([0, 0], 0, 0, [0, 0], False)
    stored=agent.memory
    running_reward = 0
    for i in range(num_episode):
        done, state = env.reset()
        if running_reward==0:
            agent.memory=stored
        running_reward = 0
        while not done:
            env.jumpCount = 0
            # ACTION NEXTSTATE REWARD DONE STATE
            action = agent.select_action(state)
            next_state, env.reward, done = env._move(action)

            running_reward += env.reward


            agent.memory_store(state, action, running_reward, next_state, done)
            if running_reward > 0:
                stored=agent.memory

            agent.train()
            state = next_state


            print(agent.memory)


            agent.update_episode_counter()
            print(f'ep {i} : running_reward: {running_reward},reward_saved:{reward_saved}')
