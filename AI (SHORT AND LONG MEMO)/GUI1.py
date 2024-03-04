import pygame
from enum import Enum
import numpy as np

BLACK = (0, 0, 0)
White = (255, 255, 255)
Red = (255, 0, 0)

# Initialize Pygame
pygame.init()

# Initialize Pygame font module
pygame.font.init()


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    FIX = 4


class Game:
    def set_text(self, string, coordx, coordy, fontSize):  # Function to set text

        font = pygame.font.Font('04B_30__.TTF', fontSize)
        # (0, 0, 0) is black, to make black text
        text = font.render(string, True, Red)
        textRect = text.get_rect()
        textRect.center = (coordx, coordy)
        return (text, textRect)

    def __init__(self, w=900, h=900):
        self.left = False
        self.right = False
        self.w = w
        self.h = h
        self.reward = 0
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('JumpGame')
        self.clock = pygame.time.Clock()
        self.ListOfThem = [pygame.Rect(300, self.h - 460, 300, 10),
                           pygame.Rect(700, self.h - 240, 200, 10),
                           pygame.Rect(340, self.h - 200, 400, 300),
                           pygame.Rect(0, self.h - 140, 300, 50)
                           ]

        self.ListOfListOfThem = self.ListOfThem.copy()

        self.reset()
        self.rec_iter=200

    def draw_text(self, text, text_col, x, y):
        font = pygame.font.Font("04B_30__.TTF", 24)
        img = font.render(text, True, text_col)
        self.display.blit(img, (x, y))

    def Collisions(self, Tile):
        if (self.D.x < Tile.x + Tile.w and
                self.D.x + self.D.w > Tile.x and
                self.D.y < Tile.y + 1.5 * Tile.h and
                self.D.y + self.D.h > Tile.y):
            self.velocity = 0

            self.is_jumping = True

            if (Tile.y - self.D.y < self.D.h) and not (Tile.x - 55 < self.D.x):
                self.D.x = Tile.x - self.D.w


            elif (Tile.y - self.D.y < self.D.h) and not (Tile.x > self.D.x - Tile.w + 5):
                self.D.x = Tile.x + Tile.w

            elif (self.D.y + self.D.h >= Tile.y) and (self.D.y < Tile.y):
                self.Tile = Tile
                self.D.y = Tile.y - self.D.h
                self.is_jumping = False
            else:
                self.on_ground = False
                if self.D.y > self.h - 200:
                    self.velocity += 0.5  # Increase velocity due to gravity
                    self.D.y += self.velocity

    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self._move(action)

        game_over = False
        if self.D.y >= self.h or self.frame_iteration > self.rec_iter:
            self.reward = -10
            self.score-=0
            game_over = True
            return self.reward, game_over, self.score
        if (self.Tile in self.ListOfListOfThem):
            self.score += 1
            self.reward = 10+self.h-self.D.y
            self.rec_iter+=200
            self.ListOfListOfThem.remove(self.Tile)

        self._update_ui()
        self.clock.tick(60)

        return self.reward, game_over, self.score

    def reset(self):
        self.direction = Direction.UP
        self.D = pygame.Rect(120, self.h - 200, 60, 60)


        self.velocity = 0
        self.score = 0
        self.on_ground = False
        self.is_jumping = False
        self.jump_height = 14
        self.game_over = False
        self.frame_iteration = 0


        self.Tile = None

    # MOOD
    def _move(self, action):

        clock_wise = [Direction.UP, Direction.RIGHT, Direction.LEFT]
        idx = clock_wise.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 3
            new_dir = clock_wise[next_idx]
        elif np.array_equal(action, [0, 0, 1]):
            next_idx = (idx - 1) % 3
            new_dir = clock_wise[next_idx]
        self.direction = new_dir
        if self.direction == Direction.UP and not self.is_jumping:

            self.left = False
            self.right = False
            self.is_jumping = True
            self.velocity = -self.jump_height
            self.on_ground = False

        elif self.direction == Direction.RIGHT:
            if not self.is_jumping:
                self.D.x += 7

            else:
                self.right = True

        elif self.direction == Direction.LEFT:
            if not self.is_jumping:
                self.D.x -= 7


            else:
                self.left = True

        if self.right == True :
            self.left = False
            self.D.x += 7
        elif self.left == True :
            self.right = False
            self.D.x -= 7

        if not self.on_ground:
            self.velocity += 0.5
            self.D.y += self.velocity

        # Apply gravity
        if not self.on_ground:
            self.velocity += 0.5  # Increase velocity due to gravity
            self.D.y += self.velocity
            for i in self.ListOfThem:
                self.Collisions(i)

        else:
            self.left = False
            self.right = False

        # Keep the character within the game boundaries
        if self.D.x < 0:
            self.D.x = 0
            self.right = True
            self.left = False




        elif self.D.x > self.w - 60:
            self.D.x = self.w - 60
            self.left = True
            self.right= False





    def _update_ui(self):
        self.display.fill(BLACK)
        for i in self.ListOfThem:
            pygame.draw.rect(self.display, White, i)
        Show = self.set_text("MOTHER FUCKER", self.w / 2, 40, 20)
        self.display.blit(Show[0], Show[1])

        pygame.draw.rect(self.display, White, self.D)

        pygame.display.flip()
