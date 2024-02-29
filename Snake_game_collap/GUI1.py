import pygame
from enum import Enum
import numpy as np

BLACK = (0, 0, 0)
White = (255, 255, 255)
Red = (255,0,0)


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
    def set_text(self,string, coordx, coordy, fontSize):  # Function to set text

        font = pygame.font.Font('04B_30__.TTF', fontSize)
        # (0, 0, 0) is black, to make black text
        text = font.render(string, True, Red)
        textRect = text.get_rect()
        textRect.center = (coordx, coordy)
        return (text, textRect)

    def __init__(self, w=900, h=900):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('JumpGame')
        self.clock = pygame.time.Clock()

        self.reset()


    def draw_text(self,text, text_col, x, y):
        font = pygame.font.Font("04B_30__.TTF",24)
        img = font.render(text, True, text_col)
        self.display.blit(img,(x,y))

    def Collisions(self, Tile):
        if (self.D.x < Tile.x + Tile.w and
                self.D.x + self.D.w > Tile.x and
                self.D.y < Tile.y + 1.5 * Tile.h and
                self.D.y + self.D.h > Tile.y):
            self.velocity = 0
            self.is_jumping = True

            if (Tile.y - self.D.y < self.D.h) and not (Tile.x - 55 < self.D.x):
                self.D.x = Tile.x - self.D.w

            elif (Tile.y - self.D.y < self.D.h) and not (Tile.x > self.D.x - 95):
                self.D.x = Tile.x + Tile.w

            elif (self.D.y + self.D.h >= Tile.y) and (self.D.y < Tile.y):
                self.D.y = Tile.y - self.D.h
                self.is_jumping = False
                self.Tile = Tile
            else:
                self.on_ground = False
                if self.D.y > self.h - 200:
                    self.velocity += 0.5  # Increase velocity due to gravity
                    self.D.y += self.velocity

    def play_step(self,action):
        self.frame_iteration+=1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()


        self._move(action)

        reward = 0
        game_over = False
        if self.D.y >= self.h or self.frame_iteration>(900-self.D.y):
            reward = -10
            game_over = True
            return reward,game_over,self.score
        if(self.Tile in self.ListOfListOfThem):
            self.score +=1
            reward = 10
            self.ListOfListOfThem.remove(self.Tile)

        self._update_ui()
        self.clock.tick(70)

        return reward,game_over,self.score



    def reset(self):
        self.direction = Direction.UP
        self.D = pygame.Rect(self.w / 2, self.h - 200, 60, 60)

        self.ListOfThem = [pygame.Rect(self.w / 2 + 100, self.h - 300, 100, 10),
                           pygame.Rect(self.w / 2 - 100, self.h - 400, 100, 10),
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
        self.score = 0
        self.on_ground = False
        self.is_jumping = False
        self.jump_height = 15
        self.game_over = False
        self.frame_iteration = 0
        self.ListOfListOfThem= self.ListOfThem.copy()
        self.ListOfListOfThem.remove(self.ListOfListOfThem[-1])

        self.Tile= None
    # MOOD
    def _move(self,action):
        clock_wise = [Direction.UP,Direction.RIGHT,Direction.LEFT]
        idx = clock_wise.index(self.direction)
        if np.array_equal(action, [1,0,0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action,[0,1,0]):
            next_idx = (idx+1)%3
            new_dir = clock_wise[next_idx]
        elif np.array_equal(action,[0,0,1]):
            next_idx = (idx-1)%3
            new_dir = clock_wise[next_idx]
        self.direction = new_dir
        if self.direction == Direction.RIGHT:
            self.D.x += 5
        elif self.direction == Direction.LEFT:
            self.D.x -= 5
        elif self.direction == Direction.UP and not self.is_jumping:
            self.is_jumping = True
            self.velocity = -self.jump_height
            self.on_ground = False

        if not self.on_ground:
            self.velocity += 0.5
            self.D.y += self.velocity



        # Apply gravity
        if not self.on_ground:
            self.velocity += 0.5  # Increase velocity due to gravity
            self.D.y += self.velocity
            for i in self.ListOfThem:
                self.Collisions(i)





        # Keep the character within the game boundaries
        if self.D.x < 0:
            self.D.x = 0
        elif self.D.x > self.w - 60:
            self.D.x = self.w - 60

    def _update_ui(self):
        self.display.fill(BLACK)
        for i in self.ListOfThem:
            pygame.draw.rect(self.display, White, i)
        Show = self.set_text("MOTHER FUCKER", self.w/2, 40, 20)
        self.display.blit(Show[0], Show[1])

        pygame.draw.rect(self.display, White, self.D)

        pygame.display.flip()

