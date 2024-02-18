import pygame
from enum import Enum

BLACK = (0, 0, 0)
BLUE1 = (0, 0, 255)
RED = (100, 0, 0)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    FIX = 4


class Game:
    def __init__(self, w=900, h=900):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('JumpGame')
        self.clock = pygame.time.Clock()
        self.direction = Direction.FIX
        self.D = pygame.Rect(self.w / 2, self.h - 200, 60, 60)
        self.Ground = pygame.Rect(0, self.h - 140, self.w, self.h)
        self.Tile1 = pygame.Rect(self.w / 2 + 100, self.h - 300, 100, 10)
        self.Tile2 = pygame.Rect(self.w / 2 - 200, self.h - 400, 100, 10)
        self.velocity = 0
        self.on_ground = False
        self.jump_height = 15
       

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
                elif event.key == pygame.K_UP :
                    self.velocity = -self.jump_height
                    self.on_ground = False

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                    self.direction = Direction.FIX
                elif event.key == pygame.K_UP:
                    self.velocity = 0
                    self.on_ground = False

        self._move()
        self._update_ui()

    def reset(self):
        self.D.x = self.w / 2
        self.h - 300
#MOOD
    def _move(self):
        if self.direction == Direction.RIGHT:
            self.D.x += 5
        elif self.direction == Direction.LEFT:
            self.D.x -= 5

        # Apply gravity
        if not self.on_ground:
            self.velocity += 0.5  # Increase velocity due to gravity
            self.D.y += self.velocity
            if (self.D.x < self.Tile1.x + self.Tile1.w and
                    self.D.x + self.D.w > self.Tile1.x and
                    self.D.y < self.Tile1.y + 1.2*self.Tile1.h and
                    self.D.y + self.D.h > self.Tile1.y):
                self.velocity = 0
                if (self.D.y + self.D.h >= self.Tile1.y) and (self.D.y < self.Tile1.y):
                    self.D.y = self.Tile1.y - self.D.h

            if (self.D.x < self.Tile2.x + self.Tile2.w and
                self.D.x + self.D.w > self.Tile2.x and
                self.D.y < self.Tile2.y + self.Tile2.h and
                self.D.y + self.D.h > self.Tile2.y):
                self.velocity = 0
                if (self.D.y + self.D.h >= self.Tile2.y) and (self.D.y < self.Tile2.y):
                    self.D.y = self.Tile2.y - self.D.h

            else:
                self.on_ground = False
                if self.D.y > self.h - 200:
                    self.velocity += 0.5  # Increase velocity due to gravity
                    self.D.y += self.velocity

        if self.D.y >= self.h - 200:  # Check if the character has landed on the ground
            self.D.y = self.h - 200
            self.on_ground = True
            self.velocity = 0

        # Keep the character within the game boundaries
        if self.D.x < 0:
            self.D.x = 0
        elif self.D.x > self.w - 60:
            self.D.x = self.w - 60

    def _update_ui(self):
        self.display.fill(BLACK)
        pygame.draw.rect(self.display, RED, self.Tile1)
        pygame.draw.rect(self.display, BLUE1, self.Tile2)
        pygame.draw.rect(self.display, RED, self.Ground)
        pygame.draw.rect(self.display, BLUE1, self.D)

        pygame.display.flip()


if __name__ == '__main__':
    pygame.init()
    game = Game()
    running = True
    while running:
        game.play_step()
        game.clock.tick(60)
    pygame.quit()
