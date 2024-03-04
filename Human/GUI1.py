import pygame
from enum import Enum

BLACK = (0, 0, 0)
White = (255, 255, 255)
Red = (255,0,0)





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




    def play_step(self):

        self.frame_iteration +=1
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
                    self.direction = Direction.FIX


        self._move()
        self._update_ui()

    def reset(self):
        self.direction = Direction.FIX
        self.D = pygame.Rect(100, self.h - 220, 60, 60)

        self.ListOfThem = [pygame.Rect(300, self.h - 460, 300, 10),
                           pygame.Rect(700, self.h - 240, 200, 10),
                           pygame.Rect(340, self.h - 200, 400, 300),
                           pygame.Rect(0, self.h - 140, 300, 50)
                           ]

        self.velocity = 0
        self.on_ground = False
        self.is_jumping = False
        self.jump_height = 15
        self.game_over = False
        self.frame_iteration = 0
        self.ListofListofThem = self.ListOfThem.copy()
        self.Tile = None


    # MOOD
    def _move(self):

        if self.direction == Direction.RIGHT:
            self.D.x += 5
        elif self.direction == Direction.LEFT:
            self.D.x -= 5

        # Apply gravity
        if not self.on_ground:
            self.velocity += 0.5  # Increase velocity due to gravity
            self.D.y += self.velocity
            for i in self.ListOfThem:
                self.Collisions(i)


        if self.D.y >= self.h  :  # Check if the character has landed on the ground
            self.reset()

        if(self.Tile in self.ListofListofThem):
            self.ListofListofThem.remove(self.Tile)


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


if __name__ == '__main__':
    pygame.init()
    game = Game()
    running = True
    while running:
        if game.game_over==False:
            game.play_step()
            game.clock.tick(60)

            if(game.D.x==840 and game.D.y ==40):
                game.game_over=True

        else:
            game.draw_text("Game Over",White,game.w/2-100,450)
            game.draw_text("You win",White,game.w/2-80,500)
            pygame.display.flip()

    pygame.quit()