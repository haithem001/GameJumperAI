import math
import time

import pygame
import random
from pygame import *
from math import *
from Player import player
from Tile import tile
from enum import Enum

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    FIX = 4
(width, height) = (900, 900)
play = False

jumping = True
FPS = 3
h = 400
Dude = player(width / 2, height - 400, 60, 60)
Tile1 = tile(width / 2, height - 90, 100, 60)
Tile2 = tile(width-200,height - 180 , 100,60)

x, y = Dude.getX(), height-100

T1 = pygame.Rect(Tile1.getX(), Tile1.getY(), 100, 10)

T2 = pygame.Rect(Tile2.getX(), Tile2.getY(), 100, 10)

background_colour = (0, 0, 0)
D = pygame.Rect(Dude.getX(), Dude.getY(), 60, 60)

screen = pygame.display.set_mode((width, height))

pygame.display.set_caption('Jumper')
screen.fill(background_colour)

direction = Direction.FIX
def Draw():
    pygame.draw.rect(screen, (100, 255, 100), D)
    pygame.draw.rect(screen, (100, 255, 100), T1)
    pygame.draw.rect(screen, (100, 255, 100), T2)



dx = 50
dy = 50
#movement functions
def play_step():
    global play
    global pos
    global direction
    for event in pygame.event.get():
        pygame.display.update()
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
        #input
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                play= True
            elif event.key == pygame.K_LEFT:
                direction = Direction.LEFT
            elif event.key == pygame.K_RIGHT:
                direction = Direction.RIGHT
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT:
                direction = Direction.FIX
            if event.key == pygame.K_RIGHT:
                direction = Direction.FIX
    _move(direction)
#action
def _move(direction):
    global play
    global pos
    if direction == Direction.RIGHT:
        pos = 0.5
    elif direction == Direction.LEFT:
        pos = -0.5
    elif direction == Direction.UP:
        play = True

    elif direction == Direction.FIX:
        pos = 0
def reset():
    Dude.setX(width / 2)
    Dude.setY(height - 400)
    frame_iteration = 0
def Gravity(x):
    if x==1:
        Dude.setY(Dude.getY() + 3.2 / FPS)
    else:
        pass



def Update():

    global dx, dy

    D.update(Dude.getX(), Dude.getY(), 60, 60)



def Jump():
    global h
    global height
    global jumping
    global play

    if Dude.getY() > height - Dude.getH() - h and play is True:
        if Tile1.getX() - Tile1.getW() / 2 <= Dude.getX() <= Tile1.getX() + Tile1.getW() and Dude.getY() >= Tile1.getY()+10:
            play = False
        else:
            Dude.setY(Dude.getY() - 3 / FPS)
    else:
        play = False

    if not play:
            if Tile1.getX()-Dude.getW()<Dude.getX()<Tile1.getX()+Tile1.getW() and Dude.getY()>Tile1.getY()-Dude.getH():
                if  Dude.getY()==Tile1.getY()-10:
                    Gravity(0)
            elif Tile2.getX()-Dude.getW()<Dude.getX()<Tile2.getX()+Tile2.getW() and Dude.getY()>Tile2.getY()-Dude.getH():
                if  Dude.getY()==Tile2.getY()-10:
                    Gravity(0)
            else :
                Gravity(1)

    if Dude.getY()>height:
        reset()




# Launch
pos = 0
running = True
while running:
    screen.fill((0, 0, 0))
    Update()

    print(Dude.getX())
    print(Dude.getY())
    Draw()

    pygame.display.flip()
    play_step()


    # Horizontal movement
    if width - Dude.getW() - pos > Dude.getX() > 0:
        Dude.setX(Dude.getX() + pos)
    elif Dude.getX() <= 0:
        Dude.setX(Dude.getX() + 0.5)
    elif Dude.getX() >= width - Dude.getW() - pos:
        Dude.setX(Dude.getX() - 0.5)

    if jumping:
        Jump()
