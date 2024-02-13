import math

import pygame
import random
from pygame import *
from math import *
from Player import player
from Tile import tile
(width, height) = (900, 900)
play = False
jumping =False
FPS=3
Dude = player(width / 2, height - 60, 60, 60)
Tile1= tile(width/2, height-90,100,60)
T1=pygame.Rect(Tile1.getX(), Tile1.getY(), 100, 10)
background_colour = (0, 0, 0)
D = pygame.Rect(Dude.getX(), Dude.getY(), 60, 60)

screen = pygame.display.set_mode((width, height))

pygame.display.set_caption('Jumper')
screen.fill(background_colour)


def Draw():
    pygame.draw.rect(screen, (100, 255, 100), D)
    pygame.draw.rect(screen, (100, 255, 100), T1)


dx = 50
dy = 50


def Update():
    global dx, dy

    D.update(Dude.getX(), Dude.getY(), 60, 60)


def Jump(h):
    global jumping
    global play
    if Dude.getY() > height-Dude.getH()-h and play == True :
        Dude.setY(Dude.getY() - 3/FPS)

    else:
        play = False
    if not play and Dude.getY() < height - Dude.getH():
        Dude.setY(Dude.getY() + 3.2/FPS)
    if Dude.getY()>=height :
        jumping=False
    if Dude.getY()<=Tile1.getY() and Dude.getX()>Tile1.getX() and Dude.getX()<Tile1.getW():
        play=False

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
    for event in pygame.event.get():
        pygame.display.update()

        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                jumping=True
                play=True
            elif event.key == pygame.K_LEFT:
                pos=-0.5

            elif event.key == pygame.K_RIGHT:
                pos=0.5
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT:
                pos=0
            if event.key == pygame.K_RIGHT:
                pos=0

    if (Dude.getX() < width-Dude.getW()-pos and Dude.getX()>0):
        Dude.setX(Dude.getX()+pos)
    elif Dude.getX()<=0:
        Dude.setX(Dude.getX()+0.5)
    elif Dude.getX() >= width-Dude.getW()-pos:
        Dude.setX(Dude.getX()-0.5)

    if jumping==True:
        Jump(160)
