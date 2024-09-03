import pygame
import sys
from environment import *
from prey import Prey
from predator import Predator
import uuid
import random

pygame.init()

SCREEN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Prey&Predator")

clock = pygame.time.Clock()
prey = Prey( uuid.uuid4(), (random.randint(0, WINDOW_WIDTH), random.randint(0, WINDOW_HEIGHT)), [], [])
predator = Predator(uuid.uuid4(), (random.randint(0, WINDOW_WIDTH), random.randint(0, WINDOW_HEIGHT)), [], [])

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    SCREEN.fill(WHITE)
    prey.draw(SCREEN)
    predator.draw(SCREEN)
    pygame.display.flip()

    clock.tick(FPS)