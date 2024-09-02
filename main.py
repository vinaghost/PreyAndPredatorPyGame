import pygame
import sys
from environment import *

pygame.init()

SCREEN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Earfh")

clock = pygame.time.Clock()

e = Environment(SCREEN)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or preys == [] or predators == []:
            pygame.quit()
            sys.exit()

    e.update()

    pygame.display.flip()

    clock.tick(FPS)