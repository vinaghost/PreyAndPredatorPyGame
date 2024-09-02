import pygame
import sys
from environment import *

pygame.init()

SCREEN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Prey&Predator")

clock = pygame.time.Clock()

e = Environment(SCREEN)
fontObj = pygame.font.Font(None, 32)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or preys == [] or predators == []:
            pygame.quit()
            sys.exit()

    e.update()
    preyCounter = fontObj.render(f'Prey {len(preys)}', True, GREEN, TEXT_BACKGROUND)
    predatorCounter = fontObj.render(f'Predator {len(predators)}', True, RED, TEXT_BACKGROUND)

    SCREEN.blit(preyCounter, (TEXT_X, TEXT_Y))
    SCREEN.blit(predatorCounter, (TEXT_X, TEXT_Y + TEXT_SPACING + predatorCounter.get_height()))
    pygame.display.flip()

    clock.tick(FPS)