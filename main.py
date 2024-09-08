import csv
import pygame

from environment import Environment
from settings import *

pygame.display.init()
pygame.font.init()

e = Environment(WINDOW_WIDTH, WINDOW_HEIGHT)

time_interval = 10_000

prey_log = []
predator_log = []

e.initialize()
e.start()

last_check_time = pygame.time.get_ticks()

try:
    while e.running:
        e.loop()

        prey_count = e.get_prey_count()
        predator_count = e.get_predator_count()

        if prey_count == 0 or predator_count == 0:
            break

        current_time = pygame.time.get_ticks()
        if current_time - last_check_time > time_interval:
            last_check_time = current_time

            prey_log.append(prey_count)
            predator_log.append(predator_count)
finally:
    e.stop()

    prey_count = e.get_prey_count()
    predator_count = e.get_predator_count()
    prey_log.append(prey_count)
    predator_log.append(predator_count)


with open('output.csv', mode='w', newline='') as file:
    writer = csv.writer(file)

    writer.writerow(prey_log)
    writer.writerow(predator_log)

