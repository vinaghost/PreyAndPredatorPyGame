import random

from pygame import Vector2, draw

from settings import *


class Entity:
    def __init__(self, position, speed, color, energy):
        self.position = Vector2(position)
        self.speed = speed
        self.color = color
        self.energy = energy

    def draw(self, screen):
        draw.circle(screen, self.color, self.position, BLOCK_SIZE)

    def move(self):
        angle = random.randint(0, 360)
        vector = Vector2()
        vector.from_polar((1, angle))

        self.position += vector * self.speed

        if self.position[0] > WINDOW_WIDTH - BLOCK_SIZE:
            self.position[0] = 0
        elif self.position[0] < 0:
            self.position[0] = WINDOW_WIDTH - BLOCK_SIZE

        if self.position[1] > WINDOW_HEIGHT - BLOCK_SIZE:
            self.position[1] = 0
        elif self.position[1] < 0:
            self.position[1] = WINDOW_HEIGHT - BLOCK_SIZE

        #self.energy -= 0.5
    
    def destroy(self):
        pass

    def spawn(self):
        pass

    def kill(self, interaction):
        self.position = interaction.position
        interaction.destroy()
        self.energy += 3

    def check_interaction(self, interactions):
        pass

    def update(self, screen):
        if self.energy < 1:
            self.destroy()
        else:
            self.move()

        self.draw(screen)