from predator import *
from prey import *


class Environment:
    def __init__(self, screen):
        self.screen = screen

        for i in range(PREY_COUNT):
            preys.append(Prey([random.randint(0, WINDOW_WIDTH), random.randint(0, WINDOW_HEIGHT) ], SPEED, GREEN, 10))
        
        for j in range(PREDATOR_COUNT):
            predators.append(Predator([random.randint(0, WINDOW_WIDTH) , random.randint(0, WINDOW_HEIGHT) ], SPEED, RED, 10))

    def update(self):
        self.screen.fill(WHITE)

        for prey in preys:
            prey.update(self.screen)

        for predator in predators:
            predator.update(self.screen)
