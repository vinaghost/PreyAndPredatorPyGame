from settings import *
from prey import Prey
from predator import Predator
import random
import uuid
from pyquadtree import QuadTree

class Environment:
    def __init__(self, screen):
        self.screen = screen

        self.preys = []
        self.predators = []

        for _ in range(PREY_COUNT):
            self.preys.append(Prey( uuid.uuid4(), (random.randint(0, WINDOW_WIDTH), random.randint(0, WINDOW_HEIGHT)), self.preys, self.predators))
        
        for _ in range(PREDATOR_COUNT):
            self.predators.append(Predator(uuid.uuid4(), (random.randint(0, WINDOW_WIDTH), random.randint(0, WINDOW_HEIGHT)), self.predators, self.preys))

    def update(self):
        self.screen.fill(WHITE)

        prey_tree = QuadTree(bbox=(0, 0, WINDOW_WIDTH + 1, WINDOW_HEIGHT + 1))
        for prey in self.preys:
            if prey.is_alive:
                prey_tree.add(prey.identity, tuple(prey.position))

        for predator in self.predators:
            predator.update(self.screen, prey_tree)

        predator_tree = QuadTree(bbox=(0, 0, WINDOW_WIDTH + 1, WINDOW_HEIGHT + 1))
        for predator in self.predators:
            if predator.is_alive:
                predator_tree.add(predator.identity, tuple(predator.position))

        for prey in self.preys:
            prey.update(self.screen, predator_tree)
