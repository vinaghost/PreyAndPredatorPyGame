from settings import *
from pyquadtree import QuadTree

class Environment:
    def __init__(self, screen):
        self.screen = screen

        self.preys = []
        self.predators = []

        self.prey_tree = None
        self.predator_tree = None

    def get_prey_count(self):
        return sum(1 for p in self.preys if p.is_alive)

    def get_predator_count(self):
        return sum(1 for p in self.predators if p.is_alive)

    def kill_all(self):
        for prey in self.preys:
            if prey.is_alive:
                prey.destroy()
        for predator in self.predators:
            if predator.is_alive:
                predator.destroy()

    def is_epoch_completed(self):
        return all([not p.is_alive for p in self.preys]) or all([not p.is_alive for p in self.predators])

    def create_prey_tree(self):
        self.prey_tree = QuadTree(bbox=(0, 0, WINDOW_WIDTH + 1, WINDOW_HEIGHT + 1))
        for prey in self.preys:
            if prey.is_alive:
                self.prey_tree.add(prey.identity, tuple(prey.position))

    def create_predator_tree(self):
        self.predator_tree = QuadTree(bbox=(0, 0, WINDOW_WIDTH + 1, WINDOW_HEIGHT + 1))
        for predator in self.predators:
            if predator.is_alive:
                self.predator_tree.add(predator.identity, tuple(predator.position))

    def update(self):
        self.screen.fill(WHITE)

        self.create_prey_tree()

        for predator in self.predators:
            predator.update(self.screen)

        self.create_predator_tree()

        for prey in self.preys:
            prey.update(self.screen)
