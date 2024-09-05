from settings import *
from pyquadtree import QuadTree
from pygame import Surface

class Environment:
    def __init__(self, screen : Surface):
        self.screen = screen

        self.preys = []
        self.predators = []

        self.prey_tree = None
        self.predator_tree = None

    def get_prey_count(self) -> int:
        return sum(1 for p in self.preys if p.is_alive)

    def get_predator_count(self) -> int:
        return sum(1 for p in self.predators if p.is_alive)

    def kill_all(self) -> None:
        for prey in self.preys:
            if prey.is_alive:
                prey.destroy()
        for predator in self.predators:
            if predator.is_alive:
                predator.destroy()

    def is_epoch_completed(self) -> bool:
        return all([not p.is_alive for p in self.preys]) or all([not p.is_alive for p in self.predators])

    def create_prey_tree(self) -> None:
        self.prey_tree = QuadTree(bbox=(0, 0, WINDOW_WIDTH + 1, WINDOW_HEIGHT + 1))
        for prey in self.preys:
            if prey.is_alive:
                self.prey_tree.add(prey.identity, tuple(prey.position))

    def create_predator_tree(self) -> None:
        self.predator_tree = QuadTree(bbox=(0, 0, WINDOW_WIDTH + 1, WINDOW_HEIGHT + 1))
        for predator in self.predators:
            if predator.is_alive:
                self.predator_tree.add(predator.identity, tuple(predator.position))

    def check_dead(self) -> None:
        for prey in self.preys:
            if prey.is_alive and prey.energy < 0:
                prey.destroy()
        for predator in self.predators:
            if predator.is_alive and predator.energy < 0:
                predator.destroy()

    def update(self, delta_time : float) -> None:
        self.screen.fill(WHITE)

        self.check_dead()

        self.create_prey_tree()
        self.create_predator_tree()

        for predator in self.predators:
            if predator.is_alive:
                predator.update(self.screen, delta_time)

        for prey in self.preys:
            if prey.is_alive:
                prey.update(self.screen, delta_time)
