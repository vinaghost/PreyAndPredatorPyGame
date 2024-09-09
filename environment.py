import random

import pygame
from pygame import Vector2
import concurrent.futures

from EntityBrain import EntityBrain
from misc.genetic import *
from settings import *
from predator import Predator
from prey import Prey
from pyquadtree import QuadTree

class Environment:
    def __init__(self, width : int, height : int):
        self.w = width
        self.h = height

        # init display
        self.screen = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Prey & Predator")
        self.clock = pygame.time.Clock()

        # init game
        self.preys = [] # type: List[Prey]
        self.predators = [] # type: List[Predator]

        self.running = False

        self.model_architecture = get_model_architecture(EntityBrain())

    def get_prey_count(self) -> int:
        return len(self.preys)

    def get_predator_count(self) -> int:
        return len(self.predators)

    def initialize(self) -> None:
        self.preys = [Prey(0, EntityBrain(), Vector2(random.randint(0, WINDOW_WIDTH), random.randint(0, WINDOW_HEIGHT))) for _ in range(PREY_COUNT)]
        self.predators = [Predator(0, EntityBrain(), Vector2(random.randint(0, WINDOW_WIDTH), random.randint(0, WINDOW_HEIGHT))) for _ in range(PREDATOR_COUNT)]

    def create_tree(self) -> QuadTree:
        tree = QuadTree(bbox=(0, 0, WINDOW_WIDTH + 1, WINDOW_HEIGHT + 1))
        for prey in self.preys:
            tree.add(prey, tuple(prey.position))
        for predator in self.predators:
            tree.add(predator, tuple(predator.position))
        return tree

    def filter_dead(self):
        self.preys = [p for p in self.preys if p.is_alive()]
        self.predators = [p for p in self.predators if p.is_alive()]

    def update_entities(self):
        tree = self.create_tree()
        for predator in self.predators:
            predator.update(tree)
        for prey in self.preys:
            prey.update(tree)
    def give_birth(self):
        pregnant_predator = [predator for predator in self.predators if predator.is_split_able()]
        predator_children_count = min(len(pregnant_predator), PREDATOR_COUNT - len(self.predators))
        if predator_children_count < len(pregnant_predator):
            pregnant_predator = random.sample(pregnant_predator, predator_children_count)


        pregnant_prey = [prey for prey in self.preys if prey.is_split_able()]
        prey_children_count = min(len(pregnant_prey), PREY_COUNT - len(self.preys))
        if prey_children_count < len(pregnant_prey):
            pregnant_prey = random.sample(pregnant_prey, prey_children_count)

        def create_brain(brain : EntityBrain) -> EntityBrain:
            info = deconstruct_statedict(brain)
            info = mutation(info)
            info = reconstruct_statedict(info, self.model_architecture)
            child_brain = EntityBrain()
            child_brain.load_state_dict(info)
            child_brain.eval()
            return child_brain

        def predator_hospital():
            children = []
            for predator in pregnant_predator:
                predator.split()
                children.append(Predator(predator.generation + 1, create_brain(predator.brain),
                                              predator.position + Vector2(random.randint(-RADIUS, RADIUS),
                                                                          random.randint(-RADIUS, RADIUS))))
            return children

        def prey_hospital():
            children = []
            for prey in pregnant_prey:
                prey.split()
                children.append(Prey(prey.generation + 1, create_brain(prey.brain),
                                      prey.position + Vector2(random.randint(-RADIUS, RADIUS),
                                                              random.randint(-RADIUS, RADIUS))))
            return children

        with concurrent.futures.ThreadPoolExecutor() as executor:
            predator_future = executor.submit(predator_hospital)
            prey_future = executor.submit(prey_hospital)

            new_predators = predator_future.result()
            new_preys = prey_future.result()

            self.predators.extend(new_predators)
            self.preys.extend(new_preys)


    def move_entities(self, delta_time: float) -> None:
        for predator in self.predators:
            predator.move(delta_time)
        for prey in self.preys:
            prey.move(delta_time)

    def draw_entities(self) -> None:
        for predator in self.predators:
            predator.draw(self.screen)
        for prey in self.preys:
            prey.draw(self.screen)

    def loop(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

        self.screen.fill(WHITE)
        delta_time = self.clock.get_time() / 1000.0

        self.update_entities()
        self.filter_dead()
        self.give_birth()
        self.move_entities(delta_time)
        self.draw_entities()

        pygame.display.flip()
        self.clock.tick(FPS)

    def start(self):
        self.running = True

    def stop(self):
        self.running = False
        pygame.quit()


