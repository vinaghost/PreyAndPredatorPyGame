import pygame
from pygame import Vector2, time

from EntityBrain import EntityBrain
from misc.genetic import *
from settings import *
from predator import Predator
from prey import Prey
from pyquadtree import QuadTree

class Environment:
    def __init__(self, width=180, height=180):
        self.w = width
        self.h = height

        # init display
        self.screen = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Prey & Predator")
        self.clock = pygame.time.Clock()

        self.last_click_time = 0
        self.double_click_threshold = 500

        # init game
        self.preys = [] # type: List[Prey]
        self.predators = [] # type: List[Predator]

        self.running = False

        self.model_architecture = get_model_architecture(EntityBrain())


    def get_prey_count(self) -> int:
        return sum(1 for p in self.preys if p.is_alive)

    def get_predator_count(self) -> int:
        return sum(1 for p in self.predators if p.is_alive)

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
        if len(self.predators) < PREDATOR_COUNT:
            predator_children = []
            for predator in self.predators:
                if predator.is_split_able():
                    predator.split()
                    brain_info = deconstruct_statedict(predator.brain)
                    brain_info = mutation(brain_info)
                    brain_info = reconstruct_statedict(brain_info, self.model_architecture)
                    brain = EntityBrain()
                    brain.load_state_dict(brain_info)
                    brain.eval()
                    predator_children.append(Predator(predator.generation + 1, brain, predator.position + Vector2(RADIUS, RADIUS)))
            self.predators.extend(predator_children)

        if len(self.preys) < PREY_COUNT:
            prey_children = []
            for prey in self.preys:
                if prey.is_split_able():
                    prey.split()
                    brain_info = deconstruct_statedict(prey.brain)
                    brain_info = mutation(brain_info)
                    brain_info = reconstruct_statedict(brain_info, self.model_architecture)
                    brain = EntityBrain()
                    brain.load_state_dict(brain_info)
                    brain.eval()
                    prey_children.append(Prey(prey.generation + 1, brain, prey.position + Vector2(RADIUS, RADIUS)))
            self.preys.extend(prey_children)

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

    def stop(self):
        self.running = False

    def start(self):
        self.running = True
        self.initialize()

        while self.running:
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

            # Check if any mouse button is pressed
            mouse_buttons = pygame.mouse.get_pressed()
            if mouse_buttons[0]:  # Left mouse button
                current_time = time.get_ticks()
                if current_time - self.last_click_time < self.double_click_threshold:
                    pass
                else:
                    self.last_click_time = current_time
                    mouse_pos = pygame.mouse.get_pos()
                    #print(f"Left mouse button is pressed at position {mouse_pos}")

            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()
