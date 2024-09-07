import os
import random
import shutil
from typing import List

import pygame
import torch
from pygame import Vector2, time

from EntityBrain import EntityBrain
from predator import Predator
from prey import Prey
from settings import *
from pyquadtree import QuadTree

def delete_folder(folder_path: str) -> None:
    try:
        shutil.rmtree(folder_path)
    except Exception as e:
        print(f'Failed to delete {folder_path}. Reason: {e}')

def create_folder(folder_path: str) -> None:
    try:
        os.makedirs(folder_path, exist_ok=True)
    except Exception as e:
        print(f'Failed to create the folder {folder_path}. Reason: {e}')

def check_folder_exists(folder_path: str) -> bool:
    return os.path.exists(folder_path)

def deconstruct_statedict(model: torch.nn.Module) -> torch.Tensor:
    one_dim_statedict = torch.Tensor()
    for key_name, weights in model.state_dict().items():
        flatten_weights = torch.flatten(weights)
        one_dim_statedict = torch.cat((one_dim_statedict, flatten_weights), dim=0)
    return one_dim_statedict

def get_model_architecture(model: torch.nn.Module) -> dict:
    model_architecture = {}
    for key_name, weights in model.state_dict().items():
        model_architecture[key_name] = weights.shape
    return model_architecture

def reconstruct_statedict(flatten_weights: torch.Tensor, model_architecture: dict) -> dict:
    state_dict = {}
    pointer = 0
    for key_name, weights_shape in model_architecture.items():
        if len(weights_shape) > 1:
            count_of_weights_this_module_needs = weights_shape[0] * weights_shape[1]
        else:
            count_of_weights_this_module_needs = weights_shape[0]
        slice_of_selected_weights = flatten_weights[pointer: pointer + count_of_weights_this_module_needs]
        state_dict[key_name] = torch.reshape(slice_of_selected_weights, model_architecture[key_name])
        pointer = count_of_weights_this_module_needs + pointer
    return state_dict

def cross_over(parents: List[torch.Tensor], count_of_children_needed: int) -> List[torch.Tensor]:
    cross_over_idx = round(len(parents[0]) * CROSS_OVER_THRESHOLD)
    children = []
    for idx in range(count_of_children_needed):
        # Find parents
        male = random.sample(parents, k=1)[0]
        female = random.sample(parents, k=1)[0]
        # Slice genes
        male_first_part = male[0:cross_over_idx]
        male_second_part = male[cross_over_idx::]
        female_first_part = female[0:cross_over_idx]
        female_second_part = female[cross_over_idx::]
        # Create new children
        child1 = torch.cat((male_first_part, female_second_part))
        child2 = torch.cat((female_first_part, male_second_part))
        children.append(child1)
        children.append(child2)
    return children

def mutation(child: torch.Tensor) -> torch.Tensor:
    count_of_initial_model_weights = child.size(dim=0)
    # Create mutation values.
    # Some random values. => exp. [0.9420, 0.8821, 0.4306, 0.7354, 0.1637]
    mutation_base_values = torch.rand(count_of_initial_model_weights)
    # Scale those random numbers. => exp. [0.0283, 0.0265, 0.0129, 0.0221, 0.0049]
    scaled_mutation_values = mutation_base_values * MUTATION_CHANGE_THRESHOLD
    # Get negation signs so weights are gonna increase and decrease. => exp. [ 1,  1,  1, -1, -1]
    negation_signs_for_scaled_mutation_values = torch.randint(0, 2, size=(
        1, count_of_initial_model_weights)).squeeze() * 2 - 1
    # Actual values which could be or not added to genes, only added if genes are selected. => exp. [ 0.0019,  0.0040,  0.0018, -0.0296, -0.0187]
    mutation_values_with_negation_signs = torch.mul(scaled_mutation_values, negation_signs_for_scaled_mutation_values)
    # Select which genes are gonna be mutated. => exp. [1, 1, 0, 0, 1]
    gene_selection_for_mutation = torch.randint(0, 2, (1, count_of_initial_model_weights)).squeeze()
    # Actual mutation, these values are gonna be added to cross overed children. => exp. [ 0.0211,  0.0058, -0.0000,  0.0000, -0.0172]
    mutation_values = torch.mul(gene_selection_for_mutation, mutation_values_with_negation_signs)
    # Perform mutation
    mutated_child = torch.add(child, mutation_values)
    return mutated_child


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
                    print(f"Left mouse button is pressed at position {mouse_pos}")

            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()
