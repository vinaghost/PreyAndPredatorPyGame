# https://github.com/ardalan-dsht/genetic-algorithm-for-pytorch/blob/main/genetic_algorithm.py

from typing import List

import pygame
import torch
import random
from environment import Environment
from NeuralNetwork import NeuralNetwork
from settings import *

from prey import Prey
from predator import Predator
import uuid
import shutil
import os

def delete_folder(folder_path: str) -> None:
    try:
        shutil.rmtree(folder_path)
        print(f'Successfully deleted {folder_path}')
    except Exception as e:
        print(f'Failed to delete {folder_path}. Reason: {e}')

def create_folder(folder_path: str) -> None:
    try:
        os.makedirs(folder_path, exist_ok=True)
        print(f'Successfully created the folder {folder_path}')
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

def reconstruct_statedict(flatten_weights: torch.Tensor, model_architecture : dict) -> dict:
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

def cross_over(population: List[torch.Tensor], count_of_children_needed: int ) -> List[torch.Tensor]:
    cross_over_idx = round(len(population[0]) * CROSS_OVER_THRESHOLD)
    children = []
    for idx in range(count_of_children_needed):
        # Find parents
        male = random.sample(population, k=1)[0]
        female = random.sample(population, k=1)[0]
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

def mutation(children: List[torch.Tensor]) -> List[torch.Tensor]:
    mutated_children = []

    if len(children) == 0:
        return mutated_children

    count_of_initial_model_weights = children[0].size(dim=0)

    for child in children:
        # Create mutation values.
        # Some random values. => exp. [0.9420, 0.8821, 0.4306, 0.7354, 0.1637]
        mutation_base_values = torch.rand(count_of_initial_model_weights)
        # Scale those random numbers. => exp. [0.0283, 0.0265, 0.0129, 0.0221, 0.0049]
        scaled_mutation_values = mutation_base_values * MUTATION_CHANGE_THRESHOLD
        # Get negation signs so weights are gonna increase and decrease. => exp. [ 1,  1,  1, -1, -1]
        negation_signs_for_scaled_mutation_values =torch.randint(0,2,size=(1, count_of_initial_model_weights)).squeeze() * 2 - 1
        # Actual values which could be or not added to genes, only added if genes are selected. => exp. [ 0.0019,  0.0040,  0.0018, -0.0296, -0.0187]
        mutation_values_with_negation_signs = torch.mul(scaled_mutation_values, negation_signs_for_scaled_mutation_values)
        # Select which genes are gonna be mutated. => exp. [1, 1, 0, 0, 1]
        gene_selection_for_mutation = torch.randint(0, 2, (1, count_of_initial_model_weights)).squeeze()
        # Actual mutation, these values are gonna be added to cross overed children. => exp. [ 0.0211,  0.0058, -0.0000,  0.0000, -0.0172]
        mutation_values = torch.mul(gene_selection_for_mutation, mutation_values_with_negation_signs)
        # Perform mutation
        mutated_child = torch.add(child, mutation_values)
        mutated_children.append(mutated_child)
    return mutated_children


def handle_predator(population: list[Predator]) -> list[NeuralNetwork]:
    sorted_population = sorted(population, key=lambda x: x.get_fitness_score(), reverse=True)

    threshold = round(len(sorted_population) * PERCENTAGE_OF_PARENTS_TO_KEEP)
    best_individuals_in_population = sorted_population[0:threshold]

    print(f'Best predator:  {best_individuals_in_population[0].get_describe()}')
    brains = [individual.brain for individual in best_individuals_in_population]
    next_generation = []
    next_generation.extend(brains)

    count_of_children_needed = PREDATOR_COUNT - len(best_individuals_in_population) // 2 + 1

    adn_list = [deconstruct_statedict(brain) for brain in brains]
    children_adn = cross_over(adn_list, count_of_children_needed)
    model_architecture = get_model_architecture(brains[0])
    for adn in children_adn:
        state_dict = reconstruct_statedict(adn, model_architecture)
        brain = NeuralNetwork()
        brain.load_state_dict(state_dict)
        brain.eval()
        next_generation.append(brain)
    return next_generation

def initialize_population(environment : Environment):
    environment.preys = []
    environment.predators = []
    for _ in range(PREY_COUNT):
        environment.preys.append(Prey(uuid.uuid4(), (random.randint(0, WINDOW_WIDTH // (2 * DIAMETER)) * 2 * DIAMETER, random.randint(0, WINDOW_HEIGHT // (2 * DIAMETER)) * 2 * DIAMETER), environment))
    if check_folder_exists(folder_name):
        first_state_dict = torch.load(os.path.join(folder_name, 'best_predator_brain.pth'))
        second_state_dict = torch.load(os.path.join(folder_name, 'second_best_predator_brain.pth'))

        for i in range(PREDATOR_COUNT):
            brain = NeuralNetwork()
            if i % 2 == 0:
                state_dict = first_state_dict
            else:
                state_dict = second_state_dict
            brain.load_state_dict(state_dict)
            environment.predators.append(Predator(uuid.uuid4(), brain,(random.randint(0, WINDOW_WIDTH // (2 * DIAMETER)) * 2 * DIAMETER, random.randint(0, WINDOW_HEIGHT // (2 * DIAMETER)) * 2 * DIAMETER), environment))
    else:
        for _ in range(PREDATOR_COUNT):
            environment.predators.append(Predator(uuid.uuid4(), NeuralNetwork(),(random.randint(0, WINDOW_WIDTH // (2 * DIAMETER)) * 2 * DIAMETER, random.randint(0, WINDOW_HEIGHT // (2 * DIAMETER)) * 2 * DIAMETER), environment))

def recreate_population(environment : Environment, brains : list[NeuralNetwork]):
    environment.preys = []
    environment.predators = []

    for _ in range(PREY_COUNT):
        environment.preys.append(Prey(uuid.uuid4(), (random.randint(0, WINDOW_WIDTH // (2 * DIAMETER)) * 2 * DIAMETER, random.randint(0, WINDOW_HEIGHT // (2 * DIAMETER)) * 2 * DIAMETER), environment))

    for brain in brains:
        environment.predators.append(Predator(uuid.uuid4(), brain, (random.randint(0, WINDOW_WIDTH // (2 * DIAMETER)) * 2 * DIAMETER, random.randint(0, WINDOW_HEIGHT // (2 * DIAMETER)) * 2 * DIAMETER), environment))


pygame.init()
SCREEN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Prey&Predator")
clock = pygame.time.Clock()
fontObj = pygame.font.Font(None, 32)

e = Environment(SCREEN)
folder_name = 'brain_storage'

initialize_population(e)
generation = 0
running = True

prey = 0
predator = 0

while running:
    if e.is_epoch_completed():
        if e.get_prey_count() == 0:
            print('All preys are dead')
            predator += 1
        else:
            print('All predators are dead')
            prey += 1

        print(f'Epoch {generation} is completed')
        print(f'Prey - Predator: {prey} - {predator}')

        e.kill_all()
        predator_brains = handle_predator(e.predators)
        recreate_population(e, predator_brains)
        generation += 1

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    delta_time = clock.get_time() / 1000.0
    e.update(delta_time)

    generationCounter = fontObj.render(f'Generation {generation}', True, BLACK, TEXT_BACKGROUND)
    preyCounter = fontObj.render(f'Prey {e.get_prey_count()}', True, GREEN, TEXT_BACKGROUND)
    predatorCounter = fontObj.render(f'Predator {e.get_predator_count()}', True, RED, TEXT_BACKGROUND)

    SCREEN.blit(generationCounter, (TEXT_X, TEXT_Y))
    SCREEN.blit(preyCounter, (TEXT_X, TEXT_Y + TEXT_SPACING + generationCounter.get_height()))
    SCREEN.blit(predatorCounter, (TEXT_X, TEXT_Y + TEXT_SPACING * 2 + preyCounter.get_height() + generationCounter.get_height()))

    pygame.display.flip()

    clock.tick(FPS)

pygame.quit()


if check_folder_exists(folder_name):
    delete_folder(folder_name)

create_folder(folder_name)

sorted_population = sorted(e.predators, key=lambda x: x.get_fitness_score(), reverse=True)

torch.save(sorted_population[0].brain.state_dict(), os.path.join(folder_name, 'first_predator_brain.pth'))
torch.save(sorted_population[1].brain.state_dict(), os.path.join(folder_name, 'second_predator_brain.pth'))
print('Best predator brains are saved')
print('#1: ' + sorted_population[0].get_describe())
print('#2: ' + sorted_population[1].get_describe())

