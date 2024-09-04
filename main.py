import pygame
import sys
import torch
import random
from environment import Environment
from NeuralNetwork import NeuralNetwork
from settings import *

from prey import Prey
from predator import Predator
import uuid
def crossover(parent1 : NeuralNetwork, parent2 : NeuralNetwork):
    child1 = NeuralNetwork()
    child2 = NeuralNetwork()

    cross_over_idx = round(len(parent1.input.weight.data) * CROSS_OVER_THRESHOLD)
    child1.input.weight.data = torch.cat((parent1.input.weight.data[0:cross_over_idx], parent2.input.weight.data[cross_over_idx::]), dim=0)
    child2.input.weight.data = torch.cat((parent2.input.weight.data[0:cross_over_idx], parent1.input.weight.data[cross_over_idx::]), dim=0)

    #cross_over_idx = round(len(parent1.hidden.weight.data) * CROSS_OVER_THRESHOLD)
    #child1.hidden.weight.data = torch.cat((parent1.hidden.weight.data[0:cross_over_idx], parent2.hidden.weight.data[cross_over_idx::]), dim=0)
    #child2.hidden.weight.data = torch.cat((parent2.hidden.weight.data[0:cross_over_idx], parent1.hidden.weight.data[cross_over_idx::]), dim=0)

    cross_over_idx = round(len(parent1.output.weight.data) * CROSS_OVER_THRESHOLD)
    child1.output.weight.data = torch.cat((parent1.output.weight.data[0:cross_over_idx], parent2.output.weight.data[cross_over_idx::]), dim=0)
    child2.output.weight.data = torch.cat((parent2.output.weight.data[0:cross_over_idx], parent1.output.weight.data[cross_over_idx::]), dim=0)

    return child1, child2


def mutate(model: NeuralNetwork) -> NeuralNetwork:
    for param in model.parameters():
        if torch.rand(1).item() < 0.5:
            if torch.rand(1).item() < 0.5:
                param.data -= torch.randn_like(param.data) * MUTATION_CHANGE_THRESHOLD
            else:
                param.data += torch.randn_like(param.data) * MUTATION_CHANGE_THRESHOLD
    return model

def handle_predator(population: list[Predator]) -> list[NeuralNetwork]:
    sorted_population = sorted(population, key=lambda x: x.get_fitness_score(), reverse=True)

    threshold = round(len(sorted_population) * PERCENTAGE_OF_PARENTS_TO_KEEP)
    best_individuals_in_population = sorted_population[0:threshold]

    print(f'Best predator:  {best_individuals_in_population[0].get_describe()}')

    next_generation = []
    next_generation.extend([individual.brain for individual in best_individuals_in_population])

    count_of_children_needed = PREDATOR_COUNT - len(best_individuals_in_population) // 2 + 1

    # Crossover and mutation
    for idx in range(count_of_children_needed):
        parent1 = random.sample(best_individuals_in_population, k=1)[0]
        parent2 = random.sample(best_individuals_in_population, k=1)[0]
        child1, child2 = crossover(parent1.brain, parent2.brain)
        child1 = mutate(child1)
        child2 = mutate(child2)
        next_generation.extend([child1, child2])
    return next_generation

def initialize_population(environment : Environment):
    for _ in range(PREY_COUNT):
        environment.preys.append(Prey(uuid.uuid4(), (random.randint(0, WINDOW_WIDTH // (2 * DIAMETER)) * 2 * DIAMETER, random.randint(0, WINDOW_HEIGHT // (2 * DIAMETER)) * 2 * DIAMETER), environment))

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
initialize_population(e)
generation = 0

while True:
    if e.is_epoch_completed():
        print(f'Epoch {generation} is completed')
        if e.get_prey_count() == 0:
            print('All preys are dead')
        else:
            print('All predators are dead')

        e.kill_all()
        predator_brains = handle_predator(e.predators)
        recreate_population(e, predator_brains)
        generation += 1

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    e.update()
    generationCounter = fontObj.render(f'Generation {generation}', True, BLACK, TEXT_BACKGROUND)
    preyCounter = fontObj.render(f'Prey {e.get_prey_count()}', True, GREEN, TEXT_BACKGROUND)
    predatorCounter = fontObj.render(f'Predator {e.get_predator_count()}', True, RED, TEXT_BACKGROUND)

    SCREEN.blit(generationCounter, (TEXT_X, TEXT_Y))
    SCREEN.blit(preyCounter, (TEXT_X, TEXT_Y + TEXT_SPACING + generationCounter.get_height()))
    SCREEN.blit(predatorCounter, (TEXT_X, TEXT_Y + TEXT_SPACING * 2 + preyCounter.get_height() + generationCounter.get_height()))

    pygame.display.flip()

    clock.tick(FPS)