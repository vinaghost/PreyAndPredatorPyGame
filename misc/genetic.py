# https://github.com/ardalan-dsht/genetic-algorithm-for-pytorch/blob/main/genetic_algorithm.py

import random
from typing import List

import torch

from settings import *

def create_instance(entity_type, *args, **kwargs):
    return entity_type(*args, **kwargs)

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

