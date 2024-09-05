import random
from typing import List

import torch
import pprint
from NeuralNetwork import NeuralNetwork
from settings import CROSS_OVER_THRESHOLD, MUTATION_CHANGE_THRESHOLD

brain = NeuralNetwork()
brain.eval()
# Print model's state_dict
print("Model's state_dict:")
for param_tensor in brain.state_dict():
    print(param_tensor, "\t", brain.state_dict()[param_tensor].size())

count_of_initial_model_weights = sum(p.numel() for p in brain.parameters() if p.requires_grad)
print('Count of initial model weights:', count_of_initial_model_weights)



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




flatten_state_dict = deconstruct_statedict(brain)
print('Flatten statedict:', flatten_state_dict.size())
print('Flatten statedict:', flatten_state_dict.size()[0])
print('Flatten statedict:', flatten_state_dict.size(dim=0))

reconstructed_state_dict = reconstruct_statedict(flatten_state_dict, get_model_architecture(brain))

#print('Reconstructed statedict:', reconstructed_state_dict)
brain.load_state_dict(reconstructed_state_dict)
brain.eval()

