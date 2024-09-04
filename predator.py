import uuid
from typing import Tuple
from entity import Entity
from settings import *
from NeuralNetwork import NeuralNetwork
from environment import Environment
from pygame import time, Vector2, Rect
import torch
import numpy as np

class Predator(Entity):
    def __init__(self, identity: uuid.UUID, brain: NeuralNetwork, position: Tuple[int, int] | Vector2, environment: Environment):
        super().__init__(identity, position, RED, PREDATOR_SIGHT, range(-24, 24, 2), environment)

        self.kill_count = 0
        self.child_count = 0
        self.last_discharge = 0

        self.brain = brain

    def spawn(self) -> None:
        if len(self.environment.predators) > PREDATOR_MAX_COUNT:
            return
        self.environment.predators.append(Predator(uuid.uuid4(), self.brain, self.position, self.environment))
        self.child_count += 1

    def logic(self) -> None:
        if self.energy < 0:
            self.destroy()
            return

        if self.energy > DEFAULT_ENERGY * 3:
            self.update_energy(-(DEFAULT_ENERGY * 3))
            self.spawn()
            return

        nearest_neighbors = self.environment.prey_tree.nearest_neighbors(tuple(self.position), max_distance= 2 * RADIUS + RADIUS/100)
        if len(nearest_neighbors) > 0:
            target =  next((p for p in self.environment.preys if p.identity == nearest_neighbors[0].item), None)
            if target:
                self.update_energy(target.energy)
                target.update_energy(-target.energy)
                target.destroy()
                self.environment.prey_tree.delete(target.identity)
                self.kill_count += 1

        current_tick = time.get_ticks()
        if self.last_discharge == 0:
            self.last_discharge = time.get_ticks()
        else:
            if current_tick - self.last_discharge > 1000:
                self.update_energy(-1)
                self.last_discharge = time.get_ticks()

        data = self.scan()
        tensor = torch.from_numpy(data).to(torch.float32)
        output = self.brain(tensor)
        output = output.detach().numpy()

        self.rotation += output[0] * 360

    def scan(self) -> np.ndarray:
        nearest_neighbors = self.environment.prey_tree.nearest_neighbors(tuple(self.position), max_distance=self.sight)

        if len(nearest_neighbors) == 0:
            result = np.zeros([25])
            result[-1] = self.energy
            return result

        result = np.empty([25])
        for i, angle in enumerate(self.lines_of_sight):
            rad = np.radians(angle + 90 + self.rotation)
            x = self.sight * np.cos(rad) + self.position.x
            y = self.sight * np.sin(rad) + self.position.y
            line = (self.position.x, self.position.y, x, y)
            max_score = 0
            for target in nearest_neighbors:
                target_rect = Rect(target.point[0] - RADIUS, target.point[1] - RADIUS, DIAMETER, DIAMETER)
                collision = target_rect.clipline(line)
                if not collision:
                    continue
                score = 1 - self.position.distance_to(Vector2(target.point)) / self.sight
                if score > max_score:
                    max_score = score

            result[i] = max_score

        result[-1] = self.energy
        return result


    def get_fitness_score(self) -> int:
        return self.kill_count + self.child_count + self.life_time() * 2

    def get_describe(self) -> str:
        return f'Predator {str(self.identity)[:8]} has lived for {self.life_time()} seconds, has {self.child_count} children and has killed {self.kill_count} preys'