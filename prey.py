import random
import uuid
from typing import Tuple
from entity import Entity
from settings import *
from pygame import time, Vector2
from environment import Environment


class Prey(Entity):
    def __init__(self, identity: uuid.UUID, position: Tuple[int, int] | Vector2, environment: Environment):
        super().__init__(identity, position, GREEN, PREY_SIGHT,range(-120, 120, 10), environment)
        self.child_count = 0
        self.last_charge = 0
        self.last_change_rotation = 0

    def spawn(self) -> None:
        if len(self.environment.preys) > PREY_MAX_COUNT:
            return
        self.environment.preys.append(Prey(uuid.uuid4(), self.position, self.environment))
        self.child_count += 1

    def logic(self) -> None:
        if self.energy > DEFAULT_ENERGY * 2:
            self.update_energy(-(DEFAULT_ENERGY * 2))
            self.spawn()

        current_tick = time.get_ticks()
        if self.last_charge == 0:
            self.last_charge = current_tick
        else:
            if current_tick - self.last_charge > 1000:
                self.update_energy(1)
                self.last_charge = current_tick

        if self.last_change_rotation == 0:
            self.last_change_rotation = current_tick
        else:
            if current_tick - self.last_change_rotation > 500:
                self.rotation += random.gauss() * 180
                self.last_change_rotation = current_tick

    def get_fitness_score(self) -> int:
        return self.child_count + self.life_time() * 2

    def get_describe(self) -> str:
        return f'Prey {str(self.identity)[:8]} has lived for {self.life_time()} seconds and has {self.child_count} children'