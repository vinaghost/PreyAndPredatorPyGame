from pyquadtree import QuadTree

from EntityBrain import EntityBrain
from entity import Entity
from settings import *
from pygame import time, Vector2

class Prey(Entity):
    def __init__(self, generation : int, brain: EntityBrain, position: Vector2):
        super().__init__(generation, brain, position, GREEN, DEFAULT_SPEED, PREY_VISION_ANGLE, PREY_SIGHT, PREY_SPLIT_REQUIRED)

        self.last_charge = 0

    def update_charge(self, current_tick: int) -> None:
        if self.last_charge == 0:
            self.last_charge = current_tick
        else:
            if current_tick - self.last_charge > COOLDOWN_SECONDS:
                self.split_energy += 1
                self.last_charge = current_tick


    def update(self, target_tree : QuadTree) -> None:
        current_tick = time.get_ticks()
        self.update_charge(current_tick)

        self.update_velocity(target_tree)


    def get_describe(self) -> str:
        return f'Prey {str(self.identity)[:8]} has lived for {self.life_time()} seconds and has {self.child_count} children'