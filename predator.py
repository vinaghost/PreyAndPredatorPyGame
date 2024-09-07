from pyquadtree import QuadTree
from pyquadtree.quadtree import Element

from entity import Entity
from settings import *
from EntityBrain import EntityBrain
from pygame import time, Vector2

class Predator(Entity):
    def __init__(self, generation: int, brain: EntityBrain, position: Vector2):
        super().__init__(generation, brain, position, RED, DEFAULT_SPEED*1.5, PREDATOR_VISION_ANGLE, PREDATOR_SIGHT, PREDATOR_SPLIT_REQUIRED)

        self.kill_count = 0

        self.energy = PREDATOR_INITIAL_ENERGY
        self.last_discharge = 0

        self.last_digestion = 0
        self.cooldown_digestion = False

    def update_discharge(self, current_tick: int) -> None:
        if self.last_discharge == 0:
            self.last_discharge = current_tick
        else:
            if current_tick - self.last_discharge > COOLDOWN_SECONDS:
                self.energy -= 1
                self.last_discharge = current_tick

                if self.energy < 0:
                    self.destroy()

    def update_digestion(self, current_tick: int) -> None:
        if not self.cooldown_digestion:
            return

        if current_tick - self.last_digestion > PREDATOR_DIGESTION_COOLDOWN:
            self.cooldown_digestion = False

    def update_collision(self, tree : QuadTree) -> None:
        dist = DIAMETER + RADIUS / 100
        bbox = (self.position.x - dist, self.position.y - dist, self.position.x + dist, self.position.y + dist)
        nearest_targets = tree.query(bbox)  # type: list[Element]

        def is_valid_target(element: Element) -> bool:
            entity = element.item
            if not entity.is_alive():
                return False
            if isinstance(entity, self.get_type()):
                return False
            return True

        target = next(filter(is_valid_target, nearest_targets), None)
        if target is None:
            return

        target.item.destroy()
        if self.cooldown_digestion:
            return

        self.split_energy += 1
        self.kill_count += 1
        self.cooldown_digestion = True
        self.last_digestion = time.get_ticks()
        self.energy = min(self.energy + PREY_ENERGY, PREDATOR_MAX_ENERGY)

    def update(self, target_tree : QuadTree) -> None:
        current_tick = time.get_ticks()
        self.update_discharge(current_tick)

        if not self.is_alive():
            return

        self.update_digestion(current_tick)
        self.update_collision(target_tree)
        self.update_velocity(target_tree)

    def get_describe(self) -> str:
        return f'Predator {str(self.identity)[:8]} has lived for {self.life_time()} seconds, has {self.child_count} children and has killed {self.kill_count} preys'