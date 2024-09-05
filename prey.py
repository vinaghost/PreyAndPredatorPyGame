import random
import uuid
from typing import Tuple
from entity import Entity
from settings import *
from pygame import time, Vector2, math
from environment import Environment
import math

class Prey(Entity):
    def __init__(self, identity: uuid.UUID, position: Tuple[int, int] | Vector2, environment: Environment):
        super().__init__(identity, position, GREEN, environment)
        self.child_count = 0
        self.last_charge = 0
        self.rotation = random.randint(0, 360)

        self._half_vision_angle = PREY_VISION_ANGLE // 2

    def spawn(self) -> None:
        if len(self.environment.preys) > PREY_MAX_COUNT:
            return
        self.environment.preys.append(Prey(uuid.uuid4(), self.position, self.environment))
        self.environment.prey_tree.add(self.identity, tuple(self.position))
        self.child_count += 1

    def logic(self, delta_time: float) -> None:
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

        def is_in_sight(position: Vector2) -> bool:
            if self.position == position:
                return False
            forward = Vector2(0, 1).rotate(self.rotation)
            direction = self.position - position
            dot_product = forward.dot(direction.normalize())
            angle = math.acos(dot_product)
            return angle >= self._half_vision_angle

        def get_in_sight() -> list['Prey']:
            nearest_neighbors = self.environment.prey_tree.nearest_neighbors(tuple(self.position), max_distance=PREY_SIGHT)
            names = [p.item for p in nearest_neighbors if is_in_sight(p.point)]
            return [p for p in self.environment.preys if p.identity in names]

        def separation(neighbors: list['Prey']) -> Vector2:
            if len(neighbors) == 0:
                return Vector2(0, 0)
            direction = Vector2(0, 0)
            for neighbor in neighbors:
                ratio = self.position.distance_to(neighbor.position) / PREY_SIGHT
                direction -= ratio * (self.position - neighbor.position)
            return direction.normalize()

        def alignment(neighbors: list['Prey']) -> Vector2:
            if len(neighbors) == 0:
                direction = Vector2(0, 1).rotate(self.rotation)
                return direction.normalize()

            direction = Vector2(0, 0)
            for neighbor in neighbors:
                direction += Vector2(0, 1).rotate(neighbor.rotation)
            direction /= len(neighbors)
            return direction.normalize()

        def cohesion(neighbors: list['Prey']) -> Vector2:
            if len(neighbors) == 0:
                return Vector2(0, 0)

            center = Vector2(0, 0)
            for neighbor in neighbors:
                center += neighbor.position

            center /= len(neighbors)
            direction = center - self.position
            return direction.normalize()

        def calculate_velocity() -> Vector2:
            neighbors = get_in_sight()

            forward = Vector2(0, 1).rotate(self.rotation)
            velocity = forward + 1.7 * separation(neighbors) + 0.1 * alignment(neighbors) + cohesion(neighbors)
            return velocity.normalize()

        self.velocity = calculate_velocity() * DEFAULT_SPEED

    def get_fitness_score(self) -> int:
        return self.child_count * 10 + self.life_time() // 10

    def get_describe(self) -> str:
        return f'Prey {str(self.identity)[:8]} has lived for {self.life_time()} seconds and has {self.child_count} children'