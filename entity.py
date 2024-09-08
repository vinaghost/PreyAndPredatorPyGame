import uuid
from random import randint
from typing import Tuple

import numpy as np
import torch
from pygame import Vector2, Surface, draw, time, Rect
from pyquadtree import QuadTree
from pyquadtree.quadtree import Element

from EntityBrain import EntityBrain
from settings import *

class Entity:
    def __init__(self, generation: int, brain: EntityBrain, position: Vector2, color: Tuple[int, int, int], speed: int, vision_range: int, sight: int, split_required: int):
        self.identity = uuid.uuid4()

        self.generation = generation

        self.brain = brain
        self.position = position
        self.color = color
        self.speed = speed

        self.step_angle = vision_range // RAY_AMOUNT
        self.half_vision_angle = vision_range // 2
        self.sight = sight

        self.split_required = split_required

        self.rotation = randint(0, 360)
        self.velocity = Vector2(0, 0)

        self.tick_of_birth = time.get_ticks()
        self.tick_of_death = 0

        self.split_energy = 0
        self.child_count = 0


    def life_time(self) -> int:
        if self.tick_of_death == 0:
            return (time.get_ticks() - self.tick_of_birth) // 1000
        else:
            return (self.tick_of_death - self.tick_of_birth) // 1000

    def is_alive(self) -> bool:
        return self.tick_of_death == 0

    def clamp_rotation(self) -> None:
        self.rotation = self.rotation % 360

    def draw(self, screen: Surface) -> None:
        draw.circle(screen, self.color, self.position, RADIUS)
        draw.line(screen, self.color, self.position, self.position + Vector2(0,1).rotate(self.rotation) * RADIUS * 1.5, RADIUS // 5)

    def move(self, delta_time : float) -> None:
        self.position += self.velocity * delta_time

        if self.position[0] > WINDOW_WIDTH - RADIUS:
            self.position[0] = 0
        elif self.position[0] < 0:
            self.position[0] = WINDOW_WIDTH - RADIUS

        if self.position[1] > WINDOW_HEIGHT - RADIUS:
            self.position[1] = 0
        elif self.position[1] < 0:
            self.position[1] = WINDOW_HEIGHT - RADIUS

    def destroy(self) -> None:
        self.tick_of_death = time.get_ticks()

    def update(self, target_tree : QuadTree) -> None:
        pass

    def get_fitness_score(self) -> int:
        return self.child_count * 10 + self.life_time() // 10

    def get_describe(self) -> str:
        pass

    def update_rotation(self, target_tree : QuadTree) -> Vector2:
        data = self.scan(target_tree)
        tensor = torch.from_numpy(data).to(torch.float32)
        output = self.brain(tensor)
        output = output.detach().numpy()

        self.rotation += output[0] * 100
        self.clamp_rotation()

        return Vector2(0,1).rotate(self.rotation).normalize() * self.speed
    def update_overlap(self, tree : QuadTree) -> Vector2:
        bbox = (self.position.x - DIAMETER, self.position.y - DIAMETER, self.position.x + DIAMETER, self.position.y + DIAMETER)
        nearest_targets = tree.query(bbox) # type: list[Element]

        def is_valid_target(element : Element) -> bool:
            entity = element.item # type: Entity
            if not entity.is_alive():
                return False
            if not isinstance(entity, self.get_type()):
                return False
            return True
        n = Vector2(0, 0)
        for target in filter(is_valid_target, nearest_targets):
            dist = self.position.distance_to(Vector2(target.point))
            overlap = (DIAMETER - dist)/2
            direction = Vector2(target.point) - self.position
            if direction == Vector2(0, 0):
                continue
            n += direction.normalize() * overlap
        return n


    def update_velocity(self, tree : QuadTree) -> None:
        forward_velocity = self.update_rotation(tree)
        overlap_velocity = self.update_overlap(tree)

        self.velocity = forward_velocity - overlap_velocity

    def is_split_able(self) -> bool:
        return self.split_energy >= self.split_required
    def split(self) -> None:
        self.split_energy = 0
        self.child_count += 1

    def get_type(self) -> type:
        return type(self)

    def scan(self, tree : QuadTree) -> np.ndarray:
        bbox = (self.position.x - self.sight, self.position.y - self.sight, self.position.x + self.sight, self.position.y + self.sight)
        nearest_targets = tree.query(bbox) # type: list[Element]

        def is_valid_target(element : Element) -> bool:
            entity = element.item
            if not entity.is_alive():
                return False
            if isinstance(entity, self.get_type()):
                return False
            return True

        def create_rect(element : Element) -> Rect:
            return Rect(element.point[0] - RADIUS, element.point[1] - RADIUS, DIAMETER, DIAMETER)

        nearest_target_rects = [create_rect(target) for target in filter(is_valid_target, nearest_targets)]

        if len(nearest_target_rects) == 0:
            return np.zeros([RAY_AMOUNT])

        result = np.empty([RAY_AMOUNT])

        for i in range(RAY_AMOUNT):
            angle = (i + 1) * self.step_angle - self.half_vision_angle
            rad = np.radians(angle + 90 + self.rotation)

            x = PREDATOR_SIGHT * np.cos(rad) + self.position.x
            y = PREDATOR_SIGHT * np.sin(rad) + self.position.y
            line = (self.position.x, self.position.y, x, y)

            max_score = 0
            for target_rect in nearest_target_rects:
                collision = target_rect.clipline(line)
                if not collision:
                    continue
                score = 1 - self.position.distance_to(Vector2(target_rect.center)) / self.sight
                if score > max_score:
                    max_score = score

            result[i] = max_score
        return result
