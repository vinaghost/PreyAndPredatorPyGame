import uuid
from typing import List, Tuple
from pygame import Vector2, Surface, draw, Rect
from settings import *
from pyquadtree import QuadTree
from NeuralNetwork import NeuralNetwork
import numpy as np

class Entity:
    def __init__(self, identity: uuid.UUID, position: Tuple[int, int], color: Tuple[int, int, int], sight: int, lines_of_sight: range,  alliance: List['Entity'], enemy: List['Entity']):
        self.identity = identity
        self.position = Vector2(position)
        self.color = color
        self.speed = DEFAULT_SPEED
        self.energy = DEFAULT_ENERGY
        self.sight = sight
        self.lines_of_sight = lines_of_sight
        self.is_alive = True

        self.rotation = 0
        self.brain = NeuralNetwork()

        self.enemy = enemy
        self.alliance = alliance

    def draw(self, screen: Surface) -> None:
        draw.circle(screen, self.color, self.position, RADIUS)
        draw.line(screen, self.color, self.position, self.position + Vector2(0,1).rotate(self.rotation) * RADIUS * 2, 5)

    def update_energy(self, energy: int) -> None:
        self.energy += energy

    def scan(self, enemy_tree: QuadTree) -> np.ndarray:
        nearest_neighbors = enemy_tree.nearest_neighbors(tuple(self.position), max_distance=self.sight)
        result = np.empty([24])
        if len(nearest_neighbors) == 0:
            return result

        for i, angle in enumerate(self.lines_of_sight):
            rad = np.radians(angle + 90 + self.rotation)
            x = self.sight * np.cos(rad) + self.position.x
            y = self.sight * np.sin(rad) + self.position.y
            line = (self.position.x, self.position.y, x, y)
            max_score = 0
            for target in nearest_neighbors:
                target_rect = Rect(target.item[0] - RADIUS, target.item[1] - RADIUS, DIAMETER, DIAMETER)
                collision = target_rect.clipline(line)
                if not collision:
                    continue
                score = 1 - self.position.distance_to(Vector2(target.point)) / self.sight
                if score > max_score:
                    max_score = score

            result[i] = max_score

        return result

    def clamp_rotation(self) -> None:
        self.rotation = self.rotation % 360

    def move(self, data: np.ndarray) -> None:
        output = self.brain.predict(data)

        self.rotation += output[0] * 360
        self.clamp_rotation()

        speed = output[1] * self.speed

        self.position += Vector2(0,1).rotate(self.rotation) * speed

        if self.position[0] > WINDOW_WIDTH - RADIUS:
            self.position[0] = WINDOW_WIDTH - RADIUS
        elif self.position[0] < 0:
            self.position[0] = 0

        if self.position[1] > WINDOW_HEIGHT - RADIUS:
            self.position[1] = WINDOW_HEIGHT - RADIUS
        elif self.position[1] < 0:
            self.position[1] = 0

    def destroy(self) -> None:
        self.is_alive = False

    def spawn(self) -> None:
        pass

    def logic(self, enemy_tree: QuadTree) -> None:
        pass

    def update(self, screen: Surface, enemy_tree: QuadTree) -> None:
        self.logic(enemy_tree)

        data = self.scan(enemy_tree)
        self.move(data)

        if self.is_alive:
            self.draw(screen)
