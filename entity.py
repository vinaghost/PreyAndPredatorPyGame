import uuid
from typing import Tuple
from pygame import Vector2, Surface, draw,  time
from settings import *
from environment import Environment

class Entity:
    def __init__(self, identity: uuid.UUID, position: Tuple[int, int] | Vector2, color: Tuple[int, int, int], sight: int, lines_of_sight: range, environment : Environment):
        self.identity = identity
        self.position = Vector2(position)
        self.color = color
        self.energy = DEFAULT_ENERGY
        self.sight = sight
        self.lines_of_sight = lines_of_sight
        self.is_alive = True

        self.rotation = 0

        self.environment = environment

        self.tick_of_birth = time.get_ticks()
        self.tick_of_death = 0

    def life_time(self) -> int:
        return (self.tick_of_death - self.tick_of_birth) // 1000

    def draw(self, screen: Surface) -> None:
        draw.circle(screen, self.color, self.position, RADIUS)
        draw.line(screen, self.color, self.position, self.position + Vector2(0,1).rotate(self.rotation) * RADIUS * 2, 5)

    def update_energy(self, energy: int) -> None:
        self.energy += energy

    def clamp_rotation(self) -> None:
        self.rotation = self.rotation % 360

    def move(self) -> None:
        self.clamp_rotation()
        self.position += Vector2(0,1).rotate(self.rotation) * DEFAULT_SPEED

        if self.position[0] > WINDOW_WIDTH - RADIUS:
            self.position[0] = WINDOW_WIDTH - RADIUS
        elif self.position[0] < 0:
            self.position[0] = 0

        if self.position[1] > WINDOW_HEIGHT - RADIUS:
            self.position[1] = WINDOW_HEIGHT - RADIUS
        elif self.position[1] < 0:
            self.position[1] = 0

    def destroy(self) -> None:
        self.tick_of_death = time.get_ticks()
        self.is_alive = False

    def spawn(self) -> None:
        pass

    def logic(self) -> None:
        pass

    def update(self, screen: Surface) -> None:
        self.logic()
        self.move()

        if self.is_alive:
            self.draw(screen)

    def get_fitness_score(self) -> int:
        pass