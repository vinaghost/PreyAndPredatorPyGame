import uuid
from typing import Tuple
from pygame import Vector2, Surface, draw,  time
from settings import *
from environment import Environment

class Entity:
    def __init__(self, identity: uuid.UUID, position: Tuple[int, int] | Vector2, color: Tuple[int, int, int], environment : Environment):
        self.identity = identity
        self.position = Vector2(position)
        self.color = color
        self.energy = DEFAULT_ENERGY
        self.is_alive = True

        self.velocity = Vector2(0, 0)

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

    def move(self, delta_time : float) -> None:
        self.clamp_rotation()
        self.position += self.velocity * delta_time

        if self.position[0] > WINDOW_WIDTH - RADIUS:
            self.position[0] = WINDOW_WIDTH - RADIUS - 10
            self.rotation += 180
        elif self.position[0] < 0:
            self.position[0] = 10
            self.rotation += 180

        if self.position[1] > WINDOW_HEIGHT - RADIUS:
            self.position[1] = WINDOW_HEIGHT - RADIUS - 10
            self.rotation += 180
        elif self.position[1] < 0:
            self.position[1] = 10
            self.rotation += 180

    def destroy(self) -> None:
        self.tick_of_death = time.get_ticks()
        self.is_alive = False

    def spawn(self) -> None:
        pass

    def logic(self, delta_time: float) -> None:
        pass

    def update(self, screen: Surface, delta_time : float) -> None:
        self.logic(delta_time)
        self.move(delta_time)

        if self.is_alive:
            self.draw(screen)

    def get_fitness_score(self) -> int:
        pass