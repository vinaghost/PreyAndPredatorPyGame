import uuid
from typing import List, Tuple
from entity import Entity
from settings import GREEN, DEFAULT_ENERGY, PREY_MAX_COUNT, PREY_SIGHT, RADIUS
from pyquadtree import QuadTree

class Prey(Entity):
    def __init__(self, identity: uuid.UUID, position: Tuple[int, int], alliance: List[Entity], enemy: List[Entity]):
        super().__init__(identity, position, GREEN, PREY_SIGHT,range(-120, 120, 10), alliance, enemy)

    def spawn(self) -> None:
        if len(self.alliance) > PREY_MAX_COUNT:
            return
        self.alliance.append(Prey(uuid.uuid4(), self.position, self.alliance, self.enemy))

    def logic(self, enemy_tree: QuadTree) -> None:
        if self.energy > DEFAULT_ENERGY * 2:
            self.update_energy(-(DEFAULT_ENERGY * 2))
            self.spawn()