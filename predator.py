import uuid
from typing import List, Tuple
from entity import Entity
from settings import RED, RADIUS, DEFAULT_ENERGY, PREDATOR_MAX_COUNT, PREDATOR_SIGHT
from pyquadtree import QuadTree

class Predator(Entity):
    def __init__(self, identity: uuid.UUID, position: Tuple[int, int], alliance: List[Entity], enemy: List[Entity]):
        super().__init__(identity, position, RED, PREDATOR_SIGHT, range(-24, 24, 2), alliance, enemy)

    def spawn(self) -> None:
        if len(self.alliance) > PREDATOR_MAX_COUNT:
            return
        self.alliance.append(Predator(uuid.uuid4(), self.position, self.alliance, self.enemy))

    def logic(self, enemy_tree: QuadTree) -> None:
        if self.energy < 0:
            self.destroy()
            return
        if self.energy > DEFAULT_ENERGY * 3:
            self.update_energy(-(DEFAULT_ENERGY * 3))
            self.spawn()
            return

        nearest_neighbors = enemy_tree.nearest_neighbors(tuple(self.position), max_distance= 2 * RADIUS + RADIUS/100)
        if len(nearest_neighbors) > 0:
            target = next((e for e in self.enemy if e.identity == nearest_neighbors[0].item), None)

            if target:
                self.update_energy(target.energy)
                target.update_energy(-target.energy)
                target.destroy()
                enemy_tree.delete(target.identity)