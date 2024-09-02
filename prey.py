from entity import *

class Prey(Entity):
    def __init__(self, position, speed, color, energy):
        super().__init__(position, speed, color, energy)

    def destroy(self):
        preys.remove(self)

    def spawn(self):
        x = random.randint(0, 10)

        if x % 2 == 0:
            preys.append(Prey([random.randint(0, WINDOW_WIDTH ), random.randint(0, WINDOW_HEIGHT )], SPEED, GREEN, 10))
