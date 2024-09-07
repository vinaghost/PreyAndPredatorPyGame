from pygame import Vector2

from EntityBrain import EntityBrain
from predator import Predator
from prey import Prey

prey = Prey(0, EntityBrain(), Vector2(0,0))

predator = Predator(0, EntityBrain(), Vector2(0,0))


print((isinstance(prey, predator.get_type()))) # False
print((isinstance(predator, predator.get_type()))) # False
print((predator.get_type()))