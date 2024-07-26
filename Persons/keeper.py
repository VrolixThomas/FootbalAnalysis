from .person import Person
from .player import Player

class Keeper(Player):
    def __init__(self, trackId, frame_num):
        super().__init__(trackId, frame_num)

    def __str__(self):
        return f'Keeper {self.name} is {self.age} years old and earns {self.salary}'

    