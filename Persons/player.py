from .person import Person


class Player(Person):
    def __init__(self, trackId, frame_num):
        super().__init__(trackId, frame_num)
