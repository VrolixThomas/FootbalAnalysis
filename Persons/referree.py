from .person import Person

class Referree(Person):
    def __init__(self, trackId, frame_num):
        super().__init__(trackId, frame_num)
        
        