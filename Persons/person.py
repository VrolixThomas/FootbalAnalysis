class Person:
    def __init__(self, trackId, frame_num):
        self.trackId = trackId
        self.bboxes = [None for _ in range(frame_num)]

    def greet(self):
        return f'Hello, {self.name}!'
    
    def get_trackId(self):
        return self.trackId
    
    def add_bbox(self, bbox):
        self.bboxes.append(bbox)
    
    def __hash__(self):
        # Use the hash of trackId for the hash value of the object
        return hash(self.trackId)

    def __eq__(self, other):
        # Check if the other object is of the same type and has the same trackId
        if isinstance(other, Person):
            return self.trackId == other.trackId
        return False