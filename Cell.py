

class Cell:
    def __init__(self, loc):
        self.loc = loc
        self.filled = True
        self.will_jump = False
        self.will_be_jumped_over = False
        self.will_be_removed = False
        self.neighbors = []

    def get_loc(self):
        return self.loc

    def add_neighbor(self, neighbor):
        self.neighbors.append(neighbor)

    def get_neighbors(self):
        return self.neighbors

    def can_jump(self, cell):
        pass
        #direction = get_direction()
        #if length == 2 and open neighbor