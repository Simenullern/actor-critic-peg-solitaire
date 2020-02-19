class Cell:
    def __init__(self, loc):
        self.loc = loc
        self.filled = True
        self.neighbors = []

    def get_loc(self):
        return self.loc

    def has_peg(self):
        return self.filled

    def remove_peg(self):
        self.filled = False

    def put_peg(self):
        self.filled = True

    def add_neighbor(self, neighbor):
        self.neighbors.append(neighbor)

    def get_neighbors(self):
        return self.neighbors