from Cell import Cell
import networkx as nx
import matplotlib.pyplot as plt
from random import randrange


class Board:
    def __init__(self, shape='triangle', size=5, open_start_cells=[(2,2), (0,0)]):
        assert(shape == "diamond" or shape == "triangle")
        assert(4 <= size <= 8 if shape == "triangle" else 3 <= size <= 6)
        self.shape = shape
        self.size = size
        if shape == "diamond":
            self.cells = [[-1 for i in range(size)] for j in range(size)]
            self.potential_neighbors = [(-1, 1), (-1, 0), (0, -1), (0, 1), (1, 0), (1, -1)]
        else:  # Shape is triangle
            self.cells = [[-1 for i in range(j)] for j in range(1, size + 1)]
            self.potential_neighbors = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, 0), (1, 1)]

        self.open_start_cells = open_start_cells



    def init_board(self):
        for i in range(0, len(self.cells)):
            row = self.cells[i]
            for j in range(0, len(row)):
                row[j] = Cell(loc=(i,j))
        self.register_neighbors()
        for opener in self.open_start_cells:
            self.cells[opener[0]][opener[1]].remove_peg()

    def register_neighbors(self):
        for r in range(0, len(self.cells)):
            for c in range(0, len(self.cells[r])):
                current_cell = self.cells[r][c]
                for pn in self.potential_neighbors:
                    try:
                        if r+pn[0] < 0 or c + pn[1] < 0:
                            continue  # Don't consider negative indexes
                        neighbor = self.cells[r+pn[0]][c+pn[1]]
                        current_cell.add_neighbor(neighbor)
                    except IndexError:
                        continue

    def get_remaining_pegs(self):
        num = 0
        for r in range(0, len(self.cells)):
            for c in range(0, len(self.cells[r])):
                current_cell = self.cells[r][c]
                if current_cell.has_peg():
                    num += 1
        return num

    def is_success(self):
        return self.get_remaining_pegs() == 1

    def get_all_possible_moves(self):
        possible_moves = []
        # Return list of tuples on the form (a, b) where a is the index to jump from and b is the index to jump to
        # for each cell, if it has a peg then get all neighbors.
        # If the neighbor has a peg
        # then check for neighbor if they have an open neighbor that is in the same direction

        for r in range(0, len(self.cells)):
            for c in range(0, len(self.cells[r])):
                current_cell = self.cells[r][c]
                current_cell_loc = current_cell.get_loc()
                if current_cell.has_peg():
                    for neighbor in current_cell.get_neighbors():
                        if neighbor.has_peg():
                            neighbor_loc = neighbor.get_loc()
                            neighbor_direction = (neighbor_loc[0] - current_cell_loc[0],
                                                  neighbor_loc[1] - current_cell_loc[1])
                            for neighbors_neighbor in neighbor.get_neighbors():
                                if not neighbors_neighbor.has_peg(): # and not neighbors_neighbor == current_cell: #Delete this later?
                                    neighbors_neighbor_loc = neighbors_neighbor.get_loc()
                                    neighbors_neighbor_direction = (neighbors_neighbor_loc[0] - neighbor_loc[0],
                                                                    neighbors_neighbor_loc[1] - neighbor_loc[1])
                                    if neighbor_direction == neighbors_neighbor_direction:
                                        possible_moves.append((current_cell_loc, neighbors_neighbor_loc))
        print(possible_moves)
        return possible_moves




    def visualize(self):
        G = nx.Graph()

        # Add Nodes
        node_colors = []
        for r in range(0, len(self.cells)):
            for c in range(0, len(self.cells[r])):
                current_cell = self.cells[r][c]
                if current_cell.has_peg():
                    node_colors.append('green')
                else:
                    node_colors.append('lightgray')
                G.add_node((r, c))

        # Add Edges
        for r in range(0, len(self.cells)):
            for c in range(0, len(self.cells[r])):
                current_cell = self.cells[r][c]
                current_loc = current_cell.get_loc()
                for neighbor in current_cell.get_neighbors():
                    G.add_edge(current_loc, neighbor.get_loc())

        # Draw the board
        options = {
            'node_size': 300,
            'font_size': 8,
            'font_color': 'black',
            'node_color': node_colors,
            'pos': nx.kamada_kawai_layout(G) if self.shape == 'triangle'
                else nx.spring_layout(G, seed=2),  # Making sure it looks decent
        }

        nx.draw(G, with_labels=True, **options)
        plt.gcf().set_tight_layout(True)
        plt.show()


if __name__ == '__main__':
    board = Board()
    board.init_board()
    board.visualize()
    possible_moves = board.get_all_possible_moves()



