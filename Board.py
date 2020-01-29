from Cell import Cell
import networkx as nx
import matplotlib.pyplot as plt


class Board:
    def __init__(self, shape="triangle", size=5, open_start_cells=[(2,2)]):
        assert(shape == "diamond" or shape == "triangle")
        assert(4 <= size <= 8 if shape == "triangle" else 3 <= size <= 6)
        self.shape = shape
        self.size = size
        if shape == "diamond":
            self.cells = [[-1 for i in range(size)] for j in range(size)]
            self.potential_neighbors = [(-1, 1), (-1, 0), (0, -1), (0, 1), (1, 0), (1, -1)]
        else:
            self.cells = [[-1 for i in range(j)] for j in range(1, size+1)]
            self.potential_neighbors = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, 0), (1, 1)]
        self.open_start_cells = open_start_cells

    def init_board(self):
        for i in range(0, len(self.cells)):
            row = self.cells[i]
            for j in range(0, len(row)):
                row[j] = Cell(loc=(i,j))
        self.register_neighbors()
        for opener in self.open_start_cells:
            self.cells[opener[0]][opener[1]].filled = False

    def register_neighbors(self):
        for r in range(0, len(self.cells)):
            for c in range(0, len(self.cells[r])):
                current_cell = self.cells[r][c]
                for pn in self.potential_neighbors:
                    try:
                        if r+pn[0] < 0 or c + pn[1] < 0:
                            #print(str(r) + "," + str(c) + " cannot have neighbor " + str(pn) + " for shape " + str(self.shape))
                            continue
                        neighbor = self.cells[r+pn[0]][c+pn[1]]
                        #print(str(r) + "," + str(c) + " can indeed have neighbor " + str(pn) + " for shape " + str(self.shape))
                        current_cell.add_neighbor(neighbor)
                    except IndexError:
                        #print(str(r)+","+str(c) + " cant have neighbor " + str(pn) + " for shape " + str(self.shape))
                        continue


    def is_success(self):
        # Only one peg remains
        pass

    def visualize(self):
        G = nx.Graph()

        # Add Nodes
        for r in range(0, len(self.cells)):
            for c in range(0, len(self.cells[r])):
                G.add_node((r, c))
                # If empty different color

        # Add Edges
        for r in range(0, len(self.cells)):
            for c in range(0, len(self.cells[r])):
                current_cell = self.cells[r][c]
                current_loc = current_cell.get_loc()
                for neighbor in current_cell.get_neighbors():
                    G.add_edge(current_loc, neighbor.get_loc())


       # Fix the position

        options = {
        'node_color': 'blue',
        'node_size': 300,
        'width': 1,
        'fontsize': 8,

        }
        nx.draw(G, with_labels=True, **options)
        plt.show()



if __name__ == '__main__':
    board = Board()
    board.init_board()
    board.visualize()



