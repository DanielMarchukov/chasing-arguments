from matplotlib import pyplot as plot
import networkx as nx


class ArgFramework:
    def __init__(self):
        self.__af = nx.DiGraph()
        self.__pos = None
        self.__colors = []
        self.__arguments = {}
        self.__size = 0
        # nx.draw_networkx_labels(self.__af, self.__pos, labels, font_size=16)

    def add_argument(self, arg):
        if arg not in self.__arguments:
            self.__af.add_node(self.__size + 1, sentence=arg)
            self.__size = self.__size + 1
            self.__arguments[arg] = self.__size
        return self.__size

    def add_relation(self, u, v, relation):
        self.__af.add_edge(self.__arguments[u], self.__arguments[v], color=relation)

    def save(self):
        with open('Argumentation_Framework_Colors.txt', mode="w") as file:
            for u, v in self.__af.edges():
                file.write(self.__af[u][v]['color'])
        nx.write_gexf(self.__af, 'Argumentation_Framework.gexf')

    def load(self):
        self.__af = nx.read_gexf('Argumentation_Framework.gexf')
        with open('Argumentation_Framework_Colors.txt', mode="r") as file:
            for line in file:
                self.__colors.append(line)

    def draw(self):
        self.__pos = nx.spring_layout(self.__af)
        edges = self.__af.edges()
        self.__colors = [self.__af[u][v]['color'] for u, v in edges]
        nx.draw(self.__af, self.__pos, edges=edges, edge_color=self.__colors)

        plot.axis('off')
        plot.show()
