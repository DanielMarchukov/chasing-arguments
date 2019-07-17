from matplotlib import pyplot as plot
import networkx as nx
import itertools
import os


class ArgFramework:
    def __init__(self):
        self.__af = nx.DiGraph()
        self.__pos = None
        self.__colors = []
        self.__arguments = {}
        self.__labels = {}
        self.__size = 0

    def add_argument(self, arg):
        if arg not in self.__arguments:
            self.__af.add_node(self.__size + 1, sentence=arg)
            self.__size = self.__size + 1
            self.__arguments[arg] = self.__size
        return self.__size

    def add_relation(self, u, v, relation):
        self.__af.add_edge(self.__arguments[u], self.__arguments[v], color=relation)

    def save(self):
        self.__pos = nx.fruchterman_reingold_layout(self.__af)
        with open(os.getcwd() + '\\graph_data\\01_Argumentation_Framework_Colors.txt', mode="w") as file:
            for u, v in self.__af.edges():
                file.write(self.__af[u][v]['color'] + "\n")
        nx.write_gexf(self.__af, os.getcwd() + '\\graph_data\\01_Argumentation_Framework.gexf')

    def load(self):
        self.__af = nx.read_gexf('\\graph_data\\01_Argumentation_Framework.gexf')
        with open(os.getcwd() + '\\graph_data\\01_Argumentation_Framework_Colors.txt', mode="r") as file:
            for line in file:
                self.__colors.append(line)

    def draw(self):
        plot.axis('off')
        plot.figure(figsize=(15, 15))
        edges = self.__af.edges()
        self.__colors = [self.__af[u][v]['color'] for u, v in edges]
        layout = nx.spring_layout(self.__af, k=2, iterations=20)
        nx.draw(self.__af, layout, edges=edges, edge_color=self.__colors, node_size=150, alpha=0.75)
        nx.draw_networkx_labels(self.__af, layout, font_size=10)
        plot.savefig(os.getcwd() + '\\graph_data\\01_Argumentation_Framework_Graph_Spring.png', dpi=1000)
        plot.show()

    def conflict_free_arguments(self):
        arguments = []
        for node in self.__af.nodes():
            arguments.append(str(node))

        all_subsets = []
        for L in range(1, len(arguments) + 1):
            for subset in itertools.combinations(arguments, L):
                all_subsets.append(subset)

        conflict_sets = []
        for u, v in self.__af.edges():
            for subset in all_subsets:
                if set([str(u), str(v)]).issubset(subset):
                    if (self.__af.has_edge(u, v) and self.__af[u][v]['color'] == 'red') or \
                            (self.__af.has_edge(v, u) and self.__af[v][u]['color'] == 'red'):
                        conflict_sets.append(subset)

        conflict_free = []
        for subset in all_subsets:
            if subset not in conflict_sets:
                conflict_free.append(subset)
                print(subset)