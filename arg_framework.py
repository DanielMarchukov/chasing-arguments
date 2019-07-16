from matplotlib import pyplot as plot
import networkx as nx
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
        plot.figure(num=1, figsize=(15, 15))
        edges = self.__af.edges()
        self.__colors = [self.__af[u][v]['color'] for u, v in edges]

        layout = nx.spring_layout(self.__af, k=0.7, iterations=20)
        nx.draw(self.__af, layout, edges=edges, edge_color=self.__colors, node_size=50, alpha=0.75)
        nx.draw_networkx_labels(self.__af, layout, font_size=9)
        plot.savefig(os.getcwd() + '\\graph_data\\01_Argumentation_Framework_Graph_Spring.png', dpi=1000)

        layout = nx.kamada_kawai_layout(self.__af)
        plot.figure(num=3, figsize=(15, 15))
        nx.draw(self.__af, layout, edges=edges, edge_color=self.__colors, node_size=50, alpha=0.75)
        nx.draw_networkx_labels(self.__af, layout, font_size=9)
        plot.savefig(os.getcwd() + '\\graph_data\\01_Argumentation_Framework_Graph_Kamada.png', dpi=1000)
        plot.show()

    def conflict_free_arguments(self):
        iterate = 1
