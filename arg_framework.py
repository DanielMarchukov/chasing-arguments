from matplotlib import pyplot as plot
import networkx as nx
import itertools
import os

GRAPH_DATA_PATH = os.getcwd() + '\\graph_data\\01_Argumentation_Framework_Colors.txt'
GRAPH_COLORS_PATH = os.getcwd() + '\\graph_data\\01_Argumentation_Framework_Colors.txt'
IMAGE_PATH = os.getcwd() + '\\graph_data\\01_Argumentation_Framework_Graph_Spring.png'

ATTACK = 'red'
SUPPORT = 'green'


class ArgFramework:
    def __init__(self):
        self.__af = nx.DiGraph()
        self.__pos = None
        self.__colors = []
        self.__arguments = {}
        self.__labels = {}
        self.__size = 0
        self.__conflict_free = []
        self.__admissible = []
        self.__complete = []
        self.__grounded = set()
        self.__supporters = dict()
        self.__grounded_must_haves = []

    def get_af(self):
        return self.__af

    def add_argument(self, arg):
        if arg not in self.__arguments:
            self.__af.add_node(self.__size + 1, sentence=arg)
            self.__size = self.__size + 1
            self.__arguments[arg] = self.__size
        return self.__size

    def add_relation(self, u, v, relation):
        self.__af.add_edge(self.__arguments[u], self.__arguments[v], color=relation)

    def save(self, custom_path=None):
        path = custom_path if custom_path else GRAPH_DATA_PATH
        with open(path, mode="w") as file:
            for u, v in self.__af.edges():
                file.write(self.__af[u][v]['color'] + "\n")
        nx.write_gexf(self.__af, path)

    def load(self, custom_path=None):
        path = custom_path if custom_path else GRAPH_DATA_PATH
        self.__af = nx.read_gexf(path)
        with open(GRAPH_COLORS_PATH, mode="r") as file:
            for line in file:
                self.__colors.append(line)

    def draw(self):
        plot.axis('off')
        plot.figure(figsize=(10, 10))
        edges = self.__af.edges()
        self.__colors = [self.__af[u][v]['color'] for u, v in edges]
        self.__pos = nx.spring_layout(self.__af, k=2, iterations=20)
        nx.draw(self.__af, self.__pos, edges=edges, edge_color=self.__colors, node_size=1000, alpha=0.75)
        nx.draw_networkx_labels(self.__af, self.__pos, font_size=18)
        plot.savefig(os.getcwd() + '\\graph_data\\01_Argumentation_Framework_Graph_Spring.png', dpi=1000)
        plot.show()

    def pre_setup(self):
        for node in self.__af.nodes():
            self.__supporters[node] = self.get_supporters(node, [])
            must_have = True
            for u, v in self.__af.in_edges(node):
                if self.__af[u][v]['color'] == ATTACK:
                    must_have = False
                    break
            if must_have:
                self.__grounded_must_haves.append(node)

    def conflict_free_arguments(self):
        arguments = []
        for node in self.__af.nodes():
            arguments.append(node)

        for L in range(1, len(arguments) + 1):
            for subset in itertools.combinations(arguments, L):
                has_conflict = False
                for u, v in self.__af.edges():
                    if set([u, v]).issubset(subset):
                        if (self.__af.has_edge(u, v) and self.__af[u][v]['color'] == ATTACK) or \
                                (self.__af.has_edge(v, u) and self.__af[v][u]['color'] == ATTACK):
                            has_conflict = True
                            break
                if not has_conflict:
                    self.__conflict_free.append(subset)
                    # print(subset)
        return self.__conflict_free

    def admissible_arguments(self):
        for subset in self.__conflict_free:
            attackers = dict()
            for n in subset:
                for u, v in self.__af.in_edges(n):
                    if self.__af[u][v]['color'] == ATTACK:
                        attackers[u] = attackers.get(u, 0) + 1

            for att in attackers.keys():
                for deff in subset:
                    if self.__af.has_edge(deff, att):
                        if self.__af[deff][att]['color'] == ATTACK:
                            attackers[att] = attackers[att] - 1
                    else:
                        for supp in self.__supporters[att]:
                            if self.__af.has_edge(deff, supp) and self.__af[deff][supp]['color'] == ATTACK:
                                attackers[att] = attackers[att] - 1
            is_admissible = True
            for att in attackers.keys():
                if attackers[att] > 0:
                    is_admissible = False
                    break

            if is_admissible:
                self.__admissible.append(subset)
                # print(subset)
        return self.__admissible

    def get_supporters(self, a, nodelist):
        if a not in nodelist:
            for u, v in self.__af.in_edges(a):
                if self.__af[u][v]['color'] == SUPPORT:
                    nodelist.append(u)
                    nodelist = self.get_supporters(u, nodelist)
        return nodelist

    def complete_extension(self):
        added = dict()
        non_complete = []
        for i in range(len(self.__admissible) - 1, 0, -1):
            for j in range(i - 1, -1, -1):
                if j not in added.keys() and set(self.__admissible[j]).issubset(set(self.__admissible[i])):
                    non_complete.append(set(self.__admissible[j]))
                    added[j] = self.__admissible[j]

        for subset in self.__admissible:
            if set(subset) not in non_complete:
                self.__complete.append(subset)
                # print(subset)
        return self.__complete

    def grounded_extension(self):
        intersect = [self.__complete[-1]]
        for i in range(len(self.__complete) - 2, -1, -1):
            intersect.append(set(intersect[-1]).intersection(self.__complete[i]))
            if len(intersect[-1]) == 0:
                self.__grounded = set(intersect[-2]).union(self.__grounded_must_haves)
                return self.__grounded
        self.__grounded = set(intersect[-1]).union(self.__grounded_must_haves)
        return self.__grounded


# af = ArgFramework()
# af.add_argument("test1")
# af.add_argument("test2")
# af.add_argument("test3")
# af.add_argument("test4")
# af.add_relation("test1", "test2", SUPPORT)
# af.add_relation("test2", "test3", ATTACK)
# af.add_relation("test3", "test4", ATTACK)
# af.pre_setup()
# print("conflict frees:")
# af.conflict_free_arguments()
# print("admissibles:")
# af.admissible_arguments()
# print("completes:")
# af.complete_extension()
# print("grounded")
# print(af.grounded_extension())

