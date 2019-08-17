import unittest
import os
from arg_framework import ArgFramework, IMAGE_PATH, SUPPORT, ATTACK
import matplotlib
import warnings
warnings.simplefilter("ignore", category=matplotlib.cbook.MatplotlibDeprecationWarning)


class GraphConstructionTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(GraphConstructionTest, self).__init__(*args, **kwargs)
        self.graph = ArgFramework()

    def test_add_node(self):
        self.graph.add_argument("test1")
        self.graph.add_argument("test2")
        self.assertEqual(2, len(self.graph.get_af().nodes()))

    def test_add_edge(self):
        self.graph.add_argument("test1")
        self.graph.add_argument("test2")
        self.graph.add_argument("test3")
        self.graph.add_argument("test4")
        self.graph.add_relation("test1", "test2", ATTACK)
        self.graph.add_relation("test2", "test3", ATTACK)
        self.graph.add_relation("test3", "test4", ATTACK)
        self.assertEqual(3, len(self.graph.get_af().edges()))

    def test_different_edges(self):
        self.graph.add_argument("test1")
        self.graph.add_argument("test2")
        self.graph.add_argument("test3")
        self.graph.add_argument("test4")
        self.graph.add_relation("test1", "test2", SUPPORT)
        self.graph.add_relation("test2", "test3", ATTACK)
        self.graph.add_relation("test3", "test4", ATTACK)
        self.assertEqual(self.graph.get_af()[1][2]['color'], 'green')
        self.assertEqual(self.graph.get_af()[2][3]['color'], 'red')
        self.assertEqual(self.graph.get_af()[3][4]['color'], 'red')

    def test_save_and_load_graph(self):
        self.graph.add_argument("test1")
        self.graph.add_argument("test2")
        self.graph.add_argument("test3")
        self.graph.add_argument("test4")
        self.graph.add_relation("test1", "test2", SUPPORT)
        self.graph.add_relation("test2", "test3", ATTACK)
        self.graph.add_relation("test3", "test4", ATTACK)
        self.graph.save(custom_path=os.getcwd() + '\\graph_data\\01_Test.gexf')
        self.assertEqual(os.path.isfile(os.getcwd() + '\\graph_data\\01_Test.gexf'), True)
        self.graph = None
        self.graph = ArgFramework()
        self.graph.load(custom_path=os.getcwd() + '\\graph_data\\01_Test.gexf')
        self.assertEqual(3, len(self.graph.get_af().edges()))
        self.assertEqual(4, len(self.graph.get_af().nodes()))

    def test_save_png_format(self):
        self.graph.add_argument("test1")
        self.graph.add_argument("test2")
        self.graph.add_argument("test3")
        self.graph.add_argument("test4")
        self.graph.add_argument("test5")
        self.graph.add_relation("test1", "test2", SUPPORT)
        self.graph.add_relation("test2", "test3", ATTACK)
        self.graph.add_relation("test3", "test4", ATTACK)
        self.graph.add_relation("test5", "test4", ATTACK)
        self.graph.add_relation("test5", "test1", SUPPORT)
        self.graph.draw()
        self.assertEqual(os.path.isfile(IMAGE_PATH), True)


if __name__ == '__main__':
    unittest.main()
