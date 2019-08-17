import unittest
from arg_framework import ArgFramework, ATTACK, SUPPORT


class TestArgFramework(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestArgFramework, self).__init__(*args, **kwargs)
        self.af = ArgFramework()
        self.af.add_argument("test1")
        self.af.add_argument("test2")
        self.af.add_argument("test3")
        self.af.add_argument("test4")
        self.af.add_relation("test1", "test2", SUPPORT)
        self.af.add_relation("test2", "test3", ATTACK)
        self.af.add_relation("test3", "test4", ATTACK)
        self.af.pre_setup()

    def test_conflict_free(self):
        actual_cf = self.af.conflict_free_arguments()
        expected_cf = [tuple([1]), tuple([2]), tuple([3]), tuple([4]), tuple([1, 2]), tuple([1, 3]), tuple([1, 4]),
                       tuple([2, 4]), tuple([1, 2, 4])]
        self.assertEqual(expected_cf, actual_cf)

    def test_admissible(self):
        self.af.conflict_free_arguments()
        actual_adm = self.af.admissible_arguments()
        expected_adm = [tuple([1]), tuple([2]), tuple([1, 2]), tuple([2, 4]), tuple([1, 2, 4])]
        self.assertEqual(expected_adm, actual_adm)

    def test_complete_extension(self):
        self.af.conflict_free_arguments()
        self.af.admissible_arguments()
        actual_ce = self.af.complete_extension()
        expected_ce = [tuple([1, 2, 4])]
        self.assertEqual(expected_ce, actual_ce)

    def test_grounded_extension(self):
        self.af.conflict_free_arguments()
        self.af.admissible_arguments()
        self.af.complete_extension()
        actual_ge = self.af.grounded_extension()
        expected_ge = set([1, 2, 4])
        self.assertEqual(actual_ge, expected_ge)


if __name__ == '__main__':
    unittest.main()
