import unittest
import os
from textual_entailment import TextualEntailment


class TestLSTM(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestLSTM, self).__init__(*args, **kwargs)
        self.textentail = TextualEntailment()

    def test_textual_entailment_predictions(self):
        s1 = "New technologies are a good thing because it helps people."
        s2 = "We need innovative technologies to help people and bring good things"
        self.textentail.load_model(os.getcwd() + '\\models\\' + 'RNN_vs128_b256_hs768_ml30')
        res = self.textentail.run_predict(s1, s2)
        self.assertEqual(res, "E")


if __name__ == '__main__':
    unittest.main()
