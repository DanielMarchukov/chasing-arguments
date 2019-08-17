import unittest
import os
from gather_tweets import TwitterMining


class TestTwitterMining(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestTwitterMining, self).__init__(*args, **kwargs)
        self.twitter = TwitterMining()

    def test_query_search(self):
        self.twitter.mine_tweets(query="no deal", since="2019-08-12", count=1)
        with open(self.twitter.tweets_csv_path, mode='r', encoding='utf-8') as file:
            for line in file:
                self.assertEqual("deal" in line, True)

    def test_save_tweets(self):
        self.twitter.mine_tweets(query="#Brexit", since="2019-08-12", count=10)
        self.assertEqual(os.path.isfile(self.twitter.tweets_csv_path), True)

    def test_remove_non_alphanumerical(self):
        text = "!@#$%^*(*^"
        text = self.twitter.remove_non_alphanumerical(text)
        self.assertEqual(text, "")
        text = " & "
        text = self.twitter.remove_non_alphanumerical(text)
        self.assertEqual(text, "and")

    def test_remove_stop_word(self):
        text = ["a", "the", "an"]
        for t in text:
            self.assertEqual(self.twitter.is_stop_word(t), True)

    def test_lemmatize(self):
        word = 'loving'
        word = self.twitter.lemmatize(word)
        self.assertEqual(word, 'love')

    def test_filter_short_tweets(self):
        long_text = "This is a very long text that must qualify for this test case to pass successfully without issues"
        short_text = "Too short"
        long_text = self.twitter.filter_shorts(long_text)
        short_text = self.twitter.filter_shorts(short_text)
        self.assertEqual(long_text, "This is a very long text that must qualify for this test case to pass "
                                    "successfully without issues")
        self.assertEqual(short_text, None)

    def test_read_tweets(self):
        self.twitter.mine_tweets(query="#Brexit", since="2019-08-12", count=10)
        with open(self.twitter.tweets_csv_path, mode='r', encoding='utf-8') as file:
            for line in file:
                self.assertEqual(line is not None, True)


if __name__ == '__main__':
    unittest.main()
