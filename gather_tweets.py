import key_secret
import tweepy
import csv
import os
import re

from re import finditer
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag


class TwitterMining:
    def __init__(self):
        self.__tweets_csv_path = os.getcwd() + '\\data\\Twitter\\tweets.csv'
        self.__auth = tweepy.OAuthHandler(key_secret.consumer_key, key_secret.consumer_secret)
        self.__auth.set_access_token(key_secret.access_token, key_secret.access_token_secret)
        self.__api = tweepy.API(self.__auth, wait_on_rate_limit=True)
        self.__lem = WordNetLemmatizer()
        self.__stop_words = set(stopwords.words("english"))
        self.__tag_map = {
            'CC': wn.NOUN,  # coordin. conjunction (and, but, or)
            'CD': wn.NOUN,  # cardinal number (one, two)
            'DT': wn.NOUN,  # determiner (a, the)
            'EX': wn.ADV,  # existential ‘there’ (there)
            'FW': wn.NOUN,  # foreign word (mea culpa)
            'IN': wn.ADV,  # preposition/sub-conj (of, in, by)
            'JJ': wn.ADJ_SAT,  # adjective (yellow)
            'JJR': wn.ADJ_SAT,  # adj., comparative (bigger)
            'JJS': wn.ADJ_SAT,  # adj., superlative (wildest)
            'LS': wn.NOUN,  # list item marker (1, 2, One)
            'MD': wn.NOUN,  # modal (can, should)
            'NN': wn.NOUN,  # noun, sing. or mass (llama)
            'NNS': wn.NOUN,  # noun, plural (llamas)
            'NNP': wn.NOUN,  # proper noun, sing. (IBM)
            'NNPS': wn.NOUN,  # proper noun, plural (Carolinas)
            'PDT': wn.ADJ,  # predeterminer (all, both)
            'POS': wn.NOUN,  # possessive ending (’s )
            'PRP': wn.NOUN,  # personal pronoun (I, you, he)
            'PRP$': wn.NOUN,  # possessive pronoun (your, one’s)
            'RB': wn.ADV,  # adverb (quickly, never)
            'RBR': wn.ADV,  # adverb, comparative (faster)
            'RBS': wn.ADV,  # adverb, superlative (fastest)
            'RP': wn.ADJ,  # particle (up, off)
            'SYM': wn.NOUN,  # symbol (+,%, &)
            'TO': wn.NOUN,  # “to” (to)
            'UH': wn.NOUN,  # interjection (ah, oops)
            'VB': wn.VERB,  # verb base form (eat)
            'VBD': wn.VERB,  # verb past tense (ate)
            'VBG': wn.VERB,  # verb gerund (eating)
            'VBN': wn.VERB,  # verb past participle (eaten)
            'VBP': wn.VERB,  # verb non-3sg pres (eat)
            'VBZ': wn.VERB,  # verb 3sg pres (eats)
            'WDT': wn.NOUN,  # wh-determiner (which, that)
            'WP': wn.NOUN,  # wh-pronoun (what, who)
            'WP$': wn.NOUN,  # possessive (wh- whose)
            'WRB': wn.NOUN,  # wh-adverb (how, where)
            '$': wn.NOUN,  # dollar sign ($)
            '#': wn.NOUN,  # pound sign (#)
            '“': wn.NOUN,  # left quote (‘ or “)
            '”': wn.NOUN,  # right quote (’ or ”)
            '(': wn.NOUN,  # left parenthesis ([, (, {, <)
            ')': wn.NOUN,  # right parenthesis (], ), }, >)
            ',': wn.NOUN,  # comma (,)
            '.': wn.NOUN,  # sentence-final punc (. ! ?)
            ':': wn.NOUN  # mid-sentence punc (: ; ... – -)
        }

    @staticmethod
    def camel_case_split(identifier):
        matches = finditer('(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])', identifier)
        split_string = []
        previous = 0
        for match in matches:
            split_string.append(identifier[previous:match.start()])
            previous = match.start()
        split_string.append(identifier[previous:])
        return split_string

    def mine_tweets(self, query, since, count, lang="en"):
        with open(self.__tweets_csv_path, mode='w', encoding='utf-8', newline='') as file:
            file = csv.writer(file, delimiter='|', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            tweet_count = 0
            for tweet in tweepy.Cursor(self.__api.search, q=query, lang=lang, since=since, tweet_mode='extended',
                                       count=count, result_type='recent').items(count):
                if tweet.full_text.startswith("RT @") and hasattr(tweet, 'retweeted_status'):
                    text = tweet.retweeted_status.full_text
                else:
                    text = tweet.full_text

                if "http" in text:
                    index = text.index("http")
                    text = text[:0 - (len(text) - index)]

                filtered = ""
                text = text.replace(".", " ")
                words = text.split(' ')
                for word in words:
                    if "@" in word:
                        continue
                    word = self.remove_non_alphanumerical(word)
                    for w in self.camel_case_split(word):
                        if len(w) == 0 or w == " ":
                            continue
                        if not self.is_stop_word(w):
                            filtered += self.lemmatize(w) + " "
                filtered = self.filter_shorts(filtered)
                if filtered is not None:
                    file.writerow([tweet_count, tweet.created_at, tweet.retweet_count, tweet.favorite_count,
                                   filtered.lower(), text.replace('\n', ' ').replace('\r', '').strip()])
                    tweet_count = tweet_count + 1

    @staticmethod
    def remove_non_alphanumerical(word):
        if "&" in word:
            word = "and"
        word = word.replace("'", "")
        word = re.sub(r'[^\w]', '', word)
        return word

    def is_stop_word(self, word):
        if word in self.__stop_words:
            return True
        return False

    def lemmatize(self, word):
        tag = pos_tag([word])
        word = self.__lem.lemmatize(word, self.__tag_map[tag[0][1]])
        return word

    @staticmethod
    def filter_shorts(text):
        if len(text) > 30:
            text = " ".join(text.split())
            return text
        else:
            return None

    @property
    def tweets_csv_path(self):
        return self.__tweets_csv_path
