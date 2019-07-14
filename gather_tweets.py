import key_secret
import tweepy
import csv
import os
import re

from re import finditer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
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
            '$': wn.NOUN,  #  dollar sign ($)
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

    def mine_data(self, query="Antifa", since="2019-07-10", count=10, lang="en"):
        with open(self.__tweets_csv_path, mode='w', encoding='utf-8', newline='') as file:
            file = csv.writer(file, delimiter='|', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            tweet_count = 0
            sentence_count = 0
            for tweet in tweepy.Cursor(self.__api.search, q=query, lang=lang, since=since, tweet_mode='extended',
                                       count=50).items(count):
                if tweet.full_text.startswith("RT @"):
                    text = tweet.retweeted_status.full_text
                else:
                    text = tweet.full_text

                if "http" in text:
                    index = text.index("http")
                    text = text[:0 - (len(text) - index)]

                tokenized = sent_tokenize(text)
                filtered = ""
                for sentence in tokenized:
                    sentence = re.sub(r'[^\w]', ' ', sentence)
                    words = word_tokenize(sentence)
                    for word in words:
                        for w in self.camel_case_split(word):
                            if w not in self.__stop_words:
                                tag = pos_tag([w])
                                w = self.__lem.lemmatize(w, self.__tag_map[tag[0][1]])
                                w = w.replace(".", "")
                                filtered += w + " "
                    if len(filtered) > 30:
                        sentence_count = sentence_count + 1
                        file.writerow([sentence_count, tweet_count, tweet.created_at, tweet.retweet_count,
                                       tweet.favorite_count, filtered + "."])
                    filtered = ""
                tweet_count = tweet_count + 1

    @property
    def tweets_csv_path(self):
        return self.__tweets_csv_path
