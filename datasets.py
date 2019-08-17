from six.moves.urllib.request import urlretrieve

import os
import zipfile

glove_6b_zip_file = os.getcwd() + "\\data\\GloVe\\glove.6B.zip"
glove_vectors_50d = os.getcwd() + "\\data\\GloVe\\glove.6B.50d.txt"
glove_vectors_100d = os.getcwd() + "\\data\\GloVe\\glove.6B.100d.txt"
glove_vectors_200d = os.getcwd() + "\\data\\GloVe\\glove.6B.200d.txt"
glove_vectors_300d = os.getcwd() + "\\data\\GloVe\\glove.6B.300d.txt"

glove_840b_zip_file = os.getcwd() + "\\data\\GloVe\\glove.840B.300d.zip"
glove_vectors_840B_300d = os.getcwd() + "\\data\\GloVe\\glove.840B.300d.txt"

glove_twitter_zip_file = os.getcwd() + "\\data\\GloVe\\glove.twitter.27B.zip"
glove_twitter_27B_25d = os.getcwd() + "\\data\\GloVe\\glove_twitter_27B_25d.txt"
glove_twitter_27B_50d = os.getcwd() + "\\data\\GloVe\\glove_twitter_27B_50d.txt"
glove_twitter_27B_100d = os.getcwd() + "\\data\\GloVe\\glove_twitter_27B_100d.txt"
glove_twitter_27B_200d = os.getcwd() + "\\data\\GloVe\\glove_twitter_27B_200d.txt"

snli_zip_file = os.getcwd() + "\\data\\SNLI\\snli_1.0.zip"
snli_dev_file = os.getcwd() + "\\data\\SNLI\\snli_1.0_dev.txt"
snli_test_file = os.getcwd() + "\\data\\SNLI\\snli_1.0_test.txt"
snli_full_dataset_file = os.getcwd() + "\\data\\SNLI\\snli_1.0_train.txt"


class CheckDownloadUnzipData:

    @staticmethod
    def unzip_single_file(zip_file_name, output_file_name):
        if not os.path.isfile(output_file_name):
            with open(output_file_name, 'wb') as out_file:
                with zipfile.ZipFile(zip_file_name) as zipped:
                    for info in zipped.infolist():
                        if output_file_name in info.filename:
                            with zipped.open(info) as requested_file:
                                out_file.write(requested_file.read())

    @staticmethod
    def check_glove_6b():
        if not os.path.isfile(glove_6b_zip_file) and (not os.path.isfile(glove_vectors_50d)
                                                      or not os.path.isfile(glove_vectors_100d)
                                                      or not os.path.isfile(glove_vectors_200d)
                                                      or not os.path.isfile(glove_vectors_300d)):
            urlretrieve("http://nlp.stanford.edu/data/glove.6B.zip", glove_6b_zip_file)

    @staticmethod
    def check_glove_840b():
        if not os.path.isfile(glove_840b_zip_file) and not os.path.isfile(glove_vectors_840B_300d):
            urlretrieve("http://nlp.stanford.edu/data/glove.840B.300d.zip", glove_840b_zip_file)

    @staticmethod
    def check_glove_twitter_27b():
        if not os.path.isfile(glove_twitter_zip_file) and (not os.path.isfile(glove_twitter_27B_25d)
                                                      or not os.path.isfile(glove_twitter_27B_50d)
                                                      or not os.path.isfile(glove_twitter_27B_100d)
                                                      or not os.path.isfile(glove_twitter_27B_200d)):
            urlretrieve("http://nlp.stanford.edu/data/glove.twitter.27B.zip", glove_twitter_zip_file)

    @staticmethod
    def check_snli():
        if not os.path.isfile(snli_zip_file) and (not os.path.isfile(snli_full_dataset_file)
                                                  or not os.path.isfile(snli_dev_file)
                                                  or not os.path.isfile(snli_test_file)):
            urlretrieve("https://nlp.stanford.edu/projects/snli/snli_1.0.zip", snli_zip_file)

    @staticmethod
    def unzip_all():
        CheckDownloadUnzipData.unzip_single_file(glove_6b_zip_file, glove_vectors_50d)
        CheckDownloadUnzipData.unzip_single_file(glove_6b_zip_file, glove_vectors_100d)
        CheckDownloadUnzipData.unzip_single_file(glove_6b_zip_file, glove_vectors_200d)
        CheckDownloadUnzipData.unzip_single_file(glove_6b_zip_file, glove_vectors_300d)
        CheckDownloadUnzipData.unzip_single_file(glove_840b_zip_file, glove_vectors_840B_300d)
        CheckDownloadUnzipData.unzip_single_file(glove_twitter_zip_file, glove_twitter_27B_25d)
        CheckDownloadUnzipData.unzip_single_file(glove_twitter_zip_file, glove_twitter_27B_50d)
        CheckDownloadUnzipData.unzip_single_file(glove_twitter_zip_file, glove_twitter_27B_100d)
        CheckDownloadUnzipData.unzip_single_file(glove_twitter_zip_file, glove_twitter_27B_200d)
        CheckDownloadUnzipData.unzip_single_file(snli_zip_file, snli_full_dataset_file)
        CheckDownloadUnzipData.unzip_single_file(snli_zip_file, snli_dev_file)
        CheckDownloadUnzipData.unzip_single_file(snli_zip_file, snli_test_file)

    @staticmethod
    def check_all_unzip():
        CheckDownloadUnzipData.check_glove_6b()
        CheckDownloadUnzipData.check_glove_840b()
        CheckDownloadUnzipData.check_glove_twitter_27b()
        CheckDownloadUnzipData.check_snli()
        CheckDownloadUnzipData.unzip_all()