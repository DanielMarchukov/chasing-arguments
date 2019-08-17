import unittest
import datasets
import os


class TestDatasets(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestDatasets, self).__init__(*args, **kwargs)
        self.ds = datasets.CheckDownloadUnzipData()

    def test_download_glove(self):
        self.ds.check_glove_840b()
        self.assertEqual(os.path.isfile(datasets.glove_840b_zip_file), True)

    def test_download_snli(self):
        self.ds.check_snli()
        self.assertEqual(os.path.isfile(datasets.snli_zip_file), True)

    def test_unzip_data(self):
        self.ds.unzip_all()
        self.assertEqual(os.path.isfile(datasets.glove_vectors_840B_300d), True)
        self.assertEqual(os.path.isfile(datasets.snli_full_dataset_file), True)
        self.assertEqual(os.path.isfile(datasets.snli_test_file), True)
        self.assertEqual(os.path.isfile(datasets.snli_dev_file), True)


if __name__ == '__main__':
    unittest.main()
