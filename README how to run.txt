Personal Developer Twitter account tokens need to be provided for Twitter API connectivity.
These have to be input in key_secret.py, four values are required:
consumer_key
consumer_secret
access_token
access_token_secret

Once you apply for Twitter Developer Account and are accepted, these can be generated on the Twitter account page.
Input them between single quotation marks.

Project was developed using PyCharm IDE.

Anaconda with Python3.7 needs to be installed for this to work.
Additionally, CUDA needs to be downloaded and installed from https://developer.nvidia.com/cuda-downloads
And cuDNN from https://developer.nvidia.com/cudnn

Alternatively, a guide that I followed can be found here, with pictures on how and where to install all of the nvidia tool kits: https://towardsdatascience.com/installing-tensorflow-with-cuda-cudnn-and-gpu-support-on-windows-10-60693e46e781

The following commands need to be run in the order specified, in a command line interface opened at the project directory:

conda create -n tf_gpu python=3.7 anaconda
activate tf_gpu
conda install -c conda-forge tweepy
conda install -c anaconda networkx
conda install -c anaconda tensorflow-gpu
conda install -c anaconda nltk
conda install -c anaconda numpy
conda install -c conda-forge matplotlib
conda install -c conda-forge tqdm
python
import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('popular')
exit()

Then to run the actual software, you have two options:

python main.py --training 1 --modelname newModelName --query YourSearchTerm

OR:

python main.py --training 0 --modelname RNN_vs200_b600_hs800_ml30 --query YourSearchTerm

The first option says that a new model should be trained, the model saved as newModelName and the query to search for tweets is YourSearchTerm and analyse the results.
The second option says that exclude training of the new model, instead load a model RNN_vs200_b600_hs800_ml30 and then with the existing model search tweets and analyse them with the term YourSearchTerm.
If you previously trained a different model, you can specify that name instead of the default RNN_vs200_b600_hs800_ml30.