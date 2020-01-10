from tflearn.datasets import imdb
import pandas

# IMDB Dataset loading
train, test, _ = imdb.load_data(path='imdb.pkl', n_words=10000, valid_portion=0.1)

print (test[0][0:10])
