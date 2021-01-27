import pandas as pd

#https://github.com/minsuk-heo/python_tutorial/blob/master/data_science/nlp/word2vec_tensorflow.ipynb

# import and clean data
data = pd.read_csv('data/data_w_genres.csv')
corpus = data['genres']
corpus = corpus[corpus != '[]']
corpus = corpus.str.lstrip('[')
corpus = corpus.str.rstrip(']')
corpus = corpus.str.replace("'", '')

# generate a set of genres to vectorize
genres = []
for values in corpus:
    for genre in values.split(', '):
        genres.append(genre)
genres = set(genres)

# generate numeric label for each genre
genre2int = {}

for i,genre in enumerate(genres):
    genre2int[genre] = i
songgenres = []
for song in corpus:
    songgenres.append(song.split(", "))

# generate list of neighboring genres as a label
WINDOW_SIZE = 8

neighbors = []
for song in songgenres:
    for idx, genre in enumerate(song):
        for neighbor in song[max(idx - WINDOW_SIZE, 0) : min(idx + WINDOW_SIZE, len(song)) + 1] : 
            if neighbor != genre:
                neighbors.append([genre, neighbor])

genreneighbors = pd.DataFrame(neighbors, columns = ['input', 'label'])

# Define Tensorflow Graph

