import sys
sys.path.append('..')
import numpy as np
from common.util import most_similar, creat_co_matrix, ppmi
from dataset import ptb

window_size = 2
word2vec_size = 100

corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)

print('calculating co-matrix')
C = creat_co_matrix(corpus, vocab_size, window_size)
print('calclulating PPMI')
W = ppmi(C, verbose=True)

print('processing SVD')
try:
    from sklearn.utils.extmath import randomized_svd
    U, S, V = randomized_svd(W, n_components=word2vec_size, n_iter=5, random_state=None)
except ImportError:
    U, S, V = np.linalg.svd(W)

word_vecs = U[:, :word2vec_size]

querys = ['you', 'year', 'car', 'toyota']
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)