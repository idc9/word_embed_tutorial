import numpy as np
import heapq

def vec(word, embedding, w2i):
	"""
	Returns the vector for word as an array
	"""
	# if embedding is sparse
	# return counts[w2i[word], :].toarray().reshape(-1)
	return embedding[w2i[word], :]


def similarity(word1, word2, embedding, w2i, sim='angle'):
	"""
	Computes the similarity between two words
	
	Parameters
	----------
	word1, word2: words to compare
	sim: which similarity measure to use (angle, cosine, jaccard, dice)
	
	Returns
	-------
	similarity measure between two words
	"""
	
	v1 = vec(word1, embedding, w2i)
	v2 = vec(word2, embedding, w2i)


	if sim == 'angle':
		return angle_between(v1, v2)
	elif sim == 'cosine':
		return cosine_sim(v1, v2)
	elif sim == 'jaccard':
		return jaccard_sim(v1, v2)
	elif sim == 'dice':
		return dice_sim(v1, v2)
	else:
		raise ValueError('sim must be one of: angle, cosine, jaccard, dice')


def word_angles(word, embedding, w2i):
    w = vec(word, embedding, w2i)
    return [angle_between(embedding[i, :], w) for i in range(embedding.shape[0])]
        
def closest(word, embedding, w2i, N=10, just_words=True):
	"""
	Finds the closest words to a given word where distance
	is measured by angles
	
	Parameters
	-----------
	word:
	embedding: embedding matrix whose rows are word vectors
	w2i: dict mapping words to indices
	N: number of words to find
	just_words: whether or not to return just the words
	"""
	# w = vec(word, embedding, w2i)
	# angles = [angle_between(embedding[i, :], w) for i in range(embedding.shape[0])]
	angles = word_angles(word, embedding, w2i)

	i2w = [''] * len(w2i)
	for w in w2i.keys():
		i2w[w2i[w]] = w

	close = heapq.nsmallest(N, zip(angles, [i2w[i] for i in range(embedding.shape[0])]))

	if just_words:
		return [p[1] for p in close] # list(list(zip(*close))[1])
	else:
		return close


def cosine_sim(v, w):
	return np.dot(v, w) / np.sqrt(np.dot(v, v) * np.dot(w, w))

def angle_between(v, w, mod=False):
	cos_angle = cosine_sim(v, w)
	angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

	angle = np.degrees(angle)
	if mod:
		angle = min(angle, 90 - angle)

	return angle

def jaccard_sim(v, w):
	return np.minimum(v, w).sum()/np.maximum(v, w).sum()

def dice_sim(v, w):
	 return 2.0*np.minimum(v, w).sum() /(v + w).sum()