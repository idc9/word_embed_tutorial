import numpy as np
import heapq

def vec(word, M, w2i):
	"""
	Returns the vector for word as an array
	"""
	# if M is sparse
	# return counts[w2i[word], :].toarray().reshape(-1)
	return M[w2i[word], :]


def similarity(word1, word2, M, w2i, sim='angle'):
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
	
	v1 = vec(word1, M, w2i)
	v2 = vec(word2, M, w2i)


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


def closest(word, M, w2i, N=10, just_words=True):
	"""
	Finds the closest words to a given word where distance
	is measured by angles
	
	Parameters
	-----------
	word:
	M: embedding matrix whose rows are word vectors
	w2i: dict mapping words to indices
	N: number of words to find
	just_words: whether or not to return just the words
	"""
	w = vec(word, M, w2i)
	angles = [angle_between(M[i, :], w) for i in range(M.shape[1])]

	i2w = [''] * len(w2i)
	for w in w2i.keys():
		i2w[w2i[w]] = w

	close = heapq.nsmallest(N, zip(angles, [i2w[i] for i in range(M.shape[1])]))

	if just_words:
		return [p[1] for p in close] # list(list(zip(*close))[1])
	else:
		return close


def cosine_sim(v, w):
	return np.dot(v, w) / np.sqrt(np.dot(v, v) * np.dot(w, w))

def angle_between(v, w):
	cos_angle = cosine_sim(v, w)
	angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
	return np.degrees(angle)

def jaccard_sim(v, w):
	return np.minimum(v, w).sum()/np.maximum(v, w).sum()

def dice_sim(v, w):
	 return 2.0*np.minimum(v, w).sum() /(v + w).sum()