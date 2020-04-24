import numpy as np
from model import Model
try:
	from gm_hmm.src import ref_hmm
except ModuleNotFoundError:
	import sys
	print("gm_hmm not found in any of the following folders:")
	for p in sys.path:
		print(p)
	raise

class GMMHMM(Model):
	def __init__(self, *args, **kwargs):
		self.__gmm_hmm = ref_hmm.GMM_HMM(*args, **kwargs)
		self.__gmm_hmm.monitor_ = ref_hmm.ConvgMonitor(
			self.__gmm_hmm.tol,
			self.__gmm_hmm.n_iter,
			self.__gmm_hmm.verbose
		)
		self.__gmm_hmm.iepoch = 1

	def train(self, train_data, is_concatenated=False):
		if not is_concatenated:
			raise ValueError("GMMHMM requires concatenated samples")
		self.__gmm_hmm.fit(train_data[0], lengths=train_data[1])
		self.__gmm_hmm.iepoch += 1
	
	def score(self, test_data, is_concatenated=False):
		if is_concatenated:
			res = np.zeros(len(test_data[1]))
			ptr = 0
			for i, l in enumerate(test_data[1]):
				sequence = test_data[0][ptr:ptr + l, :]
				res[i] = self.__gmm_hmm.score(sequence)
				ptr += l
			return res
		else:
			return np.array([self.__gmm_hmm.score(sequence) for sequence in test_data])