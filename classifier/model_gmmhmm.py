import numpy as np
from random import shuffle
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
	def __init__(self, config: dict):
		if "tol" in config["train"] and isinstance(config["train"]["tol"], str):
			config["train"]["tol"] = {"-inf": -np.inf, "inf": np.inf}[config["train"]["tol"]]
		
		self.gmm_hmm = RandomInitGMM_HMM(**config["parameters"])
		self.gmm_hmm.monitor_ = ref_hmm.ConvgMonitor(
			*(config["train"][key] for key in ("tol", "n_iter", "verbose"))
		)
		self.gmm_hmm.iepoch = 1
		self.gmm_hmm._set_rand_inits(
			config["train"]["weight_rand_init"] if "weight_rand_init" in config["train"] else 0,
			config["train"]["mean_rand_init"] if "mean_rand_init" in config["train"] else 0,
			config["train"]["covar_rand_init"] if "covar_rand_init" in config["train"] else 0
		)

	def train(self, train_data, config=None):
		if not Model.is_concatenated(train_data):
			train_data = Model.concatenated(train_data)
		self.gmm_hmm.fit(train_data[0], lengths=train_data[1])
		self.gmm_hmm.iepoch += 1
	
	def score(self, test_data):
		if Model.is_concatenated(test_data):
			res = np.zeros(len(test_data[1]))
			ptr = 0
			for i, l in enumerate(test_data[1]):
				sequence = test_data[0][ptr:ptr + l, :]
				res[i] = self.gmm_hmm.score(sequence)
				ptr += l
			return res
		else:
			return np.array([self.gmm_hmm.score(sequence) for sequence in test_data])

class RandomInitGMM_HMM(ref_hmm.GMM_HMM):
	def _set_rand_inits(self, weight_rand_init, mean_rand_init, covar_rand_init):
		self.weight_rand_init = weight_rand_init
		self.mean_rand_init = mean_rand_init
		self.covar_rand_init = covar_rand_init

	def _init(self, X, lengths):
		super()._init(X, lengths=lengths)
		w_add = self.weight_rand_init * np.random.randn(*self.weights_.shape)
		m_add = self.mean_rand_init * np.random.randn(*self.means_.shape)
		c_add = self.covar_rand_init * np.abs(np.random.randn(*self.covars_.shape))

		if 'w' not in self.init_params:
			self.weights_ = w_add
			if self.weight_rand_init == 0:
				self.weights_ += 1
		else:
			self.weights_ += w_add
		self.weights_ = np.abs(self.weights_)
		self.weights_ = self.weights_ / self.weights_.sum(axis=1)[:, None]
		
		if 'm' not in self.init_params:
			self.means_ = m_add
		else:
			self.means_ += m_add

		if 'c' not in self.init_params:
			self.covars_ = c_add
		else:
			self.covars_ += c_add