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
	def __init__(self, config: dict):
		if "tol" in config["train"] and isinstance(config["train"]["tol"], str):
			config["train"]["tol"] = {"-inf": -np.inf, "inf": np.inf}[config["train"]["tol"]]
		
		self.gmm_hmm = ref_hmm.GMM_HMM(**config["parameters"])
		self.gmm_hmm.monitor_ = ref_hmm.ConvgMonitor(
			*(config["train"][key] for key in ("tol", "n_iter", "verbose"))
		)
		self.gmm_hmm.iepoch = 1

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