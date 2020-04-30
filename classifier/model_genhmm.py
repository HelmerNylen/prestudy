import torch
from torch.utils.data import DataLoader
import numpy as np
from model import Model
try:
	from gm_hmm.src import genHMM
	from gm_hmm.src.utils import TheDataset
except ModuleNotFoundError:
	import sys
	print("gm_hmm not found in any of the following folders:")
	for p in sys.path:
		print(p)
	raise

class GenHMM(Model):
	def __init__(self, config: dict):
		self.genhmm = genHMM.GenHMM(**config["parameters"])
		self.genhmm.iepoch = 1
		self.genhmm.iclass = None

		self.__last_batch_size = None
	
	def __create_DataLoader(self, data, batch_size):
		lengths = [sequence.shape[0] for sequence in data]
		maxlen = max(lengths)
		padded_data = [np.pad(sequence, ((0, maxlen - sequence.shape[0]), (0, 0))) for sequence in data]
		return DataLoader(TheDataset(padded_data, lengths, device=self.genhmm.device), batch_size=batch_size, shuffle=True)

	def __try_push_cuda(self):
		if torch.cuda.is_available():
			try:
				device = torch.device('cuda')
				self.genhmm.device = device
				self.genhmm.pushto(self.genhmm.device)
				return True
			except:
				print("Unable to push to cuda device")
				raise
		return False

	def train(self, train_data, config):
		if Model.is_concatenated(train_data):
			train_data = Model.split(train_data)
		self.genhmm.device = 'cpu'
		if "force_cpu" not in config["train"] or not config["train"]["force_cpu"]:
			self.__try_push_cuda()
		print("Using", self.genhmm.device)

		self.__last_batch_size = config["train"]["batch_size"]

		data = self.__create_DataLoader(train_data, self.__last_batch_size)
		
		self.genhmm.number_training_data = len(train_data)
		self.genhmm.train()
		for i in range(config["train"]["n_iter"]):
			if "verbose" not in config["train"] or config["train"]["verbose"]:
				print(f"\tIteration {i}")
			self.genhmm.fit(data)
		
		self.genhmm.device = 'cpu'
		self.genhmm.pushto(self.genhmm.device)
		self.genhmm.iepoch += 1
	
	def score(self, test_data, batch_size=None, use_gpu=True):
		if Model.is_concatenated(test_data):
			test_data = Model.split(test_data)
		self.genhmm.device = 'cpu'
		if use_gpu:
			self.__try_push_cuda()

		batch_size = batch_size or self.__last_batch_size or 128

		data = self.__create_DataLoader(test_data, batch_size)

		self.genhmm.old_eval()
		self.genhmm.eval()
		scores = torch.cat([self.genhmm.pred_score(x) for x in data])
		return scores.cpu().numpy()