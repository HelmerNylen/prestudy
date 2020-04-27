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
	def __init__(self, *args, **kwargs):
		self.__genhmm = genHMM.GenHMM(*args, **kwargs)
		self.__genhmm.iepoch = 1
		self.__genhmm.iclass = None
	
	def __create_DataLoader(self, data, batch_size):
		lengths = [sequence.shape[0] for sequence in data]
		maxlen = max(lengths)
		padded_data = [np.pad(sequence, ((0, maxlen - sequence.shape[0]), (0, 0))) for sequence in data]
		return DataLoader(TheDataset(padded_data, lengths, device=self.__genhmm.device), batch_size=batch_size, shuffle=True)

	def __try_push_cuda(self):
		if torch.cuda.is_available():
			try:
				device = torch.device('cuda')
				self.__genhmm.device = device
				self.__genhmm.pushto(self.__genhmm.device)
				return True
			except:
				print("Unable to push to cuda device")
				raise
		return False

	def train(self, train_data, batch_size=128, n_iter=3):
		if Model.is_concatenated(train_data):
			train_data = Model.split(train_data)
		self.__genhmm.device = 'cpu'
		self.__try_push_cuda()
		print("Using", self.__genhmm.device)

		data = self.__create_DataLoader(train_data, batch_size)
		
		self.__genhmm.number_training_data = len(train_data)
		self.__genhmm.train()
		for i in range(n_iter):
			print(f"\tIteration {i}")
			self.__genhmm.fit(data)
		
		self.__genhmm.device = 'cpu'
		self.__genhmm.pushto(self.__genhmm.device)
		self.__genhmm.iepoch += 1
	
	def score(self, test_data, batch_size=128):
		if Model.is_concatenated(test_data):
			test_data = Model.split(test_data)
		self.__genhmm.device = 'cpu'
		self.__try_push_cuda()

		data = self.__create_DataLoader(test_data, batch_size)

		self.__genhmm.old_eval()
		self.__genhmm.eval()
		scores = torch.cat([self.__genhmm.pred_score(x) for x in data])
		return scores.cpu().numpy()