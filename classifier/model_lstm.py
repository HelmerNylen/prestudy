import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
import numpy as np
from model import Model
from math import ceil

class LSTM(Model):
	def __init__(self, hidden_dim, noise_types, num_layers=1, dropout=0, learning_rate=1e-2):
		self.hidden_dim = hidden_dim
		self.num_layers = num_layers
		self.dropout = dropout
		self.learning_rate = learning_rate
		self.lstm = None
		self.loss = None
		self.optimizer = None
		self.noise_types = list(noise_types)

		self.iepoch = 1

	def get_noise_types(self):
		return self.noise_types

	MULTICLASS = True
	def train(self, train_data, batch_size=128, niter=4):
		if self.lstm is None:
			input_dim = train_data[self.noise_types[0]][0].shape[1]
			self.lstm = LSTMModule(input_dim, self.hidden_dim, len(self.noise_types),
					#batch_size,
					self.num_layers, self.dropout)
			self.loss = nn.NLLLoss()
			self.optimizer = torch.optim.Adam(self.lstm.parameters(), lr=self.learning_rate)
			
		#self.lstm.zero_hidden()

		dataset = LSTMDataset(train_data, self.noise_types)
		dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=LSTMCollate)
		for i in range(niter):
			total_loss = 0
			for batch_idx, (sequences, labels) in enumerate(dataloader):
				self.optimizer.zero_grad()
				predictions = self.lstm(sequences)
				loss = self.loss(predictions, labels)
				loss.backward()
				self.optimizer.step()

				total_loss += loss.detach().data

				#if batch_idx % (len(dataset) / batch_size // 20) == 0:
				#	print(f"Progress: {batch_idx}/{ceil(len(dataset) / batch_size)}", flush=True)
			print(f"Iteration {i}: NLL = {total_loss:.2f}")

		self.iepoch += 1

	def score(self, test_data, batch_size=128):
		if Model.is_concatenated(test_data):
			test_data = Model.split(test_data)
		
		dataset = LSTMDataset({0: test_data}, [0])
		dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=LSTMCollate)
		scores = []
		with torch.no_grad():
			for sequences, _ in dataloader:
				scores.extend(self.lstm(sequences).numpy())
		return np.array(scores)

# https://www.jessicayung.com/lstms-for-time-series-in-pytorch/
class LSTMModule(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim,
		#batch_size,
		num_layers, dropout):
		super(LSTMModule, self).__init__()
		#self.batch_size = batch_size
		self.hidden_dim = hidden_dim
		self.num_layers = num_layers

		self.lstm = nn.LSTM(input_dim, hidden_dim,
			num_layers=num_layers, dropout=dropout, batch_first=True)
		self.linear = nn.Linear(self.hidden_dim, output_dim)
		self.logsoftmax = nn.LogSoftmax(dim=1)
		#self.hidden = None
	
	#def zero_hidden(self):
	#	self.hidden = (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
	#				   torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))
	
	def forward(self, data):
		x, _ = self.lstm(data)
		x, l = pad_packed_sequence(x, batch_first=True)
		x = x[range(x.shape[0]), l-1, :]
		x = self.linear(x)
		x = self.logsoftmax(x)
		return x

class LSTMDataset(Dataset):
	def __init__(self, data, keys):
		self.data = [(sequence, i) for i, key in enumerate(keys) for sequence in data[key]]
		
	def __getitem__(self, index):
		return self.data[index]
	
	def __len__(self):
		return len(self.data)

def LSTMCollate(samples):
	sequences, labels = zip(*samples)
	sequences = pack_sequence([torch.from_numpy(s) for s in sequences], enforce_sorted=False)
	labels = torch.tensor(labels) # pylint: disable=E1102
	return sequences, labels