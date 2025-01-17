import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
import numpy as np
from model import Model
from math import ceil

class LSTM(Model):
	MULTICLASS = True
	def __init__(self, config: dict, noise_types: list):
		self.hidden_dim = config["parameters"]["hidden_dim"]
		self.num_layers = config["parameters"]["num_layers"]
		self.dropout = config["parameters"]["dropout"]
		self.learning_rate = config["train"]["learning_rate"]
		self.lstm = None
		self.loss = None
		self.optimizer = None
		self.noise_types = list(noise_types)
		self.device = 'cpu'

		self.__last_batch_size = None

		if "input_dim" in config["parameters"]:
			self.init_lstm(config["parameters"]["input_dim"])
	
	def __try_push_cuda(self):
		if torch.cuda.is_available():
			try:
				device = torch.device('cuda')
				self.device = device
				self.lstm.to(device=self.device)
				return True
			except:
				print("Unable to push to cuda device")
				raise
		return False
	
	def __push_packed_sequence(self, sequence):
		#PackedSequence.to(device) is apparently bugged, see https://discuss.pytorch.org/t/cannot-move-packedsequence-to-gpu/57901
		if self.device == 'cpu':
			return sequence.cpu()
		else:
			return sequence.cuda(device=self.device)
	
	def init_lstm(self, input_dim):
		if self.lstm is not None:
			raise Exception("LSTM already initialized")
		self.lstm = LSTMModule(input_dim, self.hidden_dim, len(self.noise_types),
				self.num_layers, self.dropout)
		self.loss = nn.NLLLoss()
		self.optimizer = torch.optim.Adam(self.lstm.parameters(), lr=self.learning_rate)

	def get_noise_types(self):
		return self.noise_types

	def train(self, train_data, config):
		if self.lstm is None:
			self.init_lstm(train_data[self.noise_types[0]][0].shape[1])
		self.lstm.train()
		self.device = 'cpu'
		if "force_cpu" not in config["train"] or not config["train"]["force_cpu"]:
			self.__try_push_cuda()

		if "verbose" in config["train"] and config["train"]["verbose"]:
			print("Parameters:")
			total = 0
			for param in self.lstm.parameters():
				print(f"\t{type(param.data).__name__}\t{'x'.join(map(str, param.size()))}")
				total += np.prod(param.size())

			print(f"Total: {total} parameters")
			del total

		self.__last_batch_size = config["train"]["batch_size"]

		dataset = LSTMDataset(train_data, self.noise_types)
		dataloader = DataLoader(dataset, batch_size=self.__last_batch_size,
				shuffle=True, collate_fn=LSTMCollate, pin_memory=self.device != 'cpu')
		for i in range(config["train"]["n_iter"]):
			total_loss = 0
			for batch_idx, (sequences, labels) in enumerate(dataloader):
				self.optimizer.zero_grad()
				
				predictions = self.lstm(self.__push_packed_sequence(sequences))
				loss = self.loss(predictions, labels.to(self.device))
				loss.backward()
				self.optimizer.step()

				total_loss += loss.detach().data

				#if batch_idx % (len(dataset) / self.__last_batch_size // 20) == 0:
				#	print(f"Progress: {batch_idx}/{ceil(len(dataset) / self.__last_batch_size)}", flush=True)
			if "verbose" in config["train"] and config["train"]["verbose"]:
				print(f"Iteration {i}: NLL = {total_loss:.2f}")
		
		self.device = 'cpu'
		self.lstm.to(self.device)

	def score(self, test_data, batch_size=None, use_gpu=True):
		if Model.is_concatenated(test_data):
			test_data = Model.split(test_data)

		self.lstm.eval()
		self.device = 'cpu'
		if use_gpu:
			self.__try_push_cuda()

		batch_size = batch_size or self.__last_batch_size or 128
		
		dataset = LSTMDataset({0: test_data}, [0])
		dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
			collate_fn=LSTMCollate, pin_memory=self.device != 'cpu')
		scores = []
		with torch.no_grad():
			for sequences, _ in dataloader:
				scores.extend(self.lstm(self.__push_packed_sequence(sequences))\
					.cpu().numpy())

		self.device = 'cpu'
		self.lstm.to(self.device)

		return np.array(scores)

# Initially based on https://www.jessicayung.com/lstms-for-time-series-in-pytorch/
class LSTMModule(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
		super(LSTMModule, self).__init__()
		self.hidden_dim = hidden_dim
		self.num_layers = num_layers

		self.lstm = nn.LSTM(input_dim, hidden_dim,
			num_layers=num_layers, dropout=dropout, batch_first=True)
		self.linear = nn.Linear(self.hidden_dim, output_dim)
		self.logsoftmax = nn.LogSoftmax(dim=1)
	
	def forward(self, data):
		x, _ = self.lstm(data)
		x, l = pad_packed_sequence(x, batch_first=True)
		x = x[range(x.shape[0]), l-1, :]
		x = self.linear(x)
		x = self.logsoftmax(x)
		return x

class LSTMDataset(Dataset):
	def __init__(self, data, keys):
		# TODO: allow concatenated sequences
		assert not Model.is_concatenated(data[keys[0]])
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