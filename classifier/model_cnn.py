import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from model import Model
from math import ceil

class CNN(Model):
	MULTICLASS = True
	def __init__(self, config: dict, noise_types: list):
		self.noise_types = tuple(noise_types)
		self.frame_len = config["parameters"]["frame_len"]
		self.frame_overlap = config["parameters"]["frame_overlap"]
		self.channels = config["parameters"]["channels"]
		self.kernel_size = config["parameters"]["kernel_size"]
		self.maxpool_kernel_size = config["parameters"]["maxpool_kernel_size"]
		self.maxpool_stride = self.maxpool_kernel_size if "maxpool_stride" not in config["parameters"] else config["parameters"]["maxpool_stride"]
		self.use_batchnorm = True if "use_batchnorm" not in config["train"] else config["train"]["use_batchnorm"]
		self.learning_rate = config["train"]["learning_rate"]
		self.conv2d_kwargs = dict() if "conv2d_kwargs" not in config["parameters"] else config["parameters"]["conv2d_kwargs"]

		self.cnn = None
		self.loss = None
		self.optimizer = None
		self.device = 'cpu'

		self.__last_batch_size = None
		if "feat_dim" in config["parameters"]:
			self.init_cnn(config["parameters"]["feat_dim"])
	
	def __try_push_cuda(self):
		if torch.cuda.is_available():
			try:
				device = torch.device('cuda')
				self.device = device
				self.cnn.to(device=self.device)
				return True
			except:
				print("Unable to push to cuda device")
				raise
		return False

	def init_cnn(self, feat_dim):
		if self.cnn is not None:
			raise Exception("CNN already initialized")
		self.cnn = CNNModule(feat_dim, self.frame_len, len(self.noise_types),
			self.channels, self.kernel_size, self.maxpool_kernel_size,
			self.maxpool_stride, self.use_batchnorm, **self.conv2d_kwargs)
		self.loss = nn.NLLLoss()
		self.optimizer = torch.optim.Adam(self.cnn.parameters(), lr=self.learning_rate)

	def get_noise_types(self):
		return self.noise_types

	def train(self, train_data, config):
		if self.cnn is None:
			self.init_cnn(train_data[self.noise_types[0]][0].shape[1])
		self.cnn.train()
		self.device = 'cpu'
		if "force_cpu" not in config["train"] or not config["train"]["force_cpu"]:
			self.__try_push_cuda()

		if "verbose" in config["train"] and config["train"]["verbose"]:
			print("Parameters:")
			total = 0
			for param in self.cnn.parameters():
				print(f"\t{type(param.data).__name__}\t{'x'.join(map(str, param.size()))}")
				total += np.prod(param.size())

			print(f"Total: {total} parameters")
			del total

		self.__last_batch_size = config["train"]["batch_size"]

		dataset = CNNDataset(train_data, self.noise_types, self.frame_len, self.frame_overlap)
		dataloader = DataLoader(dataset, batch_size=self.__last_batch_size, shuffle=True,
			collate_fn=CNNCollate, pin_memory=self.device != 'cpu')

		for i in range(config["train"]["n_iter"]):
			total_loss = 0
			for batch_idx, (frames, labels) in enumerate(dataloader):
				self.optimizer.zero_grad()
				predictions = self.cnn(frames.to(self.device))
				loss = self.loss(predictions, labels.to(self.device))
				loss.backward()
				self.optimizer.step()

				total_loss += loss.detach().data

				#if batch_idx % (len(dataset) / self.__last_batch_size // 20) == 0:
				#	print(f"Progress: {batch_idx}/{ceil(len(dataset) / self.__last_batch_size)}", flush=True)

			if "verbose" in config["train"] and config["train"]["verbose"]:
				print(f"Iteration {i}: NLL = {total_loss:.2f}")

		self.device = 'cpu'
		self.cnn.to(self.device)

	def score(self, test_data, batch_size=None, use_gpu=True):
		if Model.is_concatenated(test_data):
			test_data = Model.split(test_data)

		self.cnn.eval()
		self.device = 'cpu'
		if use_gpu:
			self.__try_push_cuda()

		batch_size = batch_size or self.__last_batch_size or 128
		
		dataset = CNNScoringDataset(test_data, self.frame_len, self.frame_overlap)
		dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
			collate_fn=CNNCollate, pin_memory=self.device != 'cpu')

		scores = np.zeros((len(test_data), len(self.noise_types)))

		with torch.no_grad():
			for frames, sequence_idxs in dataloader:
				framewise = self.cnn(frames.to(self.device)).cpu().numpy()
				framewise = np.argmax(framewise, axis=1)
				for sequence_idx in np.unique(sequence_idxs.cpu()):
					scores[sequence_idx, :] += np.bincount(framewise[sequence_idxs == sequence_idx], minlength=scores.shape[1]) # pylint: disable=E1136
		
		self.device = 'cpu'
		self.cnn.to(self.device)

		return np.array(scores)


class CNNModule(nn.Module):
	def __init__(self, feat_dim, num_feats, output_dim, channels, kernel_size,
			pool_kernel_size, pool_stride, batchnorm, **kwargs):
		super(CNNModule, self).__init__()
		
		channels = (1,) + tuple(channels)
		modules = []
		for i in range(1, len(channels)):
			modules.append(nn.Conv2d(channels[i-1], channels[i], kernel_size, **kwargs))
			modules.append(nn.ReLU())
			modules.append(nn.MaxPool2d(pool_kernel_size, stride=pool_stride))
			if batchnorm:
				modules.append(nn.BatchNorm2d(channels[i]))
		
		self.conv = nn.Sequential(*modules)
		
		pad = (0, 0)
		if "padding" in kwargs:
			pad = kwargs["padding"] if type(kwargs["padding"]) in (list, tuple) else (kwargs["padding"],) * 2
		dil = (1, 1)
		if "dilation" in kwargs:
			dil = kwargs["dilation"] if type(kwargs["dilation"]) in (list, tuple) else (kwargs["dilation"],) * 2
		ker = kernel_size if type(kernel_size) in (list, tuple) else (kernel_size,) * 2
		stride = (1, 1)
		if "stride" in kwargs:
			stride = kwargs["stride"] if type(kwargs["stride"]) in (list, tuple) else (kwargs["stride"],) * 2
		mp_ker = pool_kernel_size if type(pool_kernel_size) in (list, tuple) else (pool_kernel_size,) * 2
		mp_stride = pool_stride if type(pool_stride) in (list, tuple) else (pool_stride,) * 2
		H = feat_dim
		W = num_feats
		for c in channels[1:]:
			# Conv
			H = ((H + 2 * pad[0] - dil[0] * (ker[0] - 1) - 1) // stride[0]) + 1
			W = ((W + 2 * pad[1] - dil[1] * (ker[1] - 1) - 1) // stride[1]) + 1
			# Maxpool
			H = ((H - mp_ker[0]) // mp_stride[0]) + 1
			W = ((W - mp_ker[1]) // mp_stride[1]) + 1

		self.inner_dim = channels[-1] * H * W

		self.linear = nn.Linear(self.inner_dim, output_dim)
		self.logsoftmax = nn.LogSoftmax(dim=1)
	
	def forward(self, data):
		x = self.conv(data)
		x = self.linear(x.view(x.size(0), -1))
		x = self.logsoftmax(x)
		return x
		

class CNNDataset(Dataset):
	def __init__(self, data, keys, frame_len, frame_overlap):
		# TODO: allow concatenated sequences
		assert not Model.is_concatenated(data[keys[0]])
		self.data = [(sequence[ptr:ptr + frame_len], key_idx)
				for key_idx, key in enumerate(keys)
					for sequence in data[key]
						for ptr in range(0, len(sequence) - frame_len, frame_len - frame_overlap)]
		
	def __getitem__(self, index):
		return self.data[index]
	
	def __len__(self):
		return len(self.data)

class CNNScoringDataset(Dataset):
	def __init__(self, data, frame_len, frame_overlap):
		assert not Model.is_concatenated(data)
		self.data = [(sequence[ptr:ptr + frame_len], sequence_idx)
			for sequence_idx, sequence in enumerate(data)
				for ptr in range(0, len(sequence) - frame_len, frame_len - frame_overlap)]
		
	def __getitem__(self, index):
		return self.data[index]
	
	def __len__(self):
		return len(self.data)
	

def CNNCollate(samples):
	frames, labels = zip(*samples)
	frames = torch.cat([torch.from_numpy(frame).view(1, 1, *frame.shape) for frame in frames])
	labels = torch.tensor(labels) # pylint: disable=E1102
	return frames, labels