from model import Model
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from random import sample

class SVM(Model):
	def __init__(self, frame_len, frame_overlap, noise_types):
		self.svm = SVC(kernel='rbf', verbose=2)
		self.noise_types = list(noise_types)
		self.svm.classes_ = self.noise_types
		self.frame_len = frame_len
		self.frame_overlap = frame_overlap
		self.scaler = StandardScaler()

	def get_noise_types(self):
		return self.noise_types

	MULTICLASS = True
	def train(self, train_data):
		data = []
		labels = []
		for noise_type, sequences in train_data.items():
			if Model.is_concatenated(sequences):
				sequences = Model.split(sequences)
			l = 0
			for sequence in sequences:
				for ptr in range(0, len(sequence) - self.frame_len, self.frame_len - self.frame_overlap):
					vector = sequence[ptr:ptr + self.frame_len].view()
					vector.shape = -1
					data.append(vector)
					l += 1
			labels.extend([self.noise_types.index(noise_type)] * l)
		print(f"Training SVM on {len(data)} vectors")
		self.scaler.fit(data)
		data = self.scaler.transform(data, True)
		self.svm.fit(data, labels)
	
	def score(self, test_data):
		if Model.is_concatenated(test_data):
			test_data = Model.split(test_data)

		scores = np.zeros((len(test_data), len(self.noise_types)))
		for i, sequence in enumerate(test_data):
			data = []
			for ptr in range(0, len(sequence) - self.frame_len, self.frame_len - self.frame_overlap):
				vector = sequence[ptr:ptr + self.frame_len].view()
				vector.shape = -1
				data.append(vector)
			scores[i] = np.bincount(self.svm.predict(self.scaler.transform(data, True)), minlength=scores.shape[1]) # pylint: disable=E1136
		
		return scores