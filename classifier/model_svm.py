from model import Model
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

class SVM(Model):
	MULTICLASS = True
	def __init__(self, config: dict, noise_types: list):
		# Consider https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel
		self.svm = SVC(
			**{key: val
				for category in ("parameters", "train")
					for key, val in config[category].items()
						if key not in ("frame_len", "frame_overlap")
			}
		)
		self.noise_types = list(noise_types)
		self.svm.classes_ = self.noise_types
		self.frame_len = config["parameters"]["frame_len"]
		self.frame_overlap = config["parameters"]["frame_overlap"]
		self.scaler = StandardScaler()

	def get_noise_types(self):
		return self.noise_types

	def train(self, train_data, config=None):
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
		if self.svm.verbose:
			print(f"Training SVM on {len(data)} vectors ...")
		self.scaler.fit(data)
		data = self.scaler.transform(data, True)
		self.svm.fit(data, labels)
		if self.svm.verbose:
			print(f"Done. Support vectors: {self.svm.n_support_}")
	
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
			framewise_prediction = self.svm.predict(self.scaler.transform(data, True))
			scores[i] = np.bincount(framewise_prediction, minlength=scores.shape[1]) # pylint: disable=E1136
		
		return scores