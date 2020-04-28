import pickle
class Model():
	def __init__(self, *args, **kwargs):
		raise NotImplementedError()

	MULTICLASS = False
	def get_noise_types(self):
		raise NotImplementedError()
	
	def train(self, train_data):
		raise NotImplementedError()
	def score(self, test_data):
		raise NotImplementedError()

	@staticmethod
	def from_file(filename):
		with open(filename, 'rb') as f:
			return pickle.load(f)
	def save_to_file(self, filename):
		with open(filename, 'wb') as f:
			pickle.dump(self, f)

	@staticmethod
	def concatenated(data):
		from feature_extraction import concat_samples
		return concat_samples(data)
	@staticmethod
	def split(data, lengths=None):
		from feature_extraction import split_samples
		return split_samples(data, lengths)
	@staticmethod
	def is_concatenated(data):
		return isinstance(data, tuple)