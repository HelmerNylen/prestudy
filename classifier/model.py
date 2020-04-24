import pickle
class Model():
	def __init__(self, *args, **kwargs):
		raise NotImplementedError()

	@staticmethod
	def from_file(filename):
		with open(filename, 'rb') as f:
			return pickle.load(f)
	def save_to_file(self, filename):
		with open(filename, 'wb') as f:
			pickle.dump(self, f)

	def train(self, train_data, is_concatenated=False):
		raise NotImplementedError()
	def test(self, test_data, is_concatenated=False):
		raise NotImplementedError()