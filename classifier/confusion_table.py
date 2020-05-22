import numpy as np

class ConfusionTable:
	TOTAL = ...
	def __init__(self, true_labels, confused_labels):
		self.true_labels = np.array(true_labels)
		assert np.alltrue(self.true_labels == np.unique(true_labels))
		self.confused_labels = np.array(confused_labels)
		assert np.alltrue(self.confused_labels == np.unique(confused_labels))
		self.data = np.zeros((len(true_labels), len(confused_labels)))
		self.totals = np.zeros(len(true_labels))
		self.time = None
	
	def __map_indices(self, true_labels, confused_labels):
		if isinstance(true_labels, slice):
			first = slice(
				true_labels.start and np.argwhere(self.true_labels == true_labels.start)[0][0],
				true_labels.stop and np.argwhere(self.true_labels == true_labels.stop)[0][0],
				true_labels.step
			)
		elif isinstance(true_labels, str):
			first = np.argwhere(self.true_labels == true_labels)[0][0]
		elif true_labels is None:
			first = true_labels
		else:
			raise IndexError(f"Only strings or string slices are allowed (got: {repr(true_labels)})")

		if isinstance(confused_labels, slice):
			second = slice(
				confused_labels.start and np.argwhere(self.confused_labels == confused_labels.start)[0][0],
				confused_labels.stop and np.argwhere(self.confused_labels == confused_labels.stop)[0][0],
				confused_labels.step
			)
		elif isinstance(confused_labels, str):
			second = np.argwhere(self.confused_labels == confused_labels)[0][0]
		elif confused_labels is None or confused_labels is ConfusionTable.TOTAL:
			second = confused_labels
		else:
			raise IndexError(f"Only strings, string slices or Ellipsis are allowed (got: {repr(confused_labels)})")
		
		return first, second

	def __getitem__(self, labels):
		if len(labels) == 1:
			first_indices, second_indices = self.__map_indices(labels[0], None)
		else:
			first_indices, second_indices = self.__map_indices(labels[0], labels[1])
		
		if second_indices is None:
			return self.data[first_indices]
		elif second_indices is ConfusionTable.TOTAL:
			return self.totals[first_indices]
		else:
			return self.data[first_indices, second_indices]

	def __setitem__(self, labels, val):
		if len(labels) == 1:
			first_indices, second_indices = self.__map_indices(labels[0], None)
		else:
			first_indices, second_indices = self.__map_indices(labels[0], labels[1])
		
		if second_indices is None:
			self.data[first_indices] = val
		elif second_indices is ConfusionTable.TOTAL:
			self.totals[first_indices] = val
		else:
			self.data[first_indices, second_indices] = val
	
	# TODO: confusion tables brukar ritas upp transponerat
	def __str__(self):
		res = []
		totals_col = not np.alltrue(self.totals == 0)

		widths = [max(map(len, self.true_labels))] + [max(7, len(c)) for c in self.confused_labels]
		if totals_col:
			widths.append(max(len(f"{n:.0f}") for n in self.totals))

		res.append("  ".join(["true".center(widths[0], '-'), "guessed".center(sum(widths[1:1+len(self.confused_labels)]) + 2*(len(self.confused_labels) - 1), '-')]))
		res.append("  ".join(s.rjust(widths[i]) for i, s in enumerate([""] + list(self.confused_labels) + (["N"] if totals_col else []))))
		for row, true_label in enumerate(self.true_labels):
			res.append("  ".join(
				[true_label.rjust(widths[0])]
				+ [f"{val:>{widths[i+1]}.2f}" if self.totals[row] == 0 else f"{val/self.totals[row]:>{widths[i+1]}.2%}"
						for i, val in enumerate(self.data[row])]
				+ ([f"{self.totals[row]:.0f}"] if totals_col else [])
			))
		
		if self.time:
			res.append(f"Classification time: {self.time:.2f} s" + (f" ({1000 * self.time / sum(self.totals):.2f} ms per sequence)" if totals_col else ""))
		
		return "\n".join(res)

	def precision(self, label: str):
		positives = sum(self[l, label] for l in self.true_labels)
		if positives == 0:
			return 0
		else:
			return self[label, label] / positives

	def recall(self, label: str):
		return self[label, label] / self[label, ConfusionTable.TOTAL]

	def F1_score(self, label: str):
		_precision = self.precision(label)
		_recall = self.recall(label)
		if _precision == 0 and _recall == 0:
			return 0
		else:
			return 2 * _precision * _recall / (_precision + _recall)
	
	def accuracy(self):
		return sum(self[label, label] for label in self.true_labels) / sum(self.totals)

	def average(self, measure, equal_weights=False):
		if isinstance(measure, str):
			measure = {
				"precision": self.__class__.precision,
				"recall": self.__class__.recall,
				"f1": self.__class__.F1_score, "f1-score": self.__class__.F1_score, "f1_score": self.__class__.F1_score
			}[measure.lower()]
		if equal_weights:
			return sum(measure(self, label) for label in self.true_labels)\
						/ len(self.true_labels)
		else:
			return sum(measure(self, label) * self[label, ConfusionTable.TOTAL] for label in self.true_labels)\
						/ sum(self[label, ConfusionTable.TOTAL] for label in self.true_labels)