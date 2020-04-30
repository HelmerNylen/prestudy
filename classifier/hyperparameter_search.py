#!/usr/bin/env python3
import argparse
import json
import os
import sys
import subprocess
import pickle
import numpy as np
from time import time
from itertools import product
from random import shuffle
from confusion_table import ConfusionTable
from shutil import copy2
from tempfile import NamedTemporaryFile

class ConfigPermutations:
	def __init__(self, search: dict):
		self.search = search
		self.types = list(search.keys())
		self.lengths = {t: dict() for t in self.types}
		self.totallengths = {t: 1 for t in self.types}
		for t in self.types:
			for category in self.search[t]:
				for key in self.search[t][category]:
					if isinstance(self.search[t][category][key], list):
						self.lengths[t][(category, key)] = len(self.search[t][category][key])
						self.totallengths[t] *= self.lengths[t][(category, key)]

		self.current_shuffle = list(range(len(self)))
		
	def __len__(self):
		return sum(self.totallengths.values())
	
	def shuffle(self):
		shuffle(self.current_shuffle)
	
	def __getitem__(self, x):
		assert 0 <= x < len(self)
		x = self.current_shuffle[x]
		_type = None
		for _t, l in self.totallengths.items():
			if x >= l:
				x -= l
			else:
				_type = _t
				break

		combination = list(tuple(product(*(range(l) for l in self.lengths[_type].values())))[x])
		combination_idx = 0
		config = dict()
		for category in self.search[_type]:
			config[category] = dict()
			for key in self.search[_type][category]:
				if isinstance(self.search[_type][category][key], list):
					config[category][key] = self.search[_type][category][key][combination[combination_idx]]
					combination_idx += 1
				else:
					config[category][key] = self.search[_type][category][key]
		return _type, config, x
	def __iter__(self):
		for x in range(len(self)):
			yield self[x]

def main(args):
	if args.json is None:
		args.json = os.path.join(args.classifier, "search.json")
	if args.saveto is None:
		args.saveto = os.path.join(args.classifier, "search")

	with open(args.json, "r") as f:
		search = json.load(f)

	permutations = ConfigPermutations(search)
	print(f"{len(permutations)} total combinations")
	if not os.path.exists(args.saveto):
		os.mkdir(args.saveto)
	
	with open(os.path.join(args.saveto, "search.json"), "w") as f:
		json.dump(search, f)
	
	permutations.shuffle()
	i = 0
	for current_classifier_type in ['LSTM', 'GMMHMM', 'GenHMM', 'SVM']: # permutations.types
		print(f"Classifier: {current_classifier_type}, {permutations.totallengths[current_classifier_type]} total configs")
		for classifier_type, config, x in permutations:
			if classifier_type != current_classifier_type:
				continue
			with NamedTemporaryFile("w") as configFile:
				print(f"Number {i} of {len(permutations)}")
				i += 1
				with open(configFile.name, "w") as f:
					json.dump({classifier_type: config}, f)

				# Train
				command = [
					os.path.join(args.classifier, "classifier.py"),
					"--models", args.saveto,
					"train",
					"-s",
					"-c", classifier_type.lower(),
					"--config", configFile.name,
					"-w", "current.classifier"
				]
				print(*command)
				result = subprocess.run(
					command,
					encoding=sys.getdefaultencoding(),
					stderr=subprocess.PIPE, stdout=subprocess.PIPE
				)
				while result.returncode and classifier_type.lower() == "genhmm" and config["train"]["batch_size"] > 16:
					# Try decreasing batch size and try again
					print(f"Decreasing batch size to {int(config['train']['batch_size'] / 2)} and trying again")
					config["train"]["batch_size"] = int(config["train"]["batch_size"] / 2)
					result = subprocess.run(
						command,
						encoding=sys.getdefaultencoding(),
						stderr=subprocess.PIPE, stdout=subprocess.PIPE
					)
				if result.returncode:
					print("Could not train", json.dumps({classifier_type: config}))
					print("Error:", result.stderr)
					continue
			# Test
			command = [
				os.path.join(args.classifier, "classifier.py"),
				"--models", args.saveto,
				"test",
				"--write-stdout",
				"current.classifier"
			]
			print(*command)
			result = subprocess.run(
				command,
				stdout=subprocess.PIPE
			)
			if result.returncode:
				print("Could not test", json.dumps({classifier_type: config}))
				continue
			confusion_table, = pickle.loads(result.stdout)
			with open(os.path.join(args.saveto, str(x) + ".confusiontable"), "wb") as f:
				pickle.dump(confusion_table, f)
			copy2(
				os.path.join(args.saveto, "current.classifier"),
				os.path.join(args.saveto, str(x) + ".classifier")
			)
	print("All done")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog="hyperparameter_search.py",
		description="Perform hyperparameter search on classifier.py"
	)
	parser.add_argument("-c", "--classifier", help="Prestudy classifier folder (default: $PWD/classifier)", default=os.path.join(os.getcwd(), "classifier"))
	parser.add_argument("-j", "--json", help="JSON file with search options (default: <CLASSIFIER>/search.json)", default=None)
	parser.add_argument("-s", "--saveto", help="Folder to save classifiers and stats to (default: <CLASSIFIER>/search)", default=None)

	args = parser.parse_args()
	start = time()
	main(args)
	print(f"Total time: {time() - start:.1f} s")