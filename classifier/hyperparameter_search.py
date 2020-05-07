#!/usr/bin/env python3
import argparse
import json
import os
import sys
import select
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

	def _get_combination(self, x):
		assert 0 <= x < len(self)
		x = self.current_shuffle[x]
		inner_x = x
		_type = None
		for _t, l in self.totallengths.items():
			if x >= l:
				x -= l
			else:
				_type = _t
				break

		return _type, list(tuple(product(*(range(l) for l in self.lengths[_type].values())))[x]), inner_x

	def __getitem__(self, x):
		_type, combination, inner_x = self._get_combination(x)
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
		return _type, config, inner_x

	def __iter__(self):
		for x in range(len(self)):
			yield self[x]

def train_test_config(classifier_type, config, x, genhmm_min_batch, intermediate_name):
	if genhmm_min_batch < 1: genhmm_min_batch = 1
	with NamedTemporaryFile("w") as configFile:
		with open(configFile.name, "w") as f:
			json.dump({classifier_type: config}, f)
		print(f"{x}: {config}")

		# Train
		command = [
			os.path.join(args.classifier, "classifier.py"),
			"--models", args.saveto,
			"train",
			"-s",
			"-c", classifier_type.lower(),
			"--config", configFile.name,
			"-w", intermediate_name
		]
		print(*command)
		result = subprocess.run(
			command,
			encoding=sys.getdefaultencoding(),
			stderr=subprocess.PIPE, stdout=subprocess.PIPE
		)
		while result.returncode and classifier_type.lower() == "genhmm" and int(config["train"]["batch_size"]/2) >= 1:
			# Try decreasing batch size and try again
			config["train"]["batch_size"] = int(config["train"]["batch_size"] / 2)
			print(f"Decreasing batch size to {config['train']['batch_size']} and trying again")
			result = subprocess.run(
				command,
				encoding=sys.getdefaultencoding(),
				stderr=subprocess.PIPE, stdout=subprocess.PIPE
			)
		if result.returncode:
			print("Could not train", json.dumps({classifier_type: config}))
			print("Error:", result.stderr)
			return
	# Test
	command = [
		os.path.join(args.classifier, "classifier.py"),
		"--models", args.saveto,
		"test",
		"--write-stdout",
		intermediate_name
	]
	print(*command)
	result = subprocess.run(
		command,
		stdout=subprocess.PIPE
	)
	if result.returncode:
		print("Could not test", json.dumps({classifier_type: config}))
		return
	confusion_table, = pickle.loads(result.stdout)
	with open(os.path.join(args.saveto, str(x) + ".confusiontable"), "wb") as f:
		pickle.dump(confusion_table, f)
	os.rename(
		os.path.join(args.saveto, intermediate_name),
		os.path.join(args.saveto, str(x) + ".classifier")
	)

def search(args):
	if args.saveto is None:
		args.saveto = os.path.join(args.classifier, "search")
	if args.json is None:
		if getattr(args, "continue"):
			args.json = os.path.join(args.saveto, "search.json")
			if not os.path.exists(args.json):
				raise ValueError(f"Cannot continue search as {args.json} does not exist")
		else:
			args.json = os.path.join(args.classifier, "search.json")

	with open(args.json, "r") as f:
		search = json.load(f)

	permutations = ConfigPermutations(search)
	print(f"{len(permutations)} total combinations")
	if not os.path.exists(args.saveto):
		os.mkdir(args.saveto)
	
	with open(os.path.join(args.saveto, "search.json"), "w") as f:
		json.dump(search, f)
	
	print("Type 'stop' and press enter to stop the search after the next completed config")
	print()

	permutations.shuffle()
	i = 0
	total = sum(permutations.totallengths[t] for t in permutations.types if t.lower() in map(str.lower, args.types))
	if args.group:
		for current_classifier_type in ['LSTM', 'CNN', 'GMMHMM', 'GenHMM', 'SVM']: # permutations.types
			if len(args.types) > 0 and current_classifier_type.lower() not in map(str.lower, args.types):
				print("Skipping", current_classifier_type)
				continue

			print(f"Classifier: {current_classifier_type}, {permutations.totallengths[current_classifier_type]} total configs")
			for classifier_type, config, x in permutations:
				if classifier_type != current_classifier_type:
					continue
				i += 1
				if getattr(args, "continue") and os.path.exists(os.path.join(args.saveto, str(x) + ".classifier")):
					print(f"Classifier {x} already exists")
					continue

				print(f"Number {i} of {total}")
				train_test_config(classifier_type, config, x, genhmm_min_batch=args.genhmm_min_batch, intermediate_name=args.intermediate_name)
				
				try:
					if select.select([sys.stdin], [], [], 0)[0] and "stop" in sys.stdin.readline().strip().lower():
						print("Stopping search. Use the --continue flag to resume search.")
						return
				except:
					pass

	else:
		for classifier_type, config, x in permutations:
			if len(args.types) > 0 and classifier_type.lower() not in map(str.lower, args.types):
				continue

			i += 1
			if getattr(args, "continue") and os.path.exists(os.path.join(args.saveto, str(x) + ".classifier")):
				print(f"Classifier {x} already exists")
				continue

			print(f"Number {i} of {total}")
			train_test_config(classifier_type, config, x, genhmm_min_batch=args.genhmm_min_batch, intermediate_name=args.intermediate_name)

			try:
				if select.select([sys.stdin], [], [], 0)[0] and "stop" in sys.stdin.readline().strip().lower():
					print("Stopping search. Use the --continue flag to resume search.")
					return
			except:
				pass
			
	print("All done")

def results(args):
	if args.saveto is None:
		args.saveto = os.path.join(args.classifier, "search")
	
	with open(os.path.join(args.saveto, "search.json"), "r") as f:
		search = json.load(f)
	permutations = ConfigPermutations(search)

	files = next(os.walk(args.saveto))[2]
	files = [f for f, ext in map(os.path.splitext, files) if ext == ".classifier" and (f + ".confusiontable") in files]

	if len(args.types) > 0:
		files = [f for f in files if permutations[int(f)][0].lower() in map(str.lower, args.types)]

	print(f"{len(files)} of {len(permutations)} classifiers in folder")
	for classifier_type in permutations.types:
		if classifier_type.lower() not in map(str.lower, args.types):
			continue
		print(sum(permutations[int(f)][0] == classifier_type for f in files), "of type", classifier_type)

	bestPrecision = None
	bestRecall = None
	bestAvgAcc = None
	for fn in files:
		x = int(fn)
		with open(os.path.join(args.saveto, str(x) + ".confusiontable"), "rb") as f:
			confusiontable = pickle.load(f)
		if bestPrecision is None:
			bestPrecision = {label: (-1, -1) for label in confusiontable.true_labels}
			bestRecall = {label: (-1, -1) for label in confusiontable.true_labels}
			bestAvgAcc = (-1, -1)

		for label in confusiontable.true_labels:
			positives = sum(confusiontable[l, label] for l in confusiontable.true_labels)
			if positives == 0:
				precision = 0
			else:
				precision = confusiontable[label, label] / positives
			recall = confusiontable[label, label] / confusiontable[label, ...]
			if precision > bestPrecision[label][0]:
				bestPrecision[label] = (precision, x)
			if recall > bestRecall[label][0]:
				bestRecall[label] = (recall, x)
		avgAcc = sum(confusiontable[label, label] / confusiontable[label, ...] for label in confusiontable.true_labels) / len(confusiontable.true_labels)
		if avgAcc > bestAvgAcc[0]:
			bestAvgAcc = (avgAcc, x)

	
	print("-- Recall --")
	for label in bestRecall:
		print(f"Label: {label}, best: {bestRecall[label][0]:.2%} by model {bestRecall[label][1]}")
	print()
	print("-- Precision --")
	for label in bestPrecision:
		print(f"Label: {label}, best: {bestPrecision[label][0]:.2%} by model {bestPrecision[label][1]}")
	print()
	print(f"Best average accuracy: {bestAvgAcc[0]:.2%} by model {bestAvgAcc[1]}")
	print()

	all_ids = set(x for _, x in bestRecall.values()).union(x for _, x in bestPrecision.values()).union([bestAvgAcc[1]])
	for x in all_ids:
		print(x)
		print(permutations[x][0], permutations[x][1])
		with open(os.path.join(args.saveto, str(x) + ".confusiontable"), "rb") as f:
			confusiontable = pickle.load(f)
		print(confusiontable)

def coverage(args):
	if args.saveto is None:
		args.saveto = os.path.join(args.classifier, "search")
	
	with open(os.path.join(args.saveto, "search.json"), "r") as f:
		search = json.load(f)
	permutations = ConfigPermutations(search)

	files = next(os.walk(args.saveto))[2]
	files = [f for f, ext in map(os.path.splitext, files) if ext == ".classifier" and (f + ".confusiontable") in files]

	print(f"{len(permutations)} classifiers are described by {os.path.join(args.saveto, 'search.json')}")
	print(f"{len(files)} have been successfully tested")
	files_by_type = dict()
	for classifier_type in permutations.types:
		files_by_type[classifier_type] = [f for f in files if permutations[int(f)][0] == classifier_type]
		print(f"\t{len(files_by_type[classifier_type])} of type {classifier_type} ({len(files_by_type[classifier_type]) / permutations.totallengths[classifier_type]:.2%})")


	has_tested = dict()
	for classifier_type in permutations.types:
		has_tested[classifier_type] = [(cat_key, [False] * l)
				for cat_key, l in permutations.lengths[classifier_type].items()]
		for fn in files_by_type[classifier_type]:
			_, combination, _ = permutations._get_combination(int(fn))
			for combination_idx, idx in enumerate(combination):
				has_tested[classifier_type][combination_idx][1][idx] = True

	
	untested = sum(not tested for ct in has_tested for _, possible in has_tested[ct] for tested in possible)
	if untested != 0:
		print(f"{untested} untested or invalid parameter setting{'s' if untested != 1 else ''}:")
		for classifier_type in permutations.types:
			untested_current = [(cat_key, idx)
					for cat_key, possible in has_tested[classifier_type]
						for idx, tested in enumerate(possible)
							if not tested]
			if len(untested_current) != 0:
				print(f"\t{classifier_type} ({len(untested_current)} settings)")
				for (category, key), idx in untested_current:
					print(
						"\t\t", classifier_type, ".", category, ".", key, " = ",
						json.dumps(permutations.search[classifier_type][category][key][idx]),
						sep=""
					)
	else:
		print("All parameter settings have been successfully tested at least once")

def repair(args):
	sys.path.append(os.path.join(args.classifier, os.pardir, "gm_hmm", os.pardir))
	from classifier import Classifier
	globals()["Classifier"] = Classifier

	def fix_inf(d: dict) -> dict:
		for key in d.keys():
			if type(d[key]) is dict:
				d[key] = fix_inf(d[key])
			elif type(d[key]) is float and d[key] == np.inf:
				d[key] = "inf"
			elif type(d[key]) is float and d[key] == -np.inf:
				d[key] = "-inf"
		return d
	
	if args.saveto is None:
		args.saveto = os.path.join(args.classifier, "search")
	
	with open(os.path.join(args.saveto, "search.json"), "r") as f:
		search = json.load(f)
	permutations = ConfigPermutations(search)

	files = next(os.walk(args.saveto))[2]
	files = [f for f, ext in map(os.path.splitext, files) if ext == ".classifier" and (f + ".confusiontable") in files]
	files = files

	print(f"Found {len(files)} files")
	equivalents = []
	for fn in files:
		c = Classifier.from_file(fn + ".classifier", folder=args.saveto)
		current_type = c.model_type.__name__.lower()
		x = int(fn)
		best_guess = x
		for t, l in permutations.totallengths.items():
			if current_type == t.lower():
				break
			best_guess += l
		best_guess_ct, best_guess_config, best_guess_x = permutations[best_guess]

		if best_guess_ct.lower() == current_type and json.dumps(fix_inf(best_guess_config)) == json.dumps(fix_inf(c.config)):
			print(end=".", flush=True)
			equivalents.append(best_guess_x)
		else:
			for ct, config, identifier in permutations:
				if ct.lower() == current_type and json.dumps(fix_inf(config)) == json.dumps(fix_inf(c.config)):
					print(end=".", flush=True)
					equivalents.append(identifier)
					break
			else:
				print("Suggested:\n", json.dumps(permutations[best_guess][1]))
				print("True:\n", json.dumps(c.config))
				raise ValueError("Couldn't find equivalents for all")
	print()
	may_overwrite = False
	for original, new in zip(files, map(str, equivalents)):
		print(f"{original}\t->\t{new}")
		if new in files and new != original:
			may_overwrite = True
	if may_overwrite:
		print("This may cause an overwrite")
	if input("Proceed (y/n)? ").strip().lower()[-1] != "y":
		return
	
	for original, new in zip(files, map(str, equivalents)):
		if original != new:
			os.rename(
				os.path.join(args.saveto, original + ".classifier"),
				os.path.join(args.saveto, new + ".classifier")
			)
			os.rename(
				os.path.join(args.saveto, original + ".confusiontable"),
				os.path.join(args.saveto, new + ".confusiontable")
			)
			print(end=".", flush=True)
	print("Done")
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog="hyperparameter_search.py",
		description="Perform hyperparameter search on classifier.py"
	)
	parser.set_defaults(func=lambda a: parser.print_usage())
	subparsers = parser.add_subparsers()

	parser.add_argument("-c", "--classifier", help="Prestudy classifier folder (default: $PWD/classifier)", default=os.path.join(os.getcwd(), "classifier"))
	parser.add_argument("-s", "--saveto", help="Folder to save classifiers and stats to (default: <CLASSIFIER>/search)", default=None)

	subparser = subparsers.add_parser("search", help="Perform a search")
	subparser.set_defaults(func=search)

	subparser.add_argument("-g", "--group", help="Group tested configurations by type", action="store_true")
	subparser.add_argument("-t", "--types", help="Only test certain types of classifiers", nargs="+", default=[])
	subparser.add_argument("--genhmm-min-batch", help="Set minimum batch size of the GenHMM", metavar="SIZE", type=int, default=1)
	subparser.add_argument("--intermediate-name", help="Name of the intermediate classifier file (default: %(default)s", default="current.classifier")

	group = subparser.add_mutually_exclusive_group()
	group.add_argument("-j", "--json", help="JSON file with search options (default: <CLASSIFIER>/search.json)", default=None)
	group.add_argument("--continue", help="Continue the search in SAVETO.", action="store_true")

	subparser = subparsers.add_parser("results", help="Show the results of a search")
	subparser.set_defaults(func=results)

	subparser.add_argument("-t", "--types", help="Only allow certain types of classifiers", nargs="+", default=[])

	subparser = subparsers.add_parser("coverage", help="Show statistics of a search")
	subparser.set_defaults(func=coverage)

	subparser = subparsers.add_parser("repair", help="Repair a search where the relative configuration indexes have been used as filenames rather than the global ones")
	subparser.set_defaults(func=repair)

	args = parser.parse_args()
	start = time()
	args.func(args)
	print(f"Total time: {time() - start:.1f} s")