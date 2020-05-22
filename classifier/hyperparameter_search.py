#!/usr/bin/env python3
import argparse
import json
import os
import sys
import select
import subprocess
import pickle
import re
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
		while result.returncode and classifier_type.lower() == "genhmm" and int(config["train"]["batch_size"]/2) >= genhmm_min_batch:
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

	# Prioritized configs
	if len(args.prioritize) > 0:
		print("Testing prioritized configs")
		for i, x in enumerate(args.prioritize):
			if getattr(args, "continue") and os.path.exists(os.path.join(args.saveto, str(x) + ".classifier")):
				print(f"Overwriting classifier {x}")

			print(f"Number {i} of {len(args.prioritize)}")
			train_test_config(*permutations[x], genhmm_min_batch=args.genhmm_min_batch, intermediate_name=args.intermediate_name)
			
			try:
				if select.select([sys.stdin], [], [], 0)[0] and "stop" in sys.stdin.readline().strip().lower():
					print("Stopping search. Use the --continue flag to resume search.")
					return
			except:
				pass


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

# TODO: parsea args.types ordentligt istället för att mappa str.lower flera gånger och hålla på
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
		if len(args.types) and classifier_type.lower() not in map(str.lower, args.types):
			continue
		print(sum(permutations[int(f)][0] == classifier_type for f in files), "of type", classifier_type)

	metrics = {"Precision": ConfusionTable.precision, "Recall": ConfusionTable.recall, "F1-score": ConfusionTable.F1_score}
	bestPerLabel = None
	bestAvg = None
	for fn in files:
		x = int(fn)
		with open(os.path.join(args.saveto, str(x) + ".confusiontable"), "rb") as f:
			confusiontable = pickle.load(f)
		if bestPerLabel is None:
			bestPerLabel = {metric: {label: (-1, -1) for label in confusiontable.true_labels} for metric in metrics}
			bestAvg = {metric: (-1, -1) for metric in metrics}

		for metric in metrics:
			measures = []
			for label in confusiontable.true_labels:
				measures.append(metrics[metric](confusiontable, label))
				if measures[-1] > bestPerLabel[metric][label][0]:
					bestPerLabel[metric][label] = (measures[-1], x)
			N = sum(confusiontable[label, ...] for label in confusiontable.true_labels)
			avg = sum(measure * confusiontable[label, ...] / N for measure, label in zip(measures, confusiontable.true_labels))
			if avg > bestAvg[metric][0]:
				bestAvg[metric] = (avg, x)
	
	all_ids = set(x for metric in metrics for _, x in bestPerLabel[metric].values())\
			.union(x for _, x in bestAvg.values())
	for x in all_ids:
		print(x)
		print(permutations[x][0], permutations[x][1])
		with open(os.path.join(args.saveto, str(x) + ".confusiontable"), "rb") as f:
			confusiontable = pickle.load(f)
		print(confusiontable)

	for metric in metrics:
		print("--", metric, "--")
		for label in bestPerLabel[metric]:
			print(f"Label: {label}, best: {bestPerLabel[metric][label][0]:.2%} by model {bestPerLabel[metric][label][1]}")
		print(f"Best average: {bestAvg[metric][0]:.2%} by model {bestAvg[metric][1]}")
		print()
	
	# Kan göras mycket effektivare
	if args.optimal:
		print("Estimated optimal settings: ")
		for classifier_type in permutations.types:
			if len(args.types) and classifier_type.lower() not in map(str.lower, args.types):
				continue
			print(classifier_type)
			optimal_combination = []
			for combination_idx, ((category, key), l) in enumerate(permutations.lengths[classifier_type].items()):
				avgs = [0] * l
				for value in range(l):
					total = 0
					N = 0
					for x in range(len(permutations)):
						c_t, combination, _ = permutations._get_combination(x)
						if c_t != classifier_type or combination[combination_idx] != value:
							continue
						if not os.path.exists(os.path.join(args.saveto, str(x) + ".confusiontable")):
							continue
						with open(os.path.join(args.saveto, str(x) + ".confusiontable"), "rb") as f:
							confusiontable = pickle.load(f)
						total += confusiontable.average("F1-score")
						N += 1
					if N > 0:
						avgs[value] = total / N
				print("\t" + category, key, sep=".", end=" = ")
				print(json.dumps(permutations.search[classifier_type][category][key][np.argmax(avgs)]))
				optimal_combination.append(np.argmax(avgs))
			print()
			for x in range(len(permutations)):
				c_t, combination, _ = permutations._get_combination(x)
				if c_t == classifier_type and combination == optimal_combination:
					print("\tIndex of optimal combination:", x)
					has_tested = os.path.exists(os.path.join(args.saveto, str(x) + ".confusiontable"))
					print("\tHas been tested:", "yes" if has_tested else "no")
					if has_tested:
						if x not in all_ids:
							with open(os.path.join(args.saveto, str(x) + ".confusiontable"), "rb") as f:
								print(pickle.load(f))
						else:
							print("See above for confusion table")
					break
			print()

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
		print(
			f"\t{len(files_by_type[classifier_type])}/{permutations.totallengths[classifier_type]} of type {classifier_type}",
			f"({len(files_by_type[classifier_type]) / permutations.totallengths[classifier_type]:.2%})"
		)


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

def epochs(args):
	if args.tolerance < 0:
		print("Tolerance must be >= 0")
		sys.exit(1)

	if args.saveto is None:
		args.saveto = os.path.join(args.classifier, "search")
	if args.start_from_config is None and args.start_from_files is None:
		args.start_from_config = os.path.join(args.classifier, "defaults.json")
	
	if args.start_from_config is not None:
		with open(args.start_from_config, "r") as f:
			config = json.load(f)
			configs = [{classifier_type: config[classifier_type]}
				for classifier_type in config
					if len(args.types) == 0 or classifier_type.lower() in map(str.lower, args.types)]
					
		if len(args.files) == 1 and args.files[0] == "*":
			args.files = ["latest_<TYPE>.classifier"]
		if len(args.files) == 1 and "<type>" in args.files[0].lower():
			# TODO: Move the latest_<TYPE>.classifier replacements and similar to a utility file instead of copypasting
			args.files = [re.sub("<type>", classifier_type, args.files[0], flags=re.IGNORECASE)
				for config in configs for classifier_type in config]

	else:
		# This is ugly and should be done in a better way
		sys.path.append(os.path.join(args.classifier, os.pardir, "gm_hmm", os.pardir))
		from classifier import Classifier
		globals()["Classifier"] = Classifier

		configs = []
		keep_file = [False] * len(args.start_from_files)
		for i, fn in enumerate(args.start_from_files):
			c = Classifier.from_file(fn, folder=args.saveto)
			if len(args.types) == 0 or c.model_type.__name__.lower() in map(str.lower, args.types):
				configs.append({c.model_type.__name__: c.config})
				keep_file[i] = True

		if len(args.files) == len(args.start_from_files):
			args.files = [fn for fn, keep in zip(args.files, keep_file) if keep]
		args.start_from_files = [fn for fn, keep in zip(args.start_from_files, keep_file) if keep]
		
		if len(args.files) == 1 and args.files[0] == "*":
			args.files = args.start_from_files

	if len(args.files) > 0 and len(args.files) != len(configs):
		raise ValueError(f"Invalid number of output files")

	with NamedTemporaryFile() as intermediateFile:
		confusion_tables = [[] for _ in range(len(configs))]
		for i, config in enumerate(configs):
			for epoch in range(1, args.max_epochs + 1):
				with NamedTemporaryFile("w") as configFile:
					with open(configFile.name, "w") as f:
						json.dump(config, f)

					# Train
					command = [
						os.path.join(args.classifier, "classifier.py"),
						"--models", args.saveto,
						"train",
						"-s",
						"-w", intermediateFile.name if len(args.files) == 0 else args.files[i]
					]
					if epoch == 1:
						command.extend([
							"-c", next(iter(config)).lower(),
							"--config", configFile.name
						])
					else:
						command.append("--read")
					
					print(*command)
					result = subprocess.run(
						command,
						encoding=sys.getdefaultencoding(),
						stderr=subprocess.PIPE, stdout=subprocess.PIPE
					)
					if result.returncode:
						print("Could not train", json.dumps(config))
						print("Error:", result.stderr)
						sys.exit(1)

				# Test
				command = [
					os.path.join(args.classifier, "classifier.py"),
					"--models", args.saveto,
					"test",
					"--write-stdout",
					intermediateFile.name if len(args.files) == 0 else args.files[i]
				]
				print(*command)
				result = subprocess.run(
					command,
					stdout=subprocess.PIPE
				)
				if result.returncode:
					print("Could not test", json.dumps(config))
					sys.exit(1)
				confusion_table, = pickle.loads(result.stdout)
				confusion_tables[i].append(confusion_table)
				print(f"Epoch {epoch}. F1-score: {confusion_table.average('F1-score'):.2%}", end="")
				
				if len(confusion_tables[i]) >= 2:
					diff = confusion_tables[i][-1].average("F1-score") - confusion_tables[i][-2].average("F1-score")
					print(f", increase: {diff:.2%}")
					if diff < args.tolerance:
						print("Minimum increase reached")
						break
				else:
					print()

				if next(iter(config)).lower() == "svm":
					print("SVM cannot train for multiple epochs")
					break
			else:
				print("Maximum number of epochs reached")

	print()
	print("-- Epochs needed --")
	for i, config in enumerate(configs):
		print(len(confusion_tables[i]), "for classifier ", end="")
		print(json.dumps(config))

def repair(args):
	sys.path.append(os.path.join(args.classifier, os.pardir, "gm_hmm", os.pardir))
	# Classifier must be defined on __main__ to allow pickle to read classifier files
	global Classifier
	from classifier import Classifier
	Classifier.__module__ = "__main__"

	# The strings "inf" and "-inf" are interpreted as +-np.inf, and saved as such by pickle
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

	# Consider all classifiers with corresponding confusion tables
	files = next(os.walk(args.saveto))[2]
	files = [f for f, ext in map(os.path.splitext, files) if ext == ".classifier" and (f + ".confusiontable") in files]

	print(f"Found {len(files)} files")
	equivalents = []
	for fn in files:
		c = Classifier.from_file(fn + ".classifier", folder=args.saveto)
		current_type = c.model_type.__name__.lower()
		x = int(fn)
		# We can calculate the most likely config id if we know its type and its type-local id
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
			# If our guess is wrong, brute-force search for the config id
			for ct, config, identifier in permutations:
				if ct.lower() == current_type and json.dumps(fix_inf(config)) == json.dumps(fix_inf(c.config)):
					print(end=".", flush=True)
					equivalents.append(identifier)
					break
			else:
				# If we cannot find an equivalent config, notify the user
				print("Suggested:\n", json.dumps(permutations[best_guess][1]))
				print("True:\n", json.dumps(c.config))
				raise ValueError("Couldn't find equivalents for all")
	print()
	
	# List all suggested renamings
	may_overwrite = False
	for original, new in zip(files, map(str, equivalents)):
		print(f"{original}\t->\t{new}")
		if new in files and new != original:
			may_overwrite = True
	if may_overwrite:
		print("This may cause an overwrite")
	if input("Proceed (y/n)? ").strip().lower()[-1] != "y":
		return
	
	# Rename classifiers and confusion tables
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
	subparser.add_argument("--prioritize", help="Push certain configuration IDs to the front of the search queue", metavar="ID", nargs="+", default=[], type=int)

	group = subparser.add_mutually_exclusive_group()
	group.add_argument("-j", "--json", help="JSON file with search options (default: <CLASSIFIER>/search.json)", default=None)
	group.add_argument("--continue", help="Continue the search in SAVETO.", action="store_true")

	subparser = subparsers.add_parser("results", help="Show the results of a search")
	subparser.set_defaults(func=results)

	subparser.add_argument("-t", "--types", help="Only allow certain types of classifiers", nargs="+", default=[])
	subparser.add_argument("-o", "--optimal", help="Estimate optimal parameters", action="store_true")

	subparser = subparsers.add_parser("coverage", help="Show statistics of a search")
	subparser.set_defaults(func=coverage)

	subparser = subparsers.add_parser("epochs", help="Train a set of models until test score converges")
	subparser.set_defaults(func=epochs)

	group = subparser.add_mutually_exclusive_group()
	group.add_argument("--start-from-config", help="Start training from config file. Specify path to classifier config file (default: <CLASSIFIER>/defaults.json)", metavar="CONFIG", default=None)
	group.add_argument("--start-from-files", help="Start training based on preexisting classifiers. Copies configs from the specified classifiers (note that it does not continue training existing classifiers). Paths are relative to <SAVETO>.", metavar="FILE", nargs="+", default=None)

	subparser.add_argument("-t", "--types", help="Only test certain types of classifiers", nargs="+", default=[])
	subparser.add_argument("files", help="Paths to save intermediate and resulting classifiers to, relative to <SAVETO>. If not specified, do not save classifiers.\n"
		+ "If --start-from-config is set (default), provide 0, 1 (containing <TYPE>) or as many as specified in config filtered by --types.\n"
		+ "If --start-from-files is set, provide 0 or as many as the number of files specified with --start-from-files filtered by --types.\n"
		+ "The wildcard argument '*' will use 'latest_<TYPE>.classifier' if --start-from-config is set, and overwrite the provided files if --start-from-files is set.", metavar="FILE", nargs="*")
	subparser.add_argument("--tolerance", help="Threshold for difference in F1-score. Stops training when this is reached. Default: %(default).2e", metavar="TOL", type=float, default=1e-3)
	subparser.add_argument("--max-epochs", help="Maximum number of epochs to train a classifier (default: %(default)d). Always 1 for SVMs.", type=int, default=1000)

	subparser = subparsers.add_parser("repair", help="Repair a search where the relative configuration indexes have been used as filenames rather than the global ones")
	subparser.set_defaults(func=repair)

	args = parser.parse_args()
	start = time()
	args.func(args)
	print(f"Total time: {time() - start:.1f} s")