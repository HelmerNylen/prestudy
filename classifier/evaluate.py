#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import pickle
from confusion_table import ConfusionTable
from time import time, strftime, localtime
from math import floor

available_classifiers = ["GMMHMM", "GenHMM", "LSTM", "CNN", "SVM"]

def _scores_stats(scores: dict):
	"""Get the combinations of possibly valid keys for a scores dict"""

	import numpy as np
	classifiers = np.unique([classifier_type for classifier_type, _, _, _ in scores])
	datasets = np.unique([dataset_name for _, dataset_name, _, _ in scores])
	num_instances = {
		(ct, dn): max((instance + 1
			for classifier_type, dataset_name, instance, _ in scores
				if classifier_type == ct and dataset_name == dn), default=0)
		for ct in classifiers
			for dn in datasets
	}
	num_epochs = {
		(ct, dn): max((epoch
			for classifier_type, dataset_name, _, epoch in scores
				if classifier_type == ct and dataset_name == dn), default=0)
		for ct in classifiers
			for dn in datasets
	}
	return classifiers, datasets, num_instances, num_epochs

def coverage(args):
	if not os.path.exists(args.working_folder):
		print(f"Working folder {args.working_folder} does not exist")
		sys.exit(1)

	print("Loading confusion tables ...", end="", flush=True)
	scores = set()
	for filename in next(os.walk(args.working_folder))[2]:
		if filename.endswith(".confusiontable"):
			with open(os.path.join(args.working_folder, filename), "rb") as f:
				confusion_table = pickle.load(f)
			scores.add(_unpack_filename(filename))
	print(" Done")
	print()

	classifiers, datasets, num_instances, num_epochs = _scores_stats(scores)

	if len(classifiers) == len(datasets) == 0:
		print("Folder is empty")
		return
	
	# Find datasets for which the last epoch is incomplete
	incomplete = dict()
	for classifier_type in classifiers:
		for dataset in datasets:
			epoch = num_epochs[(classifier_type, dataset)]
			for instance in range(num_instances[(classifier_type, dataset)]):
				if (classifier_type, dataset, instance, epoch) not in scores:
					incomplete[(classifier_type, dataset)] = instance
					break
	
	# Print all stats
	print("[(Completed in last epoch)/]Instances x Epochs")
	maxlen = max(len(d) for d in datasets)
	print(" " * maxlen, *map(lambda s: s.rjust(10), classifiers), sep="  ")
	for d in datasets:
		print(d.ljust(maxlen), end="  ")
		for ct in classifiers:
			if (ct, d) in incomplete:
				print(f"{incomplete[(ct, d)]}/{num_instances[(ct, d)]} x {num_epochs[(ct, d)]}".rjust(10), end="  ")
			else:
				print(f"{num_instances[(ct, d)]} x {num_epochs[(ct, d)]}".rjust(10), end="  ")
		print()
	

def _create_plots(scores: dict, folder: str, partial: str="error", format: str="pdf", accuracy: bool=False):
	"""Create relevant plots for the pre-study"""

	import matplotlib.pyplot as plt
	import numpy as np
	assert partial in ("error", "allow", "ignore")

	scorestr = ("Accuracy", "accuracy") if accuracy else ("F1-score", "F1-score")
	classifiers, datasets, num_instances, num_epochs = _scores_stats(scores)

	consistent_colors = dict()
	
	# Average training progress on each dataset
	for dataset in datasets:
		plt.figure()
		plotted_anything = False
		minval = np.inf
		for classifier_type in classifiers:
			if num_epochs[(classifier_type, dataset)] == num_instances[(classifier_type, dataset)] == 0:
				continue
			avgs = []
			stats = {Q: [] for Q in (0.0, 0.25, 0.5, 0.75, 1.0)}
			for epoch in range(1, num_epochs[(classifier_type, dataset)] + 1):
				vals = []
				for instance in range(num_instances[(classifier_type, dataset)]):
					if (classifier_type, dataset, instance, epoch) not in scores:
						if partial == "allow":
							continue
						elif partial == "ignore":
							vals = []
							break
						else:
							raise KeyError(str((classifier_type, dataset, instance, epoch)) + " not available")
					vals.append(scores[(classifier_type, dataset, instance, epoch)])
				if len(vals) > 0:
					avgs.append(sum(vals) / len(vals))
					for Q, arr in stats.items():
						arr.append(np.quantile(vals, Q))
				else:
					break
			x = range(1, len(avgs) + 1)
			if len(x) > 1:
				line, = plt.plot(x, avgs, label=classifier_type, marker='.')
				consistent_colors[classifier_type] = line.get_color()
				plt.fill_between(x, stats[0.25], stats[0.75], color=line.get_color(), alpha=0.2, zorder=-1)
				minval = min(minval, *stats[0.25])
				plotted_anything = True

		if plotted_anything:
			plt.legend()
			plt.xlabel("Epoch")
			plt.xticks(range(1, int(plt.xlim()[1]) + 1))
			plt.ylim((floor(minval * 20) / 20, 1))
			plt.ylabel(scorestr[0])
			yticks, _ = plt.yticks()
			plt.yticks(yticks, [f"{y:.0%}" for y in yticks])
			plt.grid()
			plt.title(f"Training progress on {dataset}")
			plt.savefig(os.path.join(folder, "training_" + dataset + "." + format))
		else:
			print("No data to plot for dataset", dataset)
	
	epoch_used = dict()

	# Score distribution
	for dataset in datasets:
		plt.figure()
		plotted_anything = False
		boxplot_data = []
		for classifier_type in classifiers:
			if num_epochs[(classifier_type, dataset)] == num_instances[(classifier_type, dataset)] == 0:
				continue
			
			#epoch = num_epochs[(classifier_type, dataset)]
			#instances_in_last = sum(1
			#	for instance in range(num_instances[(classifier_type, dataset)])
			#		if (classifier_type, dataset, instance, epoch) in scores)
			#if instances_in_last != num_instances[(classifier_type, dataset)] and partial == "ignore":
			#	epoch -= 1

			epoch = max(1, num_epochs[(classifier_type, dataset)] - 1)
			epoch_used[(classifier_type, dataset)] = epoch
			boxplot_data.append((
				[scores[(classifier_type, dataset, instance, epoch)]
					for instance in range(num_instances[(classifier_type, dataset)])
						if (classifier_type, dataset, instance, epoch) in scores],
				classifier_type
			))
		plt.boxplot(tuple(zip(*boxplot_data))[0], labels=tuple(zip(*boxplot_data))[1], whis=(0, 100))
		plt.ylim((floor(min(scores.values()) * 20) / 20, 1))
		plt.ylabel(scorestr[0])
		yticks, _ = plt.yticks()
		plt.yticks(yticks, [f"{y:.0%}" for y in yticks])
		plt.grid(axis="y")
		plt.title(f"Classifier performance distribution on {dataset}")
		plt.savefig(os.path.join(folder, "dist_" + dataset + "." + format))

	# Dataset difficulty
	plt.figure()
	width = 1 / (len(datasets) + 2)
	for i, classifier_type in enumerate(classifiers):
		# This probably does not consider --allow-partial and --ignore-partial they way it should
		avgs = []
		for dataset in datasets:
			vals = []
			for instance in range(num_instances[(classifier_type, dataset)]):
				if (classifier_type, dataset, instance, epoch_used[(classifier_type, dataset)]) in scores:
					vals.append(scores[(classifier_type, dataset, instance, epoch_used[(classifier_type, dataset)])])
			avgs.append(np.nan if len(vals) == 0 else sum(vals) / len(vals))

		if classifier_type not in consistent_colors:
			consistent_colors[classifier_type] = [c
				for c in plt.rcParams['axes.prop_cycle'].by_key()['color']
					if c not in consistent_colors.values()][0]
		
		plt.bar(
			np.arange(len(datasets)) + width * (i + 0.5 - len(classifiers) / 2),
			avgs,
			width=width,
			label=classifier_type,
			color=consistent_colors[classifier_type]
		)
		plt.legend()
		plt.xlabel("Dataset")
		plt.xticks(range(len(datasets)), datasets)
		plt.ylim((floor(min(scores.values()) * 20) / 20, 1))
		plt.ylabel(scorestr[0])
		yticks, _ = plt.yticks()
		plt.yticks(yticks, [f"{y:.0%}" for y in yticks])
		plt.grid(axis="y")
		plt.title(f"Classifier performance by dataset")
		plt.gcf().autofmt_xdate()
		plt.savefig(os.path.join(folder, "datasets." + format), bbox_inches='tight')
	
	print([(t, v) for t, v in scores.items() if v == max(scores.values())])
	print([(t, v) for t, v in scores.items() if v == min(scores.values())])
	
	# Suppress warnings
	import warnings
	warnings.filterwarnings("ignore")

def plot(args):
	if not os.path.exists(args.working_folder):
		print(f"Working folder {args.working_folder} does not exist")
		sys.exit(1)

	plots_folder = os.path.join(args.working_folder, "plots")
	if not os.path.exists(plots_folder):
		print(f"Creating plots folder {plots_folder}")
		os.mkdir(plots_folder)
	
	print("Loading confusion tables ...", end="", flush=True)
	scores = dict()
	for filename in next(os.walk(args.working_folder))[2]:
		if filename.endswith(".confusiontable"):
			classifier_type, dataset, instance, epoch = _unpack_filename(filename)
			if classifier_type.lower() not in map(str.lower, args.types):
				continue
			if args.datasets is not None and dataset.lower() not in map(str.lower, args.datasets):
				continue
			with open(os.path.join(args.working_folder, filename), "rb") as f:
				confusion_table = pickle.load(f)
			if args.accuracy:
				scores[(classifier_type, dataset, instance, epoch)] = confusion_table.accuracy()
			else:
				scores[(classifier_type, dataset, instance, epoch)] = confusion_table.average("F1-score")
	print(" Done")

	if args.datasets is not None:
		for dataset in args.datasets:
			if dataset.lower() not in (d.lower() for _, d, _, _ in scores):
				print(f"Dataset {dataset} specified but not found among confusion tables")
	for classifier_type in args.types:
		if classifier_type.lower() not in (c.lower() for c, _, _, _ in scores):
			print(f"Classifier type {classifier_type} specified but not found among confusion tables")
	
	print("Creating plots")
	partial = "allow" if args.allow_partial else ("ignore" if args.ignore_partial else "error")
	_create_plots(scores, plots_folder, partial, accuracy=args.accuracy)
	print("Done")
	
def _filename(classifier_type, dataset, instance, epoch, ext=".classifier"):
	return classifier_type + "_" + dataset + "_" + str(instance) + "_" + str(epoch) + ext

def _unpack_filename(filename):
	filename = os.path.splitext(filename)[0]
	split = filename.split("_")
	if len(split) < 4:
		raise ValueError("Invalid filename: " + filename)
	classifier_type = split[0]
	dataset = "_".join(split[1:-2])
	instance = int(split[-2])
	epoch = int(split[-1])
	return classifier_type, dataset, instance, epoch

def _time() -> str:
	return strftime("%H:%M:%S", localtime(time()))

def repeat(args):
	if args.tolerance < 0:
		print("Tolerance must be >= 0")
		sys.exit(1)
	if args.num_instances < 1:
		print("Tolerance must be >= 1")
		sys.exit(1)

	# Defaults for data folders
	if args.train is None:
		args.train = os.path.join(args.data, "train")
	if not os.path.exists(args.train):
		print(f"Training folder {args.train} does not exist")
		sys.exit(1)
	if args.test is None:
		args.test = os.path.join(args.data, "test")
	if not os.path.exists(args.test):
		print(f"Testing folder {args.test} does not exist")
		sys.exit(1)
	
	# Default for config path
	if args.config is None:
		args.config = os.path.join(args.classifier, "defaults.json")
	if not os.path.exists(args.config):
		print(f"Config file {args.config} does not exist")
		sys.exit(1)

	if not os.path.exists(args.working_folder):
		print(f"Creating working folder {args.working_folder}")
		os.mkdir(args.working_folder)
	
	# Locate all datasets
	datasets = []
	for folder in tuple(next(os.walk(args.train))[1]) + (None,):
		if folder is None:
			trainpath = args.train
			testpath = args.test
		else:
			trainpath = os.path.join(args.train, folder)
			testpath = os.path.join(args.test, folder)
		if not os.path.exists(testpath):
			continue
		if folder == "root":
			print("A dataset is named 'root'. Ignoring it.")
			continue
		if len(next(os.walk(trainpath))[2]) > 0 and len(next(os.walk(testpath))[2]) > 0:
			datasets.append((folder or "root", trainpath, testpath))
	
	# Filter datasets
	if args.datasets is not None:
		for name in args.datasets:
			if name.lower() not in (n.lower() for n, _, _ in datasets):
				print(f"Unknown dataset {name}. All datasets: {', '.join(n for n, _, _ in datasets)}.")
		datasets = [tup for tup in datasets if tup[0].lower() in map(str.lower, args.datasets)]
	else:
		print("Found datasets:", ', '.join(n for n, _, _ in datasets))
	
	scores = dict()

	for classifier_type in args.types:
		if classifier_type.lower() not in map(str.lower, available_classifiers):
			print(f"Unrecognized classifier type: {classifier_type}. Skipping.")
			continue
		# Fix capitalization so equivalent files are detected
		classifier_type = available_classifiers[[c.lower() for c in available_classifiers].index(classifier_type.lower())]

		for dataset_name, trainpath, testpath in datasets:
			# Similar to hyperparameter_search.py epochs
			# Train all instances of a classifier on a dataset until average test accuracy or F1-score stagnates
			epoch = 0
			while True:
				epoch += 1
				print("Classifier:", classifier_type, "Dataset:", dataset_name, "Epoch:", epoch, sep="\t")
				for instance in range(args.num_instances):
					current_filename =                _filename(classifier_type, dataset_name, instance, epoch)
					current_filename_confusiontable = _filename(classifier_type, dataset_name, instance, epoch, ".confusiontable")

					if args.skip_existing and os.path.exists(os.path.join(args.working_folder, current_filename))\
							and os.path.exists(os.path.join(args.working_folder, current_filename_confusiontable)):

						print("\tInstance", instance, "already exists. Skipping.")
						with open(os.path.join(args.working_folder, current_filename_confusiontable), "rb") as f:
							confusion_table = pickle.load(f)
						scores[(classifier_type, dataset_name, instance, epoch)] = confusion_table.average("F1-score")
						continue

					else:
						print("\tInstance:", instance)
						
					# Train
					command = [
						os.path.join(args.classifier, "classifier.py"),
						"--models", args.working_folder,
						"--train", trainpath,
						"train",
						"-s",
						"-w", current_filename
					]
					if epoch == 1:
						command.extend([
							"-c", classifier_type.lower(),
							"--config", args.config
						])
					else:
						command.extend([
							"--read", _filename(classifier_type, dataset_name, instance, epoch - 1)
						])
					
					print(_time() + ">", *command)
					result = subprocess.run(
						command,
						encoding=sys.getdefaultencoding(),
						stderr=subprocess.PIPE, stdout=subprocess.PIPE
					)
					if result.returncode:
						print("Could not train", current_filename)
						print("Error:", result.stderr)
						sys.exit(1)
					elif args.always_print_stderr:
						print(result.stderr)

					# Test
					command = [
						os.path.join(args.classifier, "classifier.py"),
						"--test", testpath,
						"--models", args.working_folder,
						"test",
						"--write-stdout",
						current_filename
					]
					print(_time() + ">", *command)
					result = subprocess.run(
						command,
						stdout=subprocess.PIPE
					)
					if result.returncode:
						print("Could not test", current_filename)
						sys.exit(1)
					elif args.always_print_stderr:
						print(result.stderr)

					confusion_table, = pickle.loads(result.stdout)
					if args.accuracy:
						s = confusion_table.accuracy()
						print(f"\tAccuracy: {s:.2%}")
					else:
						s = confusion_table.average('F1-score')
						print(f"\tF1-score: {s:.2%}")
					scores[(classifier_type, dataset_name, instance, epoch)] = s
					with open(os.path.join(args.working_folder, current_filename_confusiontable), "wb") as f:
						pickle.dump(confusion_table, f)
					
				current = sum(scores[(classifier_type, dataset_name, instance, epoch)]
						for instance in range(args.num_instances)) / args.num_instances
				print(f"Average: {current:.2%}", end="")
				if epoch >= 2:
					prev = sum(scores[(classifier_type, dataset_name, instance, epoch - 1)]
								for instance in range(args.num_instances)) / args.num_instances
					diff = current - prev
					print(f", increase: {diff:.2%}")
					if diff < args.tolerance:
						print("Minimum increase reached")
						break
				else:
					print()

				if classifier_type.lower() == "svm":
					print("SVM cannot train for multiple epochs")
					break
				
	if args.plot:
		plots_folder = os.path.join(args.working_folder, "plots")
		if not os.path.exists(plots_folder):
			print(f"Creating plots folder {plots_folder}")
			os.mkdir(plots_folder)

		print("Creating plots")
		_create_plots(scores, plots_folder)

	print("Done")

def reconfigure(args):
	sys.path.append(os.path.join(args.classifier, os.pardir, "gm_hmm", os.pardir))
	# Classifier must be defined on __main__ to allow pickle to read classifier files
	global Classifier
	# This imports Classifier as classifier.Classifier, which is stored during pickling
	from classifier import Classifier
	# To avoid errors when unpickling (in classifier.py when running e.g. test), we overwrite the module
	# TODO: It could also probably be fixed by moving the class definition of Classifier to its own file
	Classifier.__module__ = "__main__"

	if not os.path.exists(args.working_folder):
		print(f"Working folder {args.working_folder} does not exist")
		sys.exit(1)
	
	print("Loading classifiers ...", end="", flush=True)
	classifiers = dict()
	for filename in next(os.walk(args.working_folder))[2]:
		if filename.endswith(".classifier") and not filename.startswith("intermediate_"):
			classifier_type, dataset, instance, epoch = _unpack_filename(filename)
			if classifier_type.lower() not in map(str.lower, args.types):
				continue
			if args.datasets is not None and dataset.lower() not in map(str.lower, args.datasets):
				continue
			classifier = Classifier.from_file(filename, folder=args.working_folder)
			classifiers[(classifier_type, dataset, instance, epoch)] = classifier
	print(" Done")

	if args.datasets is not None:
		for dataset in args.datasets:
			if dataset.lower() not in (d.lower() for _, d, _, _ in classifiers):
				print(f"Dataset {dataset} specified but not found among classifiers")
	for classifier_type in args.types:
		if classifier_type.lower() not in (c.lower() for c, _, _, _ in classifiers):
			print(f"Classifier type {classifier_type} specified but not found among classifiers")

	print("--- EXISTING ---")
	print("Name", "train.force_cpu", "score.use_gpu")
	for id_tup, classifier in classifiers.items():
		print(_filename(*id_tup), end="\t")
		if "force_cpu" in classifier.config["train"]:
			print(classifier.config["train"]["force_cpu"], end="\t")
		else:
			print("undefined", end="\t")
		
		if "score" in classifier.config and "use_gpu" in classifier.config["score"]:
			print(classifier.config["score"]["use_gpu"], )
		else:
			print("undefined")

		classifier.config["train"]["force_cpu"] = args.target == "cpu"
		if "score" not in classifier.config:
			classifier.config["score"] = dict()
		classifier.config["score"]["use_gpu"] = args.target == "gpu"
		
	print("\n--- NEW ---")
	print("Name", "train.force_cpu", "score.use_gpu")
	for id_tup, classifier in classifiers.items():
		print(_filename(*id_tup), end="\t")
		print(classifier.config["train"]["force_cpu"], end="\t")
		print(classifier.config["score"]["use_gpu"])

	if input("Overwriting original files. Proceed (y/n)? ").strip().lower()[-1] != "y":
		return

	for id_tup, classifier in classifiers.items():
		print("Saving " + _filename(*id_tup))
		classifier.save_to_file(_filename(*id_tup), folder=args.working_folder)
	print("Done")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog="evaluate.py",
		description="Evaluate different classifiers' performace on different datasets"
	)
	parser.set_defaults(func=lambda a: parser.print_usage())
	subparsers = parser.add_subparsers()

	group = parser.add_argument_group("Input folders")
	group.add_argument("-c", "--classifier", help="Prestudy classifier folder (default: $PWD/classifier)", default=os.path.join(os.getcwd(), "classifier"))
	group.add_argument("-d", "--data", help="Prestudy data folder (default: $PWD/data)", default=os.path.join(os.getcwd(), "data"))
	group.add_argument("--train", help="Path to training data output folder (default: <DATA>/train)", default=None)
	group.add_argument("--test", help="Path to testing data output folder (default: <DATA>/test)", default=None)

	subparser = subparsers.add_parser("repeat", help="Train multiple instances of each classifier")
	subparser.set_defaults(func=repeat)

	subparser.add_argument("working_folder", help="Folder in which to store classifiers while evaluating them")
	subparser.add_argument("num_instances", help="Number of classifiers of each type per dataset to average", metavar="N", type=int)

	subparser.add_argument("-t", "--types", help="Only test certain types of classifiers (default: %(default)s)", nargs="+", default=available_classifiers[:-1])
	subparser.add_argument("--datasets", help="Names of data sets in <TRAIN> and <TEST> to use. Use 'root' to refer to root folder. Default: use all datasets.", default=None, nargs="+")
	subparser.add_argument("-a", "--accuracy", help="Use accuracy instead of F1-score", action="store_true")
	subparser.add_argument("--config", help="Path to classifier config file (default: <CLASSIFIER>/defaults.json)", default=None)
	subparser.add_argument("--tolerance", help="Threshold for difference in score. Stops training when this is reached. Default: %(default).2e", metavar="TOL", type=float, default=1e-3)
	subparser.add_argument("-p", "--plot", help="Create plots when the run is done", action="store_true")
	subparser.add_argument("-s", "--skip-existing", help="Skip existing classifiers in the working folder. Can be used to continue a run.", action="store_true")
	subparser.add_argument("--always-print-stderr", help="Print output from stderr even if the return code is 0.", action="store_true")

	subparser = subparsers.add_parser("plot", help="Create plots from a 'repeat' run")
	subparser.set_defaults(func=plot)

	subparser.add_argument("working_folder", help="Folder in which classifiers and confusion tables are located, and in which to store plots")

	subparser.add_argument("-t", "--types", help="Only plot certain types of classifiers (default: plots all available types)", nargs="+", default=available_classifiers)
	subparser.add_argument("--datasets", help="Only plot certain datasets. Use 'root' to refer to root folder. Default: plots all available datasets.", default=None, nargs="+")
	subparser.add_argument("-a", "--accuracy", help="Use accuracy instead of F1-score", action="store_true")
	
	group = subparser.add_mutually_exclusive_group()
	group.add_argument("--allow-partial", help="Include classifiers which have not finished training their last epoch", action="store_true")
	group.add_argument("--ignore-partial", help="Same as --allow-partial but takes the last finished epoch instead of a possibly unfinished epoch", action="store_true")

	subparser = subparsers.add_parser("coverage", help="Show stats about a 'repeat' run")
	subparser.set_defaults(func=coverage)

	subparser.add_argument("working_folder", help="Folder in which classifiers and confusion tables are located")

	subparser = subparsers.add_parser("reconfigure", help="Reconfigure existing classifiers to train/test on CPU or GPU")
	subparser.set_defaults(func=reconfigure)

	subparser.add_argument("working_folder", help="Folder in which classifiers and confusion tables are located")
	subparser.add_argument("target", help="Train/test on GPU or CPU", choices=("cpu", "gpu"), type=str.lower)

	subparser.add_argument("-t", "--types", help="Only reconfigure certain types of classifiers (default: reconfigures all available types)", nargs="+", default=available_classifiers)
	subparser.add_argument("--datasets", help="Only reconfigure for certain datasets. Use 'root' to refer to root folder. Default: reconfigures for all available datasets.", default=None, nargs="+")

	args = parser.parse_args()
	start = time()
	args.func(args)
	print(f"Total time: {time() - start:.1f} s")