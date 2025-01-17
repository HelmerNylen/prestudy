#!/usr/bin/env python3
import argparse
import os
import numpy as np
import sys
import pickle
import json
import re
from time import time
from contextlib import redirect_stdout
from model import Model
from confusion_table import ConfusionTable

class Classifier:
	"""Serves as an interface for classification using a certain type of model and correspondig configuration"""

	def __init__(self, noise_types: list, model_type: Model, config: dict, silent: bool):
		self.model_type = model_type
		self.config = config[model_type.__name__]

		if silent:
			# Overwrite any verbosity flags
			for category in self.config:
				if "verbose" in self.config[category]:
					self.config[category]["verbose"] = False

		if self.model_type.MULTICLASS:
			# Create a single model for classification, with knowledge of all noise types
			self.model = model_type(noise_types=noise_types, config=self.config)
		else:
			# Create multiple models for classification, each having knowledge of its own noise type
			self.models = {noise_type: model_type(config=self.config)
				for noise_type in noise_types
			}
	
	def train(self, labeled_features, models_folder=None, silent=False):
		"""Train the classifier's model for one epoch.
		
		If models_folder is specified and the model type is not multi-class,
		an intermediate version is saved during training to prevent loss of progress in case of an error
		
		If silent is set, stdout is redirected to stderr and some informational output is removed"""

		if self.model_type.MULTICLASS:
			with redirect_stdout(sys.stderr if silent else sys.stdout):
				self.model.train(labeled_features, config=self.config)
		else:
			for noise_type, model in self.models.items():
				if not silent:
					print(f"Training {noise_type} {model.__class__.__name__} model")

				with redirect_stdout(sys.stderr if silent else sys.stdout):
					model.train(labeled_features[noise_type], config=self.config)
				
				if models_folder is not None:
					if not silent:
						print("Saving intermediate classifier")
					self.save_to_file(
						folder=models_folder,
						filename=f"intermediate_{self.model_type.__name__}.classifier",
						verbose=not silent
					)

			if not silent:
				print("Removing intermediate file")
			os.remove(os.path.join(models_folder, f"intermediate_{self.model_type.__name__}.classifier"))
	
	def label(self, features, return_scores=False, silent=False):
		"""Label a set of feature sequences. Use test() to test performance on a dataset."""

		if "score" in self.config:
			kwargs = self.config["score"]
		else:
			kwargs = dict()

		with redirect_stdout(sys.stderr if silent else sys.stdout):
			if self.model_type.MULTICLASS:
				scores = self.model.score(features, **kwargs)
				predicted_class = np.argmax(scores, axis=1)
				noise_types = np.array(self.model.get_noise_types())

			else:
				scores = []
				noise_types = []
				for noise_type, model in self.models.items():
					scores.append(model.score(features, **kwargs))
					noise_types.append(noise_type)
				scores = np.column_stack(scores)
				predicted_class = np.argmax(scores, axis=1)
				noise_types = np.array(noise_types)
			
		return (predicted_class, noise_types) + ((scores,) if return_scores else ())
			
	def test(self, labeled_features, silent=False) -> ConfusionTable:
		"""Test the classifier's performance on a dataset.
		
		Returns a confusion table."""

		res = ConfusionTable(
			sorted(labeled_features.keys()),
			sorted(self.model.get_noise_types() if self.model_type.MULTICLASS else self.models.keys())
		)
		start = time()

		for noise_type, features in labeled_features.items():
			predicted_class, noise_types = self.label(features, silent=silent)
			for confused_type in res.confused_labels:
				idx, = np.argwhere(noise_types == confused_type)
				res[noise_type, confused_type] = sum(predicted_class == idx)
			res[noise_type, res.TOTAL] = len(predicted_class)
		
		res.time = time() - start
			
		return res


	def save_to_file(self, filename, folder=None, verbose=False):
		"""Save a classifier to a file (usually with the .classifier extension).
		
		Note that the config is saved as well."""

		if folder is not None and not os.path.exists(folder):
			if verbose:
				print(f"Creating folder {folder}")
			os.mkdir(folder)
		if folder is not None:
			filename = os.path.join(folder, filename)
		with open(filename, 'wb') as f:
			pickle.dump(self, f)
	
	@staticmethod
	def from_file(filename, folder=None) -> 'Classifier':
		"""Read a classifier from file"""

		if folder is not None:
			filename = os.path.join(folder, filename)
		with open(filename, 'rb') as f:
			classifier = pickle.load(f)
			if isinstance(classifier, Classifier):
				return classifier
			else:
				raise ValueError(f"File {filename} does not contain a {Classifier} but a {type(classifier)}")

	@staticmethod
	def from_bytes(b: bytes):
		"""Read classifier(s) from a byte sequence.
		
		This is a generator yielding each classifier detected in the byte sequence."""
		c = pickle.loads(b)
		if isinstance(c, Classifier):
			yield c
		else:
			# Assume iterable
			for classifier in c:
				if not isinstance(classifier, Classifier):
					raise ValueError(f"Non-classifier in input")
				else:
					yield classifier

	@staticmethod
	def find_classifiers(folder):
		"""Find all files ending with .classifier in the provided folder, non-recursively"""

		files = next(os.walk(folder, followlinks=True))[2]
		return [os.path.join(folder, f) for f in files if os.path.splitext(f)[1] == ".classifier"]

def get_kaldi_root(assign=False):
	"""kaldi_io uses an environment variable to find the Kaldi library's root folder.
	This returns that environment variable if it exists, or $PWD/kaldi otherwise.
	
	If assign is True, the value of $PWD/kaldi is set as the Kaldi root."""

	if 'KALDI_ROOT' in os.environ:
		return os.environ['KALDI_ROOT']
	else:
		path = os.path.join(os.getcwd(), "kaldi")
		if assign:
			os.environ['KALDI_ROOT'] = path
		return path

def allow_gm_hmm_import(folder):
	"""Make the gm_hmm folder discoverable on sys.path, so that GenHMM and GMMHMM dependencies can be imported"""
	
	if os.path.exists(folder):
		sys.path.append(os.path.join(folder, os.pardir))
	else:
		raise FileNotFoundError("gm_hmm folder does not exist: " + folder)

def train(args):
	if args.kaldi is None:
		get_kaldi_root(assign=True)
	else:
		os.environ['KALDI_ROOT'] = args.kaldi
	allow_gm_hmm_import(args.gm_hmm)
	import kaldi_io
	from feature_extraction import extract_mfcc
	from model_gmmhmm import GMMHMM
	from model_genhmm import GenHMM
	from model_lstm import LSTM
	from model_svm import SVM
	from model_cnn import CNN
	supported_classifiers = [GMMHMM, GenHMM, LSTM, SVM, CNN]

	# Defaults for data and models folders
	if args.data is None:
		args.data = ""
	if args.train is None:
		args.train = os.path.join(args.data, "train")
	if not os.path.exists(args.models):
		print("Models folder does not exist: " + args.models)
		sys.exit(1)
	
	# Determine how the user wants their classifiers saved
	if len(args.write) == 1:
		args.write = args.write[0]
	if isinstance(args.write, str):
		if "<type>" not in args.write.lower() and len(args.classifiers) != 1 \
					and (args.read is not None and len(args.read) > 1 and "type" not in args.read[0].lower()):
			print("Invalid write specifier")
			sys.exit(1)
	elif len(args.write) != 0 and len(args.write) != (len(args.classifiers) or len(supported_classifiers)):
		print(f"Invalid number of write files ({len(args.write)}, expected 0 or {len(args.classifiers) or len(supported_classifiers)})")
		sys.exit(1)

	#if args.config_stdin:
	#	config = json.loads(input()) # This doesn't work, sys.stdin.buffer.read() might.

	#elif args.read is not None:
	if args.read is not None:
		# Read classifiers from disk
		config = dict()
		if len(args.read) == 0:
			args.read = args.write
		if len(args.read) == 1:
			args.read = args.read[0]
		if isinstance(args.read, str):
			if "<type>" in args.read.lower():
				args.read = [re.sub("<type>", c.__name__, args.read, flags=re.IGNORECASE)
					for c in supported_classifiers
						if len(args.classifiers) == 0 or c.__name__.lower() in args.classifiers]
			else:
				args.read = [args.read]
		classifiers = []
		for fn in args.read:
			classifiers.append(Classifier.from_file(fn, folder=args.models))
			if classifiers[-1].model_type.__name__ in config:
				raise ValueError(f"Can only train one instance of each type of classifier at once (duplicate {classifiers[-1].model_type.__name__})")
			if len(args.classifiers) != 0 and classifiers[-1].__name__.lower() not in args.classifiers:
				raise ValueError(f"The classifier in {fn} is a {classifiers[-1].model_type.__name__}, which is disallowed by --classifiers ({args.classifiers})")
			
			config[classifiers[-1].model_type.__name__] = classifiers[-1].config

	else:
		# Create new classifiers using the provided config
		with open(args.config, "r") as f:
			config = json.load(f)
	
	args.silent = args.silent or args.write_stdout
	
	if args.override:
		# Override any specified config values
		# This also updates the classifiers' configs any have been read from disk
		for path, value in args.override:
			d = config
			path = path.split(".")
			for i, part in enumerate(path[:-1]):
				possible = [k for k in d.keys() if k.lower() == part]
				if len(possible) == 0:
					if not args.silent:
						print("Creating config key", path[:i+1])
					d[part] = dict()
					possible = [part]
				d = d[possible[0]]
			d[path[-1]] = json.loads(value)

	# Get features
	with redirect_stdout(sys.stderr if args.silent else sys.stdout):
		feats = extract_mfcc(args.train, args.recompute)

	# Create classifiers (unless previously read from disk)
	if args.read is None:
		classifiers = []
		for supported_classifier in supported_classifiers:
			if len(args.classifiers) == 0 or supported_classifier.__name__.lower() in args.classifiers:
				classifiers.append(Classifier(feats.keys(), supported_classifier, config, silent=args.silent))

	# Train the classifiers for the specified number of epochs
	for epoch in range(1, args.epochs + 1):
		for i, classifier in enumerate(classifiers):
			if classifier.model_type == SVM and epoch > 1:
				continue
			# Train
			classifier.train(feats, silent=args.silent, models_folder=args.models)
			# Save classifier
			if isinstance(args.write, str):
				classifier.save_to_file(
					filename=re.sub(
						"<type>",
						classifier.model_type.__name__,
						args.write,
						flags=re.IGNORECASE), 
					folder=args.models
				)
			elif len(args.write) == len(classifiers):
				classifier.save_to_file(
					filename=args.write[i], 
					folder=args.models
				)
		if not args.silent and args.epochs != 1:
			print("Epoch", epoch)

	if args.write_stdout:
		# Send trained classifiers on stdout
		sys.stdout.buffer.write(pickle.dumps(classifiers))

	if not args.silent:
		print("Done")


def test(args):
	if args.kaldi is None:
		get_kaldi_root(assign=True)
	else:
		os.environ['KALDI_ROOT'] = args.kaldi
	allow_gm_hmm_import(args.gm_hmm)
	import kaldi_io
	from feature_extraction import extract_mfcc
	from model_gmmhmm import GMMHMM
	from model_genhmm import GenHMM
	from model_lstm import LSTM
	from model_svm import SVM
	from model_cnn import CNN
	supported_classifiers = [GMMHMM, GenHMM, LSTM, SVM, CNN]

	# Defaults for data and models folders
	if args.data is None:
		args.data = ""
	if args.test is None:
		args.test = os.path.join(args.data, "test")
	if not os.path.exists(args.models):
		print("Models folder does not exist: " + args.models)
		sys.exit(1)
	
	args.silent = args.silent or args.write_stdout

	# Read classifiers, either from disk or stdin
	if not args.silent:
		print("Reading classifiers ...", end="", flush=True)
	if args.read_stdin:
		classifiers = [c for c in Classifier.from_bytes(sys.stdin.buffer.read())]
	else:
		classifiers = []
	for f in args.read:
		if "<type>" in f.lower():
			f = [re.sub("<type>", c.__name__, f, flags=re.IGNORECASE) for c in supported_classifiers]
		else:
			f = [f]
		for fn in f:
			classifiers.append(Classifier.from_file(fn, folder=args.models))
	if not args.silent:
		print(" Done")

	if len(classifiers) == 0:
		print("Found no classifiers")
		sys.exit(1)
	if not args.silent:
		print(f"Loaded {len(classifiers)} classifier{'' if len(classifiers) == 1 else 's'}.")
	
	# Get features
	with redirect_stdout(sys.stderr if args.silent else sys.stdout):
		feats = extract_mfcc(args.test, args.recompute)

	if not args.silent:
		print("Calculating scores ...", flush=True)
	# Score all classifiers
	confusion_tables = []
	for classifier in classifiers:
		if not args.silent:
			print(classifier.model_type.__name__)
		confusion_tables.append(classifier.test(feats))
	if not args.silent:
		print("Done")

		# Print confusion tables to console
		for classifier, confusion_table in zip(classifiers, confusion_tables):
			print("Confusion table for", classifier.model_type.__name__)
			print(confusion_table)
			print()
	
	if args.write_stdout:
		# Send confusion tables on stdout
		sys.stdout.buffer.write(pickle.dumps(confusion_tables))

def _list(args):
	if args.kaldi is None:
		get_kaldi_root(assign=True)
	else:
		os.environ['KALDI_ROOT'] = args.kaldi
	allow_gm_hmm_import(args.gm_hmm)
	import kaldi_io
	from feature_extraction import extract_mfcc
	from model_gmmhmm import GMMHMM
	from model_genhmm import GenHMM
	from model_lstm import LSTM
	from model_svm import SVM
	from model_cnn import CNN
	supported_classifiers = [GMMHMM, GenHMM, LSTM, SVM, CNN]

	if not os.path.exists(args.models):
		print("Models folder does not exist: " + args.models)
		sys.exit(1)

	# Get information about all classifiers in the models folder
	classifiers = []
	for f in Classifier.find_classifiers(args.models):
		c = Classifier.from_file(f, folder=args.models)
		classifiers.append([
			os.path.basename(f),
			c.model_type.__name__,
			str(os.path.getsize(f))
		])
	
	if len(classifiers) > 0:
		classifiers = [["Filename", "Model Type", "Size in Bytes"]] + classifiers
		widths = [max(len(classifiers[row][i]) for row in range(len(classifiers))) for i in range(3)]
		print("  ".join(classifiers[0][i].ljust(widths[i]) for i in range(3)))
		for row in classifiers[1:]:
			print("  ".join(row[i].rjust(widths[i]) for i in range(3)))
	else:
		print(f"No classifiers in {args.models}")
	

"""
def test_vad(args):
	from feature_extraction import load_all_train

	if args.data is None:
		args.data = ""
	if args.train is None:
		args.train = os.path.join(args.data, "train")

	print(f"Evaluating VAD on {len(next(os.walk(args.train, followlinks=True))[2])} files in {args.train}")

	for frame_len in (10, 20, 30):
		print(f"Frame length: {frame_len} ms")
		for agg in range(4):
			a = ([activity for audio, activity in load_all_train(args.train, agg, frame_len)])
			total = sum(len(row) for row in a)
			speech = sum(sum(row) for row in a)
			print(f"\tVAD aggressiveness {agg} flags {speech/total:.0%} as speech")
"""

if __name__ == "__main__":
	supported_types = ["GenHMM", "GMMHMM", "LSTM", "SVM", "CNN"]
	parser = argparse.ArgumentParser(
		prog="classifier.py",
		description="Train or test a classifier. Supported types: " + ", ".join(supported_types)
	)
	parser.set_defaults(func=lambda a: parser.print_usage())
	subparsers = parser.add_subparsers()

	parser.add_argument("--profile", help="Profile the script", action="store_true")

	group = parser.add_argument_group("input folders")
	group.add_argument("--data", help="Path to data folder (default: $PWD/data)", default=os.path.join(os.getcwd(), "data"))
	group.add_argument("--train", help="Path to training data output folder (default: <DATA>/train)", default=None)
	group.add_argument("--test", help="Path to testing data output folder (default: <DATA>/test)", default=None)
	group.add_argument("--kaldi", help=f"Path to kaldi root folder (default: {get_kaldi_root()})", default=None)
	group.add_argument("--gm_hmm", help="Path to gm_hmm root folder (default: $PWD/gm_hmm). gm_hmm and its dependencies are needed for GMMHMM and GenHMM.", default=os.path.join(os.getcwd(), "gm_hmm"))
	group.add_argument("--models", help="Path to classifier models save folder (default: $PWD/classifier/models)", default=os.path.join(os.getcwd(), "classifier", "models"))

	subparser = subparsers.add_parser("train", help="Perform training")
	subparser.set_defaults(func=train)

	subparser.add_argument("-r", "--recompute", help="Ignore saved features and recompute", action="store_true")
	subparser.add_argument("-c", "--classifiers", help=f"Classes to train. If none are specified, all types are trained. Available: {', '.join(supported_types)}.", metavar="TYPE",
							nargs="*", choices=list(map(str.lower, supported_types))+[[]], type=str.lower, default=[])
	subparser.add_argument("-o", "--override", help="Override a classifier parameter. PATH takes the form 'classifier.category.parameter'. VALUE is a JSON-parsable object.", nargs=2, metavar=("PATH", "VALUE"), action="append")
	subparser.add_argument("-s", "--silent", help="Suppress all informational output on stdout (certain output is instead routed to stderr)", action="store_true")
	subparser.add_argument("-e", "--epochs", help="Number of epochs to run. Repeatedly trains each classifier <TYPE>.train.n_iter times and may save the intermediate classifiers (see --write). No effect on SVM training. Default: %(default)d", metavar="NUM_EPOCHS", type=int, default=1)
	
	out = subparser.add_argument_group("Classifier output")
	out.add_argument("-w", "--write", help="Files relative to <MODELS> to save classifier(s) to. Must be 0, 1 (containing <TYPE>), or same as number of classes. Default: %(default)s", metavar="FILE", nargs="*", default="latest_<TYPE>.classifier")
	out.add_argument("--write-stdout", help="Write resulting classifiers to stdout. Implies --silent.", action="store_true")

	configs = subparser.add_mutually_exclusive_group()
	configs.add_argument("--config", help="Path to classifier config file (default: $PWD/classifier/defaults.json)", default=os.path.join(os.getcwd(), "classifier", "defaults.json"))
	configs.add_argument("--read", help="Continue training existing classifiers for another epoch. Note that learning rates and iteration counts can be overridden by -o.\n"
		+ "Specify files relative to <MODELS> to read classifier(s) from. Must be 0, 1 (containing <TYPE>), or same as number of classes. If 0 files are provided the value of --write is used.", metavar="FILE", nargs="*", default=None)
	# TODO: Not working
	#configs.add_argument("--config-stdin", help="Read config from stdin", action="store_true")

	subparser = subparsers.add_parser("test", help="Perform testing")
	subparser.set_defaults(func=test)

	subparser.add_argument("-r", "--recompute", help="Ignore saved features and recompute", action="store_true")
	subparser.add_argument("-s", "--silent", help="Suppress all informational output on stdout (certain output is instead routed to stderr)", action="store_true")
	subparser.add_argument("--write-stdout", help="Write resulting confusion tables to stdout. Implies --silent.", action="store_true")

	inp = subparser.add_argument_group("Classifier input")
	inp.add_argument("read", help="Files relative to <MODELS> to read classifier(s) from. '<TYPE>' can be used in FILE to match all classifier type names. Default: %(default)s", metavar="FILE", nargs="*", default=["latest_<TYPE>.classifier"])
	inp.add_argument("--read-stdin", help="Read classifiers from stdin", action="store_true")

	"""
	group = subparser.add_argument_group("Voice Activity Detection settings")
	group.add_argument("-a", "--aggressiveness", help="VAD aggressiveness (0-3, default: %(default)d). Increase to flag more frames as non-speech.", default=2, type=int)
	group.add_argument("--vad-frame-length", help="VAD frame length in ms (10, 20 or 30, default: %(default)d)", default=20, type=int)
	"""

	subparser = subparsers.add_parser("list", help="List all classifiers in <MODELS>")
	subparser.set_defaults(func=_list)

	"""
	subparser = subparsers.add_parser("test-vad", help="Try all combinations of VAD parameters and list statistics")
	subparser.set_defaults(func=test_vad)
	"""

	args = parser.parse_args()
	if args.profile:
		import cProfile
		cProfile.run("args.func(args)", sort="cumulative")
	else:
		start = time()
		args.func(args)
		if ("silent" not in vars(args) or not args.silent) and ("write_stdout" not in vars(args) or not args.write_stdout):
			print(f"Total time: {time() - start:.1f} s")