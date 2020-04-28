#!/usr/bin/env python3
import argparse
import os
import numpy as np
import sys
import pickle
from time import time
from model import Model
from confusion_table import ConfusionTable

class Classifier:
	def __init__(self, noise_types, model_type: Model, *args, **kwargs):
		self.model_type = model_type

		if self.model_type.MULTICLASS:
			self.model = model_type(*args, noise_types=noise_types, **kwargs)
		else:
			self.models = dict(
				(noise_type, model_type(*args, **kwargs))
				for noise_type in noise_types
			)
	
	def train(self, labeled_features, verbose=True, save_models_to=None):
		if self.model_type.MULTICLASS:
			self.model.train(labeled_features)
		else:
			for noise_type, model in self.models.items():
				if verbose:
					print(f"Training {noise_type} {model.__class__.__name__} model")

				model.train(labeled_features[noise_type])
				
				if save_models_to is not None:
					if not os.path.exists(save_models_to):
						if verbose:
							print(f"Creating folder {save_models_to}")
						os.mkdir(save_models_to)
					model.save_to_file(os.path.join(save_models_to, self.filename(noise_type)))
	
	def label(self, features, return_scores=False):
		if self.model_type.MULTICLASS:
			scores = self.model.score(features)
			predicted_class = np.argmax(scores, axis=1)
			noise_types = np.array(self.model.get_noise_types())

		else:
			scores = []
			noise_types = []
			for noise_type, model in self.models.items():
				scores.append(model.score(features))
				noise_types.append(noise_type)
			scores = np.column_stack(scores)
			predicted_class = np.argmax(scores, axis=1)
			noise_types = np.array(noise_types)
			
		return (predicted_class, noise_types) + ((scores,) if return_scores else ())
			
	def test(self, labeled_features) -> ConfusionTable:
		res = ConfusionTable(
			sorted(labeled_features.keys()),
			sorted(self.model.get_noise_types() if self.model_type.MULTICLASS else self.models.keys())
		)
		start = time()

		for noise_type, features in labeled_features.items():
			predicted_class, noise_types = self.label(features)
			for confused_type in res.confused_labels:
				idx, = np.argwhere(noise_types == confused_type)
				res[noise_type, confused_type] = sum(predicted_class == idx)
			res[noise_type, res.TOTAL] = len(predicted_class)
		
		res.time = time() - start
			
		return res

	
	@staticmethod
	def from_file(*, filename=None, folder=None, model_type=None) -> 'Classifier':
		if filename is None:
			if model_type is None:
				raise ValueError()
			filename = model_type + ".classifier"
		if folder is not None:
			filename = os.path.join(folder, filename)
		with open(filename, 'rb') as f:
			classifier = pickle.load(f)
			if isinstance(classifier, Classifier):
				return classifier
			else:
				raise ValueError(f"File {filename} does not contain a Classifier")

	def save_to_file(self, *, folder=None, filename=None):
		if folder is not None and not os.path.exists(folder):
			print(f"Creating folder {folder}")
			os.mkdir(folder)
		filename = filename or self.filename()
		if folder is not None:
			filename = os.path.join(folder, filename)
		with open(filename, 'wb') as f:
			pickle.dump(self, f)
	
	def filename(self, noise_type=None):
		if noise_type is None:
			return self.model_type.__name__ + ".classifier"
		else:
			return self.model_type.__name__ + "_" + noise_type + ".model"
	@staticmethod
	def find_classifiers(folder):
		files = next(os.walk(folder, followlinks=True))[2]
		return [os.path.join(folder, f) for f in files if os.path.splitext(f)[1] == ".classifier"]

def get_kaldi_root(assign=False):
	if 'KALDI_ROOT' in os.environ:
		return os.environ['KALDI_ROOT']
	else:
		path = os.path.join(os.getcwd(), "kaldi")
		if assign:
			os.environ['KALDI_ROOT'] = path
		return path

def allow_gm_hmm_import(folder):
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

	if args.data is None:
		args.data = ""
	if args.train is None:
		args.train = os.path.join(args.data, "train")
	if not os.path.exists(args.models):
		print("Models folder does not exist: " + args.models)
		sys.exit(1)

	feats = extract_mfcc(args.train, args.recompute)

	classifiers = []
	"""
	n = 3
	K = 2
	niter = 5
	classifiers.append(Classifier(feats.keys(), GMMHMM,
		n_components=n, n_mix=K, covariance_type="diag",
		tol=-np.inf, n_iter=niter, verbose=True
	))
	
	# Verkar bara fungera när net_D är 12??????
	classifiers.append(Classifier(feats.keys(), GenHMM,
		n_states=3, n_prob_components=2, em_skip=4, device="cpu",
		lr=0.004, net_H=24, net_D=12, net_nchain=4, p_drop=0,
		mask_type="cross", startprob_type="first", transmat_type="triangular"
	))
	"""
	classifiers.append(Classifier(feats.keys(), LSTM,
		hidden_dim=20, num_layers=2
	))
	"""
	classifiers.append(Classifier(feats.keys(), SVM,
		frame_len=20, frame_overlap=5
	))
	"""

	for classifier in classifiers:
		classifier.train(feats, save_models_to=args.models)
		classifier.save_to_file(folder=args.models)
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

	if args.data is None:
		args.data = ""
	if args.test is None:
		args.test = os.path.join(args.data, "test")
	if not os.path.exists(args.models):
		print("Models folder does not exist: " + args.models)
		sys.exit(1)

	feats = extract_mfcc(args.test, args.recompute)
	
	print("Reading classifiers ...", end="", flush=True)
	classifiers = [Classifier.from_file(filename=f) for f in Classifier.find_classifiers(args.models)]
	print(" Done")

	if len(classifiers) == 0:
		print("Found no classifiers")
		sys.exit(1)
	print(f"Found classifiers: {', '.join(c.model_type.__name__ for c in classifiers)}")
	
	print("Calculating scores ...", flush=True)
	confusion_tables = []
	for classifier in classifiers:
		print(classifier.model_type.__name__)
		confusion_tables.append(classifier.test(feats))
	print("Done")
	
	for classifier, confusion_table in zip(classifiers, confusion_tables):
		print("Confusion table for", classifier.model_type.__name__)
		print(confusion_table)
		print()
	

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

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog="classifier.py",
		description="Train or test the classifier"
	)
	parser.set_defaults(func=lambda a: parser.print_usage())
	subparsers = parser.add_subparsers()

	parser.add_argument("--profile", help="Profile the classifier", action="store_true")

	group = parser.add_argument_group("input folders")
	group.add_argument("--data", help="Path to data folder (default: $PWD/data)", default=os.path.join(os.getcwd(), "data"))
	group.add_argument("--train", help="Path to training data output folder (default: <DATA>/train)", default=None)
	group.add_argument("--test", help="Path to testing data output folder (default: <DATA>/test)", default=None)
	group.add_argument("--kaldi", help=f"Path to kaldi root folder (default: {get_kaldi_root()})", default=None)
	group.add_argument("--gm_hmm", help="Path to gm_hmm root folder (default: $PWD/gm_hmm)", default=os.path.join(os.getcwd(), "gm_hmm"))
	group.add_argument("--models", help="Path to classifier models save folder (default: $PWD/classifier/models)", default=os.path.join(os.getcwd(), "classifier", "models"))

	subparser = subparsers.add_parser("train", help="Perform training")
	subparser.set_defaults(func=train)

	subparser.add_argument("-r", "--recompute", help="Ignore saved features and recompute", action="store_true")

	subparser = subparsers.add_parser("test", help="Perform testing")
	subparser.set_defaults(func=test)

	subparser.add_argument("-r", "--recompute", help="Ignore saved features and recompute", action="store_true")

	group = subparser.add_argument_group("Voice Activity Detection settings")
	group.add_argument("-a", "--aggressiveness", help="VAD aggressiveness (0-3, default: %(default)d). Increase to flag more frames as non-speech.", default=2, type=int)
	group.add_argument("--vad-frame-length", help="VAD frame length in ms (10, 20 or 30, default: %(default)d)", default=20, type=int)

	subparser = subparsers.add_parser("test-vad", help="Try all combinations of VAD parameters and list statistics")
	subparser.set_defaults(func=test_vad)

	args = parser.parse_args()
	if args.profile:
		import cProfile
		cProfile.run("args.func(args)", sort="cumulative")
	else:
		start = time()
		args.func(args)
		print(f"Total time: {time() - start:.1f} s")