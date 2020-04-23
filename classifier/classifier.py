#!/usr/bin/env python3
import argparse
import os
import numpy as np
import sys
import pickle
from time import time

class Classifier():
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

def classifier_file_name(noise_type, classifier_type):
	return classifier_type.__name__ + "_" + noise_type + ".model"

def train(args):
	if args.kaldi is None:
		get_kaldi_root(assign=True)
	else:
		os.environ['KALDI_ROOT'] = args.kaldi
	allow_gm_hmm_import(args.gm_hmm)
	import kaldi_io
	from feature_extraction import extract_mfcc
	from classifier_gmmhmm import GMMHMM

	if args.data is None:
		args.data = ""
	if args.train is None:
		args.train = os.path.join(args.data, "train")
	if not os.path.exists(args.models):
		print("Models folder does not exist: " + args.models)
		return

	feats = extract_mfcc(args.train, args.recompute, concatenate=True)

	classifier_sets = []
	n = 3
	K = 2
	niter = 5
	classifier_sets.append(dict(
		(noise_type, GMMHMM(
			n_components=n, n_mix=K, covariance_type="diag",
			tol=-np.inf, n_iter=niter, verbose=True
		)) for noise_type in feats
	))
	for classifier_set in classifier_sets:
		for noise_type, classifier in classifier_set.items():
			print(f"Training {noise_type} {classifier.__class__.__name__} classifier")
			classifier.train(feats[noise_type], True)
			classifier.save_to_file(os.path.join(args.models, classifier_file_name(noise_type, classifier.__class__)))
	print("Done")

def test(args):
	if args.kaldi is None:
		get_kaldi_root(assign=True)
	else:
		os.environ['KALDI_ROOT'] = args.kaldi
	allow_gm_hmm_import(args.gm_hmm)
	import kaldi_io
	from feature_extraction import extract_mfcc
	from classifier_gmmhmm import GMMHMM

	if args.data is None:
		args.data = ""
	if args.test is None:
		args.test = os.path.join(args.data, "test")
	if not os.path.exists(args.models):
		print("Models folder does not exist: " + args.models)
		return

	feats = extract_mfcc(args.test, args.recompute, concatenate=True)
	
	print("Reading models ...", end="", flush=True)
	classifier_sets = []
	classifier_types = (GMMHMM,)
	for classifier_type in classifier_types:
		classifier_sets.append(dict(
			(noise_type, classifier_type.from_file(
				os.path.join(args.models, classifier_file_name(noise_type, classifier_type))
			)) for noise_type in feats
	))
	print(" Done")

	print("Calculating scores ...", end="", flush=True)
	confusion_tables = []
	for classifier_set in classifier_sets:
		confusion_table = []
		for noise_type in sorted(feats.keys()):
			scores = []
			for classifier_nt in sorted(classifier_set.keys()):
				scores.append(classifier_set[classifier_nt].test(feats[noise_type], is_concatenated=True))
			scores = np.column_stack(scores)
			predicted_class = np.argmax(scores, axis=1)
			confusion_table.append([noise_type] + [sum(predicted_class == nti) / len(predicted_class) for nti in range(len(feats))])
		confusion_tables.append(confusion_table)
	print("Done")
	
	for classifier_type, confusion_table in zip(classifier_types, confusion_tables):
		widths = [max(map(lambda row: len(row[0]), confusion_table))] + [max(7, len(nt)) for nt in sorted(feats.keys())]

		print("Confusion table for", classifier_type.__name__)
		print("  ".join(["true".center(widths[0], '-'), "confused with".center(sum(widths[1:]) + 2*(len(widths) - 2), '-')]))
		print("  ".join(s.rjust(widths[i]) for i, s in enumerate([""] + list(sorted(feats.keys())))))
		for row in confusion_table:
			print("  ".join(val.rjust(widths[0]) if type(val) is str else f"{val:>{widths[i]}.2%}" for i, val in enumerate(row)))
		print()
		print(f"Total accuracy: {sum(confusion_table[i][i+1] for i in range(len(feats))) / len(feats):.2%}")
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