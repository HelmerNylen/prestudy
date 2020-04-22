#!/usr/bin/env python3
import argparse
import os
import numpy as np
import sys
from time import time

class Classifier():
	...

def get_kaldi_root(assign=False):
	if 'KALDI_ROOT' in os.environ:
		return os.environ['KALDI_ROOT']
	else:
		path = os.path.join(os.getcwd(), "..", "kaldi")
		if assign:
			os.environ['KALDI_ROOT'] = path
		return path

def train(args):
	if args.kaldi is None:
		get_kaldi_root(assign=True)
	else:
		os.environ['KALDI_ROOT'] = args.kaldi
	import kaldi_io
	from feature_extraction import extract_mfcc

	if args.data is None:
		args.data = ""
	if args.train is None:
		args.train = os.path.join(args.data, "train")

	feats = extract_mfcc(args.train, args.recompute)
	for noise_type in feats:
		print(noise_type)
		for key, mat in feats[noise_type]:
			... # PyTorch dataloading etc.

def test(args):
	if args.data is None:
		args.data = ""
	if args.test is None:
		args.test = os.path.join(args.data, "test")

	raise NotImplementedError()

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
	group.add_argument("--data", help="Path to data folder. Defaults to current folder.", default=None)
	group.add_argument("--train", help="Path to training data output folder (default: <DATA>/train)", default=None)
	group.add_argument("--test", help="Path to testing data output folder (default: <DATA>/test)", default=None)
	group.add_argument("--kaldi", help=f"Path to kaldi root folder (default: {get_kaldi_root()})", default=None)

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