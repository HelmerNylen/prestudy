#!/usr/bin/env python3
import argparse
import os
import numpy as np
import sys
from time import time

def get_kaldi_root(assign=False):
	if 'KALDI_ROOT' in os.environ:
		return os.environ['KALDI_ROOT']
	else:
		path = os.path.join(os.getcwd(), "..", "kaldi")
		if assign:
			os.environ['KALDI_ROOT'] = path
		return path

def get_noise_features(noise_folder, recompute):
	from feature_extraction import extract_mfcc

	noise_types = next(os.walk(noise_folder))[1]
	mfccs = dict()
	for noise_type in noise_types:
		mfccs[noise_type] = np.concatenate(
			tuple(mfcc for _, mfcc in extract_mfcc(os.path.join(noise_folder, noise_type), recompute))
		)
	return mfccs

def main(args):
	if args.kaldi is None:
		get_kaldi_root(assign=True)
	else:
		os.environ['KALDI_ROOT'] = args.kaldi
	import kaldi_io

	if args.data is None:
		args.data = ""
	if args.train is None:
		args.train = os.path.join(args.data, "train")
	if args.test is None:
		args.test = os.path.join(args.data, "test")
	if args.noise is None:
		args.noise = os.path.join(args.data, "noise")

	feats = get_noise_features(args.noise, args.recompute)
	print(dict((t, feats[t].shape) for t in feats))

	"""
	if args.query:
		for frame_len in (10, 20, 30):
			print(f"Frame length: {frame_len} ms")
			for agg in range(4):
				a = ([activity for audio, activity in load_all_train(args.train, agg, frame_len)])
				total = sum(len(row) for row in a)
				speech = sum(sum(row) for row in a)
				print(f"\tVAD aggressiveness {agg} flags {speech/total:.0%} as speech")
	else:
		features = load_all_train(args.train, args.aggressiveness, args.vad_frame_length)
	"""

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog="classifier.py",
		description="Train or test the classifier"
	)
	group = parser.add_argument_group("input folders")
	group.add_argument("--data", help="Path to data folder. Defaults to current folder.", default=None)
	group.add_argument("--train", help="Path to training data output folder (default: <DATA>/train)", default=None)
	group.add_argument("--test", help="Path to testing data output folder (default: <DATA>/test)", default=None)
	group.add_argument("--noise", help="Path to noise data output folder (default: <DATA>/noise)", default=None)
	group.add_argument("--kaldi", help=f"Path to kaldi root folder (default: {get_kaldi_root()})", default=None)

	# Bör ligga under ett testkommando
	group = parser.add_argument_group("Voice Activity Detection settings")
	group.add_argument("-a", "--aggressiveness", help="VAD aggressiveness (0-3, default: %(default)d). Increase to flag more frames as non-speech.", default=2, type=int)
	group.add_argument("--vad-frame-length", help="VAD frame length in ms (10, 20 or 30, default: %(default)d)", default=20, type=int)

	parser.add_argument("-r", "--recompute", help="Ignore saved features and recompute", action="store_true")
	
	# Borde vara ett kommando istället för en flagga
	parser.add_argument("--query", help="Try all combinations of VAD parameters and list statistics", action="store_true")
	parser.add_argument("--profile", help="Profile the dataset creator", action="store_true")

	args = parser.parse_args()
	if args.profile:
		import cProfile
		cProfile.run("main(args)")
	else:
		start = time()
		main(args)
		print(f"Total time: {time() - start:.1f} s")