#!/usr/bin/env python3
import argparse
import os
import numpy as np
import subprocess
import sys
from time import time
from tempfile import NamedTemporaryFile
from shutil import copy2

def get_kaldi_root(assign=False):
	if 'KALDI_ROOT' in os.environ:
		return os.environ['KALDI_ROOT']
	else:
		path = os.path.join(os.getcwd(), "..", "kaldi")
		if assign:
			os.environ['KALDI_ROOT'] = path
		return path

# Flytta till create_dataset?
def prep_folder(folder, force=False, to_mono=True, downsample=True):
	files = next(os.walk(folder))[2]
	commands = []
	with NamedTemporaryFile() as tmp:
		for f in files:
			path, ext = os.path.splitext(os.path.join(folder, f))
			if ext.lower() in (".txt", ".ark"):
				continue
			info = subprocess.run([
				'sox',
				'--i',
				path + ext],
				stdout=subprocess.PIPE,
				check=True,
				encoding=sys.getdefaultencoding()
			).stdout
			info = (line.split(":") for line in info.splitlines() if len(line.strip()) > 0)
			info = dict((splitline[0].strip(), ":".join(splitline[1:]).strip()) for splitline in info)
			
			command = []
			if to_mono and not info["Channels"] == "1":
				command.append(("-c", "1"))
			if info["Precision"].rstrip("-bit") != "16":
				command.append(("-b", "16"))
			if downsample and int(info["Sample Rate"]) > 16000:
				command.append(("-r", "16000"))

			if len(command) > 0 or ext.lower() != ".wav":
				command.append(("-t", "wav"))
				commands.append(['sox', path + ext] + [x for c in command for x in c] + [tmp.name])
				commands.append((f"Move {tmp.name} to {path + '.wav'}", lambda x, y: copy2(x, y, follow_symlinks=True), tmp.name, path + ".wav"))

		if not force:
			print("Suggesting the following operations:")
			for c in commands:
				if type(c) is tuple:
					print(c[0])
				else:
					print(" ".join(c))
			if not input("Continue? (y/n): ").lower().lstrip().startswith("y"):
				print("Discarding suggestions")
				return
		
		for c in commands:
			if type(c) is tuple:
				c[1](*c[2:])
			else:
				result = subprocess.run(c, check=True, stderr=subprocess.PIPE, encoding=sys.getdefaultencoding())
				if len(result.stderr) > 0:
					print("stderr from", " ".join(result.args))
					print(result.stderr)


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
	# Gör till kommando
	prep_folder(os.path.join(args.noise, "ac"))
	prep_folder(os.path.join(args.noise, "electric"))

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