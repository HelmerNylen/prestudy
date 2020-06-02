#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import numpy as np
from tempfile import NamedTemporaryFile
from time import time
from random import sample, random
from math import floor
import matplotlib.pyplot as plt

# Get the MFCCs of each noise type in a folder
def uncached_tmp_extract_mfcc(folder):
	import kaldi_io
	from classifier.feature_extraction import type_sorted_files
	tsf = type_sorted_files(folder).items()

	start = time()

	arks = dict()
	for noise_type, filenames in tsf:
		f = NamedTemporaryFile(suffix=".ark")
		print(f"Extracting {noise_type} features in {folder} ...", end="", flush=True)
		result = subprocess.run([
			'compute-mfcc-feats',
			'--dither=0.1',
			'scp:-',
			'ark:' + f.name],
			input="\n".join(fn + " " + os.path.join(folder, fn)
				for fn in filenames) + "\n",
			encoding=sys.getdefaultencoding(),
			stderr=subprocess.PIPE
		)
		print(" Done")
		if result.returncode:
			print(result.stderr)
			result.check_returncode()
		else:
			for line in result.stderr.splitlines(False):
				if not line.startswith("LOG") and line.strip() != " ".join(result.args).strip():
					print(line)
		print(f"Saved to {f.name}")
		arks[noise_type] = f
	
	if len(arks) == 0:
		raise ValueError(f"Folder {folder} is not a dataset")
	
	res = dict()
	# Read all arks
	for noise_type, ark in arks.items():
		res[noise_type] = [mat for _, mat in kaldi_io.read_mat_ark(ark.name)]
		ark.close()
	return res, time() - start

def librosa_extract(folder):
	from librosa.core import load
	from librosa.feature import mfcc
	from classifier.feature_extraction import type_sorted_files
	tsf = type_sorted_files(folder).items()

	start = time()

	res = dict()
	for noise_type, filenames in tsf:
		print(f"Extracting {noise_type} features in {folder} ...", end="", flush=True)
		t = []
		for fn in filenames:
			y, fs = load(os.path.join(folder, fn), None, None)
			# Kanske vill ha lifter=22
			t.append(mfcc(
				y, fs,
				n_mfcc=13,
				n_fft=round(0.025 * fs),
				hop_length=round(0.010 * fs),
				window="hamming",
				n_mels=23,
				fmin=20,
				lifter=22
			))
		res[noise_type] = t
		print(" Done")

	return res, time() - start


def feature_comparison(args):
	kaldi = uncached_tmp_extract_mfcc(args.folder)
	librosa = librosa_extract(args.folder)
	print(f"Kaldi\t{kaldi[1]:.2f} s")
	print(f"Librosa\t{librosa[1]:.2f} s")

	if args.plot:
		from librosa.display import specshow
		noise_type = next(iter(kaldi[0]))
		(kaldi_mfcc, lib_mfcc), = sample(list(zip(kaldi[0][noise_type], librosa[0][noise_type])), 1)
		lib_mfcc = lib_mfcc.T
		print(type(kaldi_mfcc), type(lib_mfcc))
		print(kaldi_mfcc.shape, lib_mfcc.shape)

		i = random()

		plt.figure(figsize=(10, 8))

		plt.subplot(2, 1, 1)
		specshow(kaldi_mfcc.T, x_axis='time')
		plt.colorbar()
		plt.title("Kaldi")
		print("Kaldi min:", kaldi_mfcc.min())
		print("Kaldi max:", kaldi_mfcc.max())
		print(f"Kaldi at {i:.2f}: {kaldi_mfcc[floor(kaldi_mfcc.shape[0] * i), :]}")

		plt.subplot(2, 1, 2)
		specshow(lib_mfcc.T, x_axis='time')
		plt.colorbar()
		plt.title("Librosa")
		print("Librosa min:", lib_mfcc.min())
		print("Librosa max:", lib_mfcc.max())
		print(f"Librosa at {i:.2f}: {lib_mfcc[floor(lib_mfcc.shape[0] * i), :]}")

		plt.tight_layout()
		plt.show()

	print("Done")

def timit(args):
	timit_root = os.path.join(os.getcwd(), "data", "timit")

	print("Reading TIMIT ...", flush=True, end="")
	lengths = []
	all_fns = []
	for d, _, fns in os.walk(timit_root, followlinks=True):
		for fn in fns:
			if fn.endswith(".wav"):
				lengths.append(float(subprocess.run(["soxi", "-D", os.path.join(d, fn)], stdout=subprocess.PIPE).stdout))
				all_fns.append(os.path.join(d, fn))
	lengths = np.array(lengths)
	print(" Done")

	print(f"N: {len(lengths)}")
	print(f"Min: {min(lengths):.2f} s, by {all_fns[np.argmin(lengths)]}")
	print(f"Max: {max(lengths):.2f} s, by {all_fns[np.argmax(lengths)]}")
	print(f"Mean: {np.mean(lengths):.2f} s")
	print(f"Median: {np.median(lengths):.2f} s")
	print(f"Std. dev.: {np.std(lengths):.2f} s")

	if args.plot_lengths:
		plt.figure()
		plt.hist(lengths, bins=np.arange(0, np.ceil(max(lengths)), 1/3))
		plt.xlabel("Length [s]")
		plt.ylabel("Number of utterances")
		plt.title("TIMIT utterance lengths")
		plt.savefig("timit.png")
		print("Plot saved")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog="evaluate.py",
		description="Evaluate different classifiers' performace on different datasets"
	)
	parser.set_defaults(func=lambda a: parser.print_usage())
	subparsers = parser.add_subparsers()

	subparser = subparsers.add_parser("timit", help="Print TIMIT stats")
	subparser.set_defaults(func=timit)

	subparser.add_argument("-p", "--plot-lengths", help="Create a histogram of utterance lengths", action="store_true")

	subparser = subparsers.add_parser("feature-comparison", help="Compare Librosa and Kaldi")
	subparser.set_defaults(func=feature_comparison)

	subparser.add_argument("folder", help="The folder to extract features from")
	subparser.add_argument("-p", "--plot", help="Create a plot of extracted features", action="store_true")

	args = parser.parse_args()
	start = time()
	args.func(args)
	print(f"Total time: {time() - start:.1f} s")