#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import numpy as np
from time import time
import matplotlib.pyplot as plt

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

	args = parser.parse_args()
	start = time()
	args.func(args)
	print(f"Total time: {time() - start:.1f} s")