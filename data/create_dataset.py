#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import math
from time import time

def _noise_files(data_folder, rel_noise_folder, keep=[".wav"]):
	noise_folder = os.path.join(data_folder, rel_noise_folder)
	walk = os.walk(noise_folder, followlinks=True)
	# Skip .WAV (uppercase) files as these are unprocessed timit sphere files (i.e. don't use os.path.splitext(fn)[-1].lower())
	walk = ((dp, dn, [fn for fn in filenames if os.path.splitext(fn)[-1] in keep])
			for dp, dn, filenames in walk)
	walk = ((os.path.relpath(dirpath, noise_folder), (dirpath, filenames))
			for dirpath, dirnames, filenames in walk
			if dirpath != noise_folder and len(filenames) > 0)
	return walk

def _speech_files(data_folder, rel_speech_folder, keep=[".wav"]):
	speech_folder = os.path.join(data_folder, rel_speech_folder)
	walk = os.walk(speech_folder, followlinks=True)
	# Skip .WAV (uppercase) files as these are unprocessed timit sphere files (i.e. don't use os.path.splitext(fn)[-1].lower())
	walk = ((dp, dn, [fn for fn in filenames if os.path.splitext(fn)[-1] in keep])
			for dp, dn, filenames in walk)
	walk = ((dirpath, filenames)
			for dirpath, dirnames, filenames in walk
			if len(filenames) > 0)
	return walk

def read_noise(data_folder, rel_noise_folder):
	noise = dict(_noise_files(data_folder, rel_noise_folder))
	n_noisefiles = sum(len(noise[noisetype][1]) for noisetype in noise)
	return noise, n_noisefiles

def read_speech(data_folder, rel_speech_folder):
	speech = list(_speech_files(data_folder, rel_speech_folder))
	n_speechfiles = sum(len(f) for d, f in speech)
	return speech, n_speechfiles

def _output_dirs(train: str, test: str, move: bool):
	"""Prepares output directories and moves the existing files to a new subdirectory."""
	for folder in [train, test]:
		if os.path.exists(folder):
			files = next(os.walk(folder))[2]
			if len(files) > 0:
				if move:
					subfolder = "old"
					subfolderind = 1
					while os.path.exists(os.path.join(folder, subfolder + str(subfolderind))):
						subfolderind += 1
					subfolder = os.path.join(folder, subfolder + str(subfolderind))
					os.mkdir(subfolder)
					print(f"Moving {len(files)} file{'' if len(files) == 1 else 's'} from {folder} to {subfolder} ...", end="")
					for fn in files:
						os.rename(os.path.join(folder, fn), os.path.join(subfolder, fn))
					print(" Done")
				else:
					print(f"{len(files)} file{'' if len(files) == 1 else 's'} already in {folder}")
		else:
			os.mkdir(folder)
			print(f"Created {folder}")

def _get_available_filename(folder: str, label: str, ext: str, startind=0, pad=3) -> str:
	"""Returns a free filename by increasing an appended number."""
	ind = startind
	path = os.path.join(folder, label + str(ind).rjust(pad, "0") + "." + ext)
	while os.path.exists(path):
		ind += 1
		path = os.path.join(folder, label + str(ind).rjust(pad, "0") + "." + ext)
	return path

def _get_available_filenames(folder: str, labels: list, ext: str, startind=0, pad=3) -> list:
	"""Returns multiple free filenames by increasing an appended number."""
	inds = dict()
	paths = []
	path = None
	for label in labels:
		while path is None or os.path.exists(path):
			if label not in inds:
				inds[label] = startind
			path = os.path.join(folder, label + str(inds[label]).rjust(pad, "0") + "." + ext)
			inds[label] += 1
		paths.append(path)
		path = None
	return paths

def create(args):
	# Matlab must be loaded before any module which depends on random
	print("Importing Matlab ...", end="", flush=True)
	import matlab.engine
	print(" Done")

	import random
	from degradations import get_degradations, setup_matlab_degradations

	# Begin starting Matlab asynchronously
	m_eng = matlab.engine.start_matlab(background=True)

	# Find all source sound files
	noise, n_noisefiles = read_noise(args.data, args.noise)
	speech, n_speechfiles = read_speech(args.data, args.speech)
	
	if n_noisefiles == 0:
		print("No noise files found")
		sys.exit(1)
	if n_speechfiles == 0:
		print("No speech files found")
		sys.exit(1)

	# Parse and verify degradation classes
	available_classes = [d for d in get_degradations(noise) if d not in ("pad",)]
	if len(args.classes) == 0 or args.classes == ["all"]:
		args.classes = map(str, available_classes)
	tmp = []
	for c in args.classes:
		try:
			tmp.append(available_classes[available_classes.index(None if c.lower() == "none" else c.lower())])
		except ValueError:
			print(f"Unknown degradation class \"{c}\"")
			print("Available:", ", ".join(available_classes))
			print()
			print("(Stopping Matlab ...", end="", flush=True)
			m_eng.cancel()
			print(" Done)")
			sys.exit(1)
	args.classes = tmp
	n_classes = len(args.classes)

	# Verify partition
	if 0 <= args.train <= 1:
		args.train = round(args.train * n_speechfiles)
	else:
		args.train = int(args.train)
	if 0 <= args.test <= 1:
		args.test = round(args.test * n_speechfiles)
	else:
		args.test = int(args.test)
	if args.train < 0 or args.test < 0 or args.train + args.test > n_speechfiles:
		print(f"Invalid partition of files: {args.train} train, {args.test} test, {args.train + args.test} total")
		sys.exit(1)
	print(f"Speech: {args.train} training files and {args.test} testing files")

	# Make sure output directories exist and optionally are clean
	if args.output_train is None:
		args.output_train = os.path.join(args.data, "train")
	if args.output_test is None:
		args.output_test = os.path.join(args.data, "test")
	_output_dirs(args.output_train, args.output_test, not args.keep_old)
	
	# Partition speech set
	speech = [os.path.join(dirpath, filepath) for dirpath, filepaths in speech for filepath in filepaths]
	random.shuffle(speech)
	speech_train = speech[:args.train]
	speech_test = speech[args.train:args.train + args.test]
	del speech

	# Partition noise set
	noise_train = dict()
	noise_test = dict()
	print("Noise:")
	for noise_type, tup in noise.items():
		folder, files = tup
		random.shuffle(files)
		split = round(len(files) * args.train / (args.train + args.test))
		noise_train[noise_type] = (folder, files[:split])
		noise_test[noise_type] = (folder, files[split:])
		print(f"\t{noise_type}: {split} training files and {len(files) - split} testing files")
	del noise

	# Assign noise types to speech files
	labels_train = list(range(n_classes)) * (len(speech_train) // n_classes)
	labels_train = labels_train + ([n_classes - 1] * (len(speech_train) - len(labels_train)))
	random.shuffle(labels_train)
	labels_test = list(range(n_classes)) * (len(speech_test) // n_classes)
	labels_test = labels_test + ([n_classes - 1] * (len(speech_test) - len(labels_test)))
	random.shuffle(labels_test)

	# Setup Matlab
	print("Setting up Matlab ...", end="", flush=True)
	m_eng = m_eng.result()
	if len(args.data) > 0:
		m_eng.cd(args.data, nargout=0)
	if args.adt is None:
		args.adt = os.path.join(args.data, "adt")
	args.adt = os.path.join(args.adt, "AudioDegradationToolbox")
	if not os.path.exists(args.adt):
		print("\nAudio Degradation Toolbox folder does not exist:", args.adt)
		sys.exit(1)
	print(" Done")

	# Create datasets
	for t, speech_t, labels_t, noise_t, output_t in [
		["train", speech_train, labels_train, noise_train, args.output_train],
		["test", speech_test, labels_test, noise_test, args.output_test]
	]:
		print(f"Creating {t}ing data")

		print("Setting up Matlab arguments")
		degradations = [args.classes[label] for label in labels_t]
		if args.pad is not None:
			degradations = [["pad", degradation] for degradation in degradations]
		# Load degradation instructions into Matlab memory
		# The overhead for passing audio data between Python and Matlab is extremely high,
		# so Matlab is only sent the filenames and left to figure out the rest for itself
		# Further, certain datatypes (struct arrays) cannot be sent to/from Python
		setup_matlab_degradations(noise_t, speech_t, degradations, m_eng, args, "degradations")

		# Store all variables in Matlab memory
		m_eng.workspace["speech_files"] = speech_t
		m_eng.eval("speech_files = string(speech_files);", nargout=0)
		m_eng.workspace["output_files"] = _get_available_filenames(
			output_t,
			((args.classes[label] or "none") + "_" for label in labels_t),
			"wav",
			1,
			math.ceil(math.log10(len(speech_t) / n_classes))
		)
		m_eng.eval("output_files = string(output_files);", nargout=0)
		m_eng.workspace["use_cache"] = True
		m_eng.workspace["adt_root"] = args.adt

		print("Creating samples")
		try:
			# Actual function call to create_samples.m, which in turn uses the ADT
			m_eng.eval("create_samples(speech_files, degradations, output_files, use_cache, adt_root);", nargout=0)
			# Save degradation instructions so that a sample can be recreated if needed
			m_eng.save(os.path.join(output_t, "degradations.mat"), "speech_files", "degradations", "output_files", nargout=0)
			
		except matlab.engine.MatlabExecutionError as e: # pylint: disable=E1101
			print(e)
			print("A Matlab error occurred")
			print("Launching Matlab desktop so you may debug. Press enter to exit.", end="", flush=True)
			m_eng.desktop(nargout=0)
			input()
			raise e

		print("Done")
		
	m_eng.exit()
	print("Dataset created")

def list_files(args):
	# This currently imports Matlab (in degradations) and doesn't need to
	from degradations import get_degradations

	# Find all source sound files
	noise, n_noisefiles = read_noise(args.data, args.noise)
	speech, n_speechfiles = read_speech(args.data, args.speech)
	
	print(f"Found {n_noisefiles} noise files")
	for noisetype in noise:
		print(f"\t{len(noise[noisetype][1])} of type \"{noisetype}\"")
	print(f"Found {n_speechfiles} speech files")
	for t in ["test", "train"]:
		print(f"\t{sum(len(fs) for d, fs in speech if t in d.lower())} in set \"{t}\"")
	# This also lists "pad" as a noise type, which is inaccurate
	print(f"Noise types: {', '.join(map(str, get_degradations(noise)))}")

def prepare(args):
	from preparations import prep_folder
	noise_folder = os.path.join(args.data, args.noise)
	speech_folder = os.path.join(args.data, args.speech)

	if not os.path.exists(noise_folder):
		print(f"Noise folder {noise_folder} does not exist")
		return

	if not os.path.exists(speech_folder):
		print(f"Speech folder {speech_folder} does not exist")
		return

	tot = prep_folder(
		noise_folder,
		recursive=True,
		prompt=args.prompt,
		skip_if_fixed=not args.no_skip,
		to_mono=not args.keep_stereo,
		downsample=not args.no_downsample
	)
	print(f"Prepared {tot} noise files")
	tot = prep_folder(
		speech_folder,
		recursive=True,
		prompt=args.prompt,
		skip_if_fixed=not args.no_skip,
		to_mono=not args.keep_stereo,
		downsample=not args.no_downsample
	)
	print(f"Prepared {tot} speech files")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog="create_dataset.py",
		description="Introduce problems into speech recordings"
	)
	parser.set_defaults(func=lambda a: parser.print_usage())
	subparsers = parser.add_subparsers()

	parser.add_argument("--profile", help="Profile the dataset creator", action="store_true")

	group = parser.add_argument_group("input folders")
	group.add_argument("--data", help="Path to data folder", default="")
	group.add_argument("--speech", help="Speech folder path relative to data folder (default: %(default)s)", default="timit")
	group.add_argument("--noise", help="Noise folder path relative to data folder (default: %(default)s)", default="noise")

	subparser = subparsers.add_parser("create", help="Create a dataset from available speech and noise")
	subparser.set_defaults(func=create)

	group = subparser.add_argument_group("tools")
	group.add_argument("--adt", help="Path to Audio Degradation Toolbox root folder (default: <DATA>/adt)", default=None)
	
	group = subparser.add_argument_group("output folders")
	group.add_argument("--output-train", help="Path to training data output folder (default: <DATA>/train)", default=None)
	group.add_argument("--output-test", help="Path to testing data output folder (default: <DATA>/test)", default=None)
	group.add_argument("--keep-old", help="Keep existing data instead of moving to subfolder (default: false)", action="store_true")

	group = subparser.add_argument_group("degradation parameters")
	group.add_argument("--snr", help="Signal-to-noise ratio in dB (speech is considered signal)", type=float, default=None)
	# TODO: implement?
	# Currently, noise is down- or upsampled to match speech
	# group.add_argument("--downsample-speech", help="Downsample speech signal if noise sample rate is lower", action="store_true")
	group.add_argument("-p", "--pad", help="Pad the speech with PAD seconds of silence at the beginning and end", type=float, default=None)

	subparser.add_argument("-c", "--classes", help="The class types to use in addition to silence (default: all)", metavar="CLASS", nargs="+", default=[])
	subparser.add_argument("--train", help="Ratio/number of files in training set (default: %(default).2f)", default=11/15, type=float)
	subparser.add_argument("--test", help="Ratio/number of files in testing set (default: %(default).2f)", default=4/15, type=float)
	subparser.add_argument("--no-cache", help="Disable caching noise files. Increases runtime but decreases memory usage.", action="store_true")

	subparser = subparsers.add_parser("prepare", help="Prepare the audio files in the dataset (convert from nist to wav, stereo to mono etc.)")
	subparser.set_defaults(func=prepare)
	
	subparser.add_argument("-p", "--prompt", help="List commands and ask for confirmation before performing preparation.", action="store_true")
	subparser.add_argument("--no-skip", help="Process a file even if a processed .wav file with the same name already exists.", action="store_true")
	subparser.add_argument("--keep-stereo", help="Do not convert stereo files to mono (kaldi requires mono for feature extraction)", action="store_true")
	subparser.add_argument("--no-downsample", help="Do not downsample files to 16 kHz (kaldi and the WebRTC VAD require 16kHz audio)", action="store_true")

	subparser = subparsers.add_parser("list", help="List available files and degradations")
	subparser.set_defaults(func=list_files)

	args = parser.parse_args()
	if args.profile:
		import cProfile
		cProfile.run("args.func(args)", sort="cumulative")
	else:
		start = time()
		args.func(args)
		print(f"Total time: {time() - start:.1f} s")