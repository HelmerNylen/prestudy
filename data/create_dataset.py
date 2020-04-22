#!/usr/bin/env python3
import os
import subprocess
import argparse
import math
from time import time

def _output_dirs(train: str, test: str, move: bool):
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
	ind = startind
	path = os.path.join(folder, label + str(ind).rjust(pad, "0") + "." + ext)
	while os.path.exists(path):
		ind += 1
		path = os.path.join(folder, label + str(ind).rjust(pad, "0") + "." + ext)
	return path

def create(args):
	# Matlab must be imported before random (and thus any module which depends on random)
	if args.use_matlab:
		import matlab.engine
	import random
	from create_sample import (read_noise,
		read_speech,
		get_noise_types,
		apply_noise,
		load_audio,
		pad)

	# Find all source sound files
	noise, n_noisefiles = read_noise(args.data, args.noise)
	speech, n_speechfiles = read_speech(args.data, args.speech)
	
	if n_noisefiles == 0:
		print("No noise files found")
		return
	if n_speechfiles == 0:
		print("No speech files found")
		return

	# Parse and verify noise classes
	if len(args.classes) == 0 or "all" in args.classes:
		args.classes = list(filter(lambda c: c != "all", set(args.classes + get_noise_types(noise))))
	for c in args.classes:
		if c not in get_noise_types(noise):
			print(f"Unknown noise class \"{c}\"")
			print("Available:", ", ".join(noise.keys()))
			return
	args.classes.append(None) # None represents no noise/silence
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
		return
	print(f"{args.train} training files and {args.test} testing files")

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

	# Assign noise types to speech files
	noise_train = list(range(n_classes)) * (len(speech_train) // n_classes)
	noise_train = noise_train + ([n_classes - 1] * (len(speech_train) - len(noise_train)))
	random.shuffle(noise_train)
	noise_test = list(range(n_classes)) * (len(speech_test) // n_classes)
	noise_test = noise_test + ([n_classes - 1] * (len(speech_test) - len(noise_test)))
	random.shuffle(noise_test)

	# Optionally, start matlab
	matlab_engine = None
	if args.use_matlab:
		print("Starting Matlab ...", end="", flush=True)
		matlab_engine = matlab.engine.start_matlab()
		if len(args.data) > 0:
			matlab_engine.cd(args.data)
		if args.adt is None:
			args.adt = os.path.join(args.data, "matlab-audio-degradation-toolbox")
		matlab_engine.addpath(args.adt)
		matlab_engine.addpath(os.path.join(args.adt, "AudioDegradationToolbox"))
		matlab_engine.addpath(os.path.join(args.adt, "AudioDegradationToolbox", "degradationUnits"))
		print(" Done")

	# Create datasets
	noise_cache = None if args.no_cache else dict()

	for t, speech_t, noise_t, output_t in [
		["train", speech_train, noise_train, args.output_train],
		["test", speech_test, noise_test, args.output_test]
	]:
		print(f"Creating {t}ing data ", end="", flush=True)
		for i in range(len(speech_t)):
			# Load speech file
			audio = load_audio(speech_t[i], matlab_engine)
			# Pad speech with silence
			if args.pad is not None:
				audio = pad(audio, args.pad, args.pad, matlab_engine)
			# Apply noise to speech
			audio = apply_noise(
				audio,
				args.classes[noise_t[i]],
				noise,
				args,
				noise_cache,
				matlab_engine
			)
			# Save result to output folder
			filename = _get_available_filename(
				output_t,
				(args.classes[noise_t[i]] or "none") + "_",
				"wav",
				1,
				math.ceil(math.log10(len(speech_t) / n_classes))
			)
			if args.use_matlab:
				matlab_engine.audiowrite(os.path.abspath(filename), audio.samples, audio.sample_rate, nargout=0)
			else:
				audio.export(filename)

			if i % (len(speech_t) // 10 or 1) == 0:
				print(".", end="", flush=True)
		print(" Done", flush=True)
	print("Dataset created")

def list_files(args):
	from create_sample import (read_noise,
		read_speech,
		get_noise_types)

	# Find all source sound files
	noise, n_noisefiles = read_noise(args.data, args.noise)
	speech, n_speechfiles = read_speech(args.data, args.speech)
	
	print(f"Found {n_noisefiles} noise files")
	for noisetype in noise:
		print(f"\t{len(noise[noisetype][1])} of type \"{noisetype}\"")
	print(f"Found {n_speechfiles} speech files")
	for t in ["test", "train"]:
		print(f"\t{sum(len(fs) for d, fs in speech if t in d.lower())} in set \"{t}\"")
	print(f"Noise types: {', '.join(get_noise_types(noise))}")

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
	group.add_argument("--matlab", help="Path to Matlab root folder (default: %(default)s)", default=os.path.join("..", "matlab"))
	group.add_argument("--adt", help="Path to Matlab Audio Degradation Toolbox root folder (default: <DATA>/matlab-audio-degradation-toolbox)", default=None)
	group.add_argument("-m", "--use-matlab", help="Use the original ADT implemented in Matlab", action="store_true")
	
	group = subparser.add_argument_group("output folders")
	group.add_argument("--output-train", help="Path to training data output folder (default: <DATA>/train)", default=None)
	group.add_argument("--output-test", help="Path to testing data output folder (default: <DATA>/test)", default=None)
	group.add_argument("--keep-old", help="Keep existing data instead of moving to subfolder (default: false)", action="store_true")

	group = subparser.add_argument_group("degradation parameters")
	group.add_argument("--snr", help="Signal-to-noise ratio in dB (speech is considered signal)", type=float, default=None)
	group.add_argument("--downsample-speech", help="Downsample speech signal if noise sample rate is lower", action="store_true")
	group.add_argument("-p", "--pad", help="Pad the speech with PAD seconds of silence at the beginning and end", type=float, default=None)

	subparser.add_argument("-c", "--classes", help="The class types to use in addition to silence (default: all)", nargs="+", default=[])
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