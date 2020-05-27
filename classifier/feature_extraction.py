import pydub
import webrtcvad
import os
import sys
import subprocess
import kaldi_io
import numpy as np

# Combine features sequences into one big matrix and a list of sequence lengths
def concat_samples(data):
	return np.concatenate(data), [arr.shape[0] for arr in data]

# Split the sequences into a list of individual sequence matrices
def split_samples(data, lengths=None):
	if lengths == None:
		data, lengths = data
	return np.split(data, np.cumsum(lengths[:-1]))

# Get the files in a folder sorted by the type of noise
def type_sorted_files(folder):
	files = [fn for fn in next(os.walk(folder))[2] if fn.lower().endswith(".wav")]
	noise_types = set(os.path.splitext(f)[0].rstrip("0123456789").rstrip("_") for f in files)
	res = dict((t, [f for f in files if f.startswith(t)]) for t in sorted(noise_types))
	if sum(len(res[t]) for t in res) != len(files):
		raise LookupError(f"{sum(len(res[t]) for t in res)} sorted files but {len(files)} unsorted files.\nThe filenames are probably invalid (should be e.g. 'add_hum_012.wav').")
	return res

# Get the MFCCs of each noise type in a folder
def extract_mfcc(folder, overwrite, concatenate=False):
	arks = dict()
	for noise_type, filenames in type_sorted_files(folder).items():
		path = os.path.join(folder, noise_type + "_mfcc.ark")
		if overwrite or not os.path.exists(path):
			# Create <folder>/<noise_type>_mfcc.ark, containing features of all wav files
			print(f"Extracting {noise_type} features in {folder} ...", end="", flush=True)
			result = subprocess.run([
				'compute-mfcc-feats',
				'scp:-',
				'ark:' + path],
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
			print(f"Saved to {path}")
		arks[noise_type] = path
	
	if len(arks) == 0:
		raise ValueError(f"Folder {folder} is not a dataset")
	
	res = dict()
	# Read all arks
	for noise_type, ark in arks.items():
		res[noise_type] = [mat for _, mat in kaldi_io.read_mat_ark(ark)]
		if concatenate:
			res[noise_type] = concat_samples(res[noise_type])
	return res

"""
def load_all_train(train_folder, vad_aggressiveness, frame_length):
	if not os.path.exists(train_folder):
		raise ValueError(f"Training data folder does not exist: {train_folder}")
	if frame_length not in (10, 20, 30):
		raise ValueError("Only frame lengths of 10, 20 or 30 ms are allowed")
	if vad_aggressiveness not in (0, 1, 2, 3):
		raise ValueError("VAD mode must be between 0 and 3")

	vad = webrtcvad.Vad(vad_aggressiveness)

	files = next(os.walk(train_folder, followlinks=True))[2]
	res = [None] * len(files)
	for i, f in enumerate(files):
		path = os.path.join(train_folder, f)
		audio = pydub.AudioSegment.from_wav(path)

		if audio.channels != 1:
			raise ValueError(f"Only mono audio is supported, but file {path} has {audio.channels} channels")
		if audio.sample_width != 2:
			raise ValueError(f"Audio must be 16 bit, but file {path} is {audio.sample_width * 8}-bit")
		if audio.frame_rate not in (8000, 16000, 32000, 48000):
			raise ValueError(f"Audio must be sampled at 8, 16, 32 or 48 kHz, but file {path} is sampled at {audio.frame_rate} Hz.")
		n_bytes_per_frame = audio.sample_width * int(audio.frame_rate * frame_length / 1000)
		data = audio.raw_data
		n_frames = len(data) // n_bytes_per_frame
		activity = np.full(n_frames, False)

		for ptr in range(n_frames):
			frame = data[ptr*n_bytes_per_frame : (ptr+1)*n_bytes_per_frame]
			activity[ptr] = vad.is_speech(frame, audio.frame_rate)

		res[i] = (audio, activity)
	return res

def load_all_test():
	pass
"""