import pydub
import webrtcvad
import os
import sys
import subprocess
import kaldi_io
import numpy as np
from tempfile import NamedTemporaryFile
from shutil import copy2

def extract_mfcc(folder, overwrite=True):
	path = os.path.join(folder, "mfcc.ark")
	# Create folder/mfcc.ark, containing features of all wav files
	if overwrite or not os.path.exists(path):
		print(f"Extracting features in {folder} ...", end="", flush=True)
		result = subprocess.run([
			'compute-mfcc-feats',
			'scp:-',
			'ark:' + path],
			input="\n".join(fn + " " + os.path.join(folder, fn)
				for fn in next(os.walk(folder))[2] if fn.lower().endswith(".wav")) + "\n",
			encoding=sys.getdefaultencoding(),
			stderr=subprocess.PIPE
		)
		print(" Done")
		if result.returncode:
			print(result.stderr)
			result.check_returncode()
		else:
			for line in result.stderr.splitlines(False):
				if not line.startswith("LOG") and line != " ".join(result.args + []):
					print(line)
	
	return kaldi_io.read_mat_ark(path)


def process(audio, activity):
	"""with NamedTemporaryFile() as tmp:
		subprocess.run([
			os.path.join(kaldi_folder, "tools", "sph2pipe_v2.5", "sph2pipe"),
			"-f", "wav",
			input_file,
			tmp.name], check=True)
		copy2(tmp.name, input_file)
	
	"""
	return (audio, activity)

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

		res[i] = process(audio, activity)
	return res
		

def load_all_test():
	pass