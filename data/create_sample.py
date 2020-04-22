import os
import random
import numpy as np
from array import array
from pydub.exceptions import CouldntDecodeError

class AudioWrapper:
	def __init__(self, samples, sample_rate):
		self.samples = samples
		self.sample_rate = sample_rate

degradationWrapperInstance = None
class DegradationWrapper:
	def __init__(self, matlab_engine):
		global degradationWrapperInstance
		self.matlab_engine = matlab_engine
		degradationWrapperInstance = self

	@staticmethod
	def get(matlab_engine):
		return degradationWrapperInstance or DegradationWrapper(matlab_engine)

	# For python ADT compability
	def apply_normalization(self, audio, maxAmplitude=None):
		return AudioWrapper(self.matlab_engine.adthelper_normalizeAudio(
			audio.samples, audio.sample_rate, [], {"maxAmplitude": maxAmplitude} if maxAmplitude else []
		), audio.sample_rate)
	
	# For python ADT compability
	def apply_resample(self, audio, sample_rate_new):
		return AudioWrapper(self.matlab_engine.resample(
			audio.samples, sample_rate_new, audio.sample_rate
		), sample_rate=sample_rate_new)
	
	def addSound(self, audio, addSound, **kwargs):
		kwargs["addSound"] = addSound.samples
		kwargs["addSoundSamplingFreq"] = addSound.sample_rate
		kwargs["loadInternalSound"] = 0
		return AudioWrapper(self.matlab_engine.degradationUnit_addSound(
			audio.samples, audio.sample_rate, [], kwargs
		), audio.sample_rate)

def _noise_files(data_folder, rel_noise_folder, keep=[".wav"]):
	noise_folder = os.path.join(data_folder, rel_noise_folder)
	walk = os.walk(noise_folder, followlinks=True)
	walk = ((dp, dn, [fn for fn in filenames if os.path.splitext(fn)[-1].lower() in keep])
			for dp, dn, filenames in walk)
	walk = ((os.path.relpath(dirpath, noise_folder), (dirpath, filenames))
			for dirpath, dirnames, filenames in walk
			if dirpath != noise_folder and len(filenames) > 0)
	return walk

def _speech_files(data_folder, rel_speech_folder, keep=[".wav"]):
	speech_folder = os.path.join(data_folder, rel_speech_folder)
	walk = os.walk(speech_folder, followlinks=True)
	walk = ((dp, dn, [fn for fn in filenames if os.path.splitext(fn)[-1].lower() in keep])
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

def get_noise_types(*args) -> list:
	if len(args) == 1:
		noise = args[0]
	else:
		noise = read_noise(args[0], args[1])[0]
	res = ["add_" + k for k in noise.keys()]
	return res

def load_audio(source: str, matlab_engine=None):
	if matlab_engine is not None:
		audio = AudioWrapper(*matlab_engine.audioread(source, nargout=2))
		if audio.samples.size[1] != 1:
			audio.samples = matlab_engine.transpose(audio.samples)[0]
			audio.samples.reshape((len(audio.samples), 1))
		return audio
	else:
		from audio_degradation_toolbox.audio import Audio
		try:
			return Audio(source)
		except CouldntDecodeError as e:
			if "Decoding failed" in e.args[0] and "invalid start code NIST in RIFF header" in e.args[0]:
				print(f"It seems you are trying to open an unprocessed TIMIT audio file: {source}\nHave you prepared the dataset with './create_dataset.py prepare'?")
			raise

def pad(audio, t_start: float=0, t_end: float=0, matlab_engine=None):
	"""Pad the audio with a number of seconds of silence at the start and end"""
	assert t_start >= 0 and t_end >= 0
	n_samples_start = round(t_start * audio.sample_rate)
	n_samples_end = round(t_end * audio.sample_rate)
	samples = audio.samples

	if matlab_engine is not None:
		if n_samples_start > 0:
			samples = matlab_engine.cat(1, matlab_engine.zeros(n_samples_start, 1), samples)
		if n_samples_end > 0:
			samples = matlab_engine.cat(1, samples, matlab_engine.zeros(n_samples_end, 1))
		return AudioWrapper(samples, audio.sample_rate)

	else:
		from audio_degradation_toolbox.audio import Audio
		if n_samples_start > 0:
			samples = array(audio.sound.array_type, [0] * n_samples_start) + samples
		if n_samples_end > 0:
			samples = samples + array(audio.sound.array_type, [0] * n_samples_end)
		return Audio(samples=samples, old_audio=audio)


def apply_noise(audio, noise_type: str, noise_files: dict, params, noise_cache=None, matlab_engine=None):
	if matlab_engine is None:
		from audio_degradation_toolbox.audio import Audio
		import audio_degradation_toolbox.degradations as degradations
	else:
		degradations = DegradationWrapper.get(matlab_engine)

	if noise_type is None:
		return degradations.apply_normalization(audio)

	elif noise_type.startswith("add_") and noise_type[len("add_"):] in noise_files.keys():
		if params.snr is None:
			raise ValueError(f"Please specify SNR to apply additive noise")

		audio = degradations.apply_normalization(audio)
		
		# Get a random file of the specified noise type
		folder, files = noise_files[noise_type[len("add_"):]]
		path = os.path.join(folder, random.choice(files))
		if noise_cache is None or path not in noise_cache:
			noise = load_audio(path, matlab_engine=matlab_engine)
			noise = degradations.apply_normalization(noise)
			if noise_cache is not None:
				noise_cache[path] = noise
		else:
			noise = noise_cache[path]

		# Downsample noise if necessary
		if noise.sample_rate > audio.sample_rate:
			noise = degradations.apply_resample(noise, audio.sample_rate)
		elif noise.sample_rate < audio.sample_rate:
			if params.downsample_speech:
				audio = degradations.apply_resample(audio, noise.sample_rate)
			else:
				raise RuntimeError(f"Noise has a lower sample rate ({noise.sample_rate}) than speech ({audio.sample_rate}).\nBy default only noise is downsampled. Please pass --downsample-speech if you wish to downsample the speech signal.")
		
		if len(noise.samples) > len(audio.samples):
			# If noise is longer than speech, slice a random section of the noise
			offset = random.randint(0, len(noise.samples) - len(audio.samples))
			samples = noise.samples[offset:offset + len(audio.samples)]
			noise = Audio(samples=samples, old_audio=noise) if matlab_engine is None else AudioWrapper(samples, noise.sample_rate)
		elif len(noise.samples) < len(audio.samples):
			# Repeat noise if necessary (done automatically by Matlab ADT)
			if matlab_engine is None:
				noise = degradations._stretch_mix(audio, noise)
			

		# Mix the sounds together
		if matlab_engine is None:
			mix_data = np.frombuffer(
				noise.samples, dtype=noise.sound.array_type
			).astype(np.float64)
			return degradations.apply_normalization(degradations._mix(audio, mix_data, params.snr))
		else:
			# Matlab ADT normalizes result by default
			return degradations.addSound(audio, noise, snrRatio=params.snr)

	else:
		raise ValueError(f"Invalid noise type \"{noise_type}\". Expected one of {', '.join(get_noise_types(noise_files))}")