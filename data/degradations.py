import os
from matlab.engine import MatlabEngine
from random import choice

additive_noise_prefix = "additive_"

def get_degradations(noise_files: dict) -> list:
	res = [None, "pad", "white"]
	res.extend(additive_noise_prefix + k for k in noise_files.keys())
	return res

def setup_matlab_degradations(noise_files: dict, speech_files: list, degradations_by_file: list,
		m_eng: MatlabEngine, degradation_parameters, workspace_var: str="degs") -> str:
	"""Setup the structure in Matlab needed to perform the specified degradations
	Returns: the Matlab workspace variable in which the structure is stored"""

	if len(speech_files) != len(degradations_by_file):
		raise ValueError(f"Different number of speech files ({len(speech_files)}) and degradations ({len(degradations_by_file)})")

	allowed_degradations = get_degradations(noise_files)
	matlab_degradations_by_file = []
	max_num_degradations = 0
	null_degradation = {"name": "", "params": ""}

	normalize_audio = not hasattr(degradation_parameters, "normalizeOutputAudio") or degradation_parameters.normalizeOutputAudio
	
	for degradations in degradations_by_file:
		if isinstance(degradations, str) or not hasattr(degradations, '__iter__'):
			degradations = degradations,

		matlab_degradations_by_file.append([])
		has_normalized = False
		for degradation in degradations:
			if degradation is None:
				if normalize_audio and not has_normalized:
					name = "normalize"
					params = ""
					has_normalized = True

					matlab_degradations_by_file[-1].append({"name": name, "params": params})
			
			elif degradation.startswith(additive_noise_prefix) and degradation in allowed_degradations:
				if degradation_parameters.snr is None:
					raise ValueError("Please specify SNR to apply additive noise")
				
				name = "addSound"

				folder, files = noise_files[degradation[len(additive_noise_prefix):]]
				path = os.path.join(folder, choice(files))
				params = {
					"addSoundFile": path,
					"addSoundFileStart": "random",
					"snrRatio": degradation_parameters.snr,
					"normalizeOutputAudio": normalize_audio
				}
				has_normalized = True

				matlab_degradations_by_file[-1].append({"name": name, "params": params})

			elif degradation == "pad":
				if degradation_parameters.snr is None:
					raise ValueError("Please specify padding amount to apply padding")

				name = "pad"
				params = {"before": degradation_parameters.pad, "after": degradation_parameters.pad}

				matlab_degradations_by_file[-1].append({"name": name, "params": params})

			elif degradation == "white":
				name = "addNoise"
				params = {
					"noiseColor": "white",
					"snrRatio": degradation_parameters.snr,
					"normalizeOutputAudio": normalize_audio
				}
				has_normalized = True

				matlab_degradations_by_file[-1].append({"name": name, "params": params})

			else:
				raise ValueError(f'Unrecognized degradation "{degradation}"')

		max_num_degradations = max(max_num_degradations, len(matlab_degradations_by_file[-1]))
	
	for degradations in matlab_degradations_by_file:
		if len(degradations) < max_num_degradations:
			degradations.extend([null_degradation] * (max_num_degradations - len(degradations)))

	# Create a cell array of {name: "", params: {}} structs
	m_eng.workspace[workspace_var] = matlab_degradations_by_file
	# Cast it to a struct array
	m_eng.eval(workspace_var + " = cell2mat(cellfun(@cell2mat, " + workspace_var + ", 'UniformOutput', false)');", nargout=0)
	
	return workspace_var

if __name__ == "__main__":
	print("Please use create_dataset.py")