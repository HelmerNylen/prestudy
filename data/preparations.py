import os
import sys
import subprocess
from tempfile import NamedTemporaryFile
from shutil import copy2

# Get the file formats supported by SoX
def audio_file_formats():
	output = subprocess.run(
		["sox", "-h"],
		check=True,
		stdout=subprocess.PIPE,
		encoding=sys.getdefaultencoding()
	).stdout
	line = next(l for l in output.splitlines(False) if l.upper().startswith("AUDIO FILE FORMATS:"))
	return ["." + f.lower() for f in line[line.index(":") + 1:].split()]

# Get information about an audio file in a dict
def get_info(filename):
	info = subprocess.run([
		'sox',
		'--i',
		filename],
		stdout=subprocess.PIPE,
		check=True,
		encoding=sys.getdefaultencoding()
	).stdout
	info = (line.split(":") for line in info.splitlines() if len(line.strip()) > 0)
	info = dict((splitline[0].strip(), ":".join(splitline[1:]).strip()) for splitline in info)
	return info

# Output the needed sox options to make the file usable
def _build_command(info, to_mono, downsample):
	command = []
	# We need mono files ...
	if to_mono and not info["Channels"] == "1":
		command.append(("-c", "1"))
	# .. with 16-bit precision ...
	if info["Precision"].rstrip("-bit") != "16":
		command.append(("-b", "16"))
	# ... at 16 kHz sample rate (we don't upsample, though)...
	if downsample and int(info["Sample Rate"]) > 16000:
		command.append(("-r", "16000"))
	# ... which aren't sphere files (which is what TIMIT contains)
	if info["Input File"].endswith(" (sph)"):
		command.append(None)
	return command

# Convert files to kaldi-processable .wav files
# Existing .wav files are overwritten
def prep_folder(folder, recursive=False, prompt=False, skip_if_fixed=True, to_mono=True, downsample=True, audio_exts=None):
	if audio_exts is None:
		audio_exts = audio_file_formats()

	if recursive:
		files = [os.path.join(d, f) for d, _, fs in os.walk(folder, followlinks=True) for f in fs]
	else:
		files = [os.path.join(folder, f) for f in next(os.walk(folder))[2]]
	commands = []

	with NamedTemporaryFile() as tmp:
		# Iterate over all audio files found
		for f in files:
			path, ext = os.path.splitext(f)
			if ext.lower() not in audio_exts:
				continue
			info = get_info(path + ext)
			# Build a command which makes the file usable by the classifiers
			command = _build_command(info, to_mono, downsample)

			if ext != ".wav":
				if skip_if_fixed and os.path.exists(path + ".wav") \
						and len(_build_command(get_info(path + ".wav"), to_mono, downsample)) == 0:
					# Already has a processed .wav version
					continue
				else:
					command.append(None)

			if len(command) > 0:
				command.append(("-t", "wav"))
				commands.append(['sox', path + ext] + [x for c in command if c for x in c] + [tmp.name])
				commands.append((f"Copy {tmp.name} to {path + '.wav'}", lambda x, y: copy2(x, y, follow_symlinks=True), tmp.name, path + ".wav"))

		# Let the user confirm that we're about to make irreversible changes to their precious data
		if prompt and len(commands) > 0:
			print("Suggesting the following operations:")
			if len(commands) // 2 <= 20 \
					or input(f"Modify {len(commands) // 2} files\nView all commands? (y/n): ")\
					.lower().lstrip().startswith("y"):
				for c in commands:
					if type(c) is tuple:
						print(c[0])
					else:
						print(" ".join(c))
			if not input("Continue? (y/n): ").lower().lstrip().startswith("y"):
				print("Discarding suggestions")
				return
		
		# Run all commands
		for c in commands:
			if type(c) is tuple:
				c[1](*c[2:])
			else:
				result = subprocess.run(c, check=True, stderr=subprocess.PIPE, encoding=sys.getdefaultencoding())
				if len(result.stderr) > 0:
					print("stderr from", " ".join(result.args))
					print(result.stderr)

		return len(commands) // 2

if __name__ == "__main__":
	print("Please run me via create_dataset.py prepare")