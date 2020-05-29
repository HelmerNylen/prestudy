% View what noise files were used in training and testing of a dataset

dataset_name = input("Please provide a dataset name: ", 's');
deg_files = fullfile(["train", "test"], dataset_name, "degradations.mat");

if exist(deg_files(1), 'file')
    load(deg_files(1));
    with_recorded_noise = degradations(arrayfun(@(s) isstruct(s.params) && isfield(s.params, 'addSoundFile'), degradations));
    noise_files_train = arrayfun(@(s) string(s.params.addSoundFile), with_recorded_noise);
    disp("Noise files in training set");
    disp(unique(noise_files_train));
else
    disp("No degradations.mat file in the specified training set");
    noise_files_train = string([]);
end

if exist(deg_files(2), 'file')
    load(deg_files(2));
    with_recorded_noise = degradations(arrayfun(@(s) isstruct(s.params) && isfield(s.params, 'addSoundFile'), degradations));
    noise_files_test = arrayfun(@(s) string(s.params.addSoundFile), with_recorded_noise);
    disp("Noise files in testing set");
    disp(unique(noise_files_test));
else
    disp("No degradations.mat file in the specified testing set");
    noise_files_test = string([]);
end

all_noise_files = string([]);
folders = dir("noise");
for i = 1:length(folders)
    folder = folders(i);
    if folder.isdir && ~strcmp(folder.name(1), '.')
        files = dir(fullfile("noise", folder.name));
        for j = 1:length(files)
            file = files(j);
            if ~file.isdir && length(file.name) > 4 && strcmp(file.name(end-4+1:end), '.wav')
                all_noise_files = [all_noise_files; string(fullfile("noise", folder.name, file.name))];
            end
        end
    end
end
in_neither = all_noise_files(arrayfun(@(f) ~ismember(f, noise_files_test) && ~ismember(f, noise_files_train), all_noise_files));
if ~isempty(in_neither)
    disp("Unused noise files");
    disp(in_neither);
end