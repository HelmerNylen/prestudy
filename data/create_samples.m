function create_samples(speech_files, degradations, output_files, use_cache, adt_root)
% Description:
%   This is a port of applyDegradation.m and demo_batchProcessing.m for
%   more modern versions of Matlab
% Input:
%   speech_files          N-length string array containing names of files
%                           to apply degradations to. Files must have been
%                           prepared beforehand.
%   degradations          NxM struct array containing the corresponding
%                           degradations. M is the highest number of
%                           degradations to perform on any file. Each
%                           element of degradations should have two fields:
%                             name      a string with the degradation unit
%                                         name. Set to [] or "" for no
%                                         degradation.
%                             params    a struct with the parameters passed
%                                         to the degradation function
%   output_files          N-length string array containing corresponding
%                           output file names
%   use_cache (optional)  whether to memoize calls to audioread when
%                           reading parameters of e.g. addSound
%   adt_root (optional)   path to Audio Degradation Toolbox root folder
%                           (the folder containing applyDegradation.m)
% Output: None
% Author: Helmer Nylen

assert(length(speech_files) == size(degradations, 1) ...
        && length(degradations) == length(output_files), ...
    sprintf("Mismatched input lengths: %d speech files, %d degradations, %d output files", ...
        length(speech_files), size(degradations, 1), length(output_files)) ...
)
[N, M] = size(degradations);

if nargin < 5
    adt_root = "adt/AudioDegradationToolbox";
end
addpath(adt_root + "/degradationUnits");
addpath(adt_root + "/support");
if nargin < 4 || use_cache
    mem_audioread = memoize(@audioread);
    mem_audioread.CacheSize = 50;
else
    mem_audioread = @audioread;
end

% for i=1:N
parfor i = 1:N
    [f_audio, samplerate] = audioread(speech_files(i));
    
    for j = 1:M
        funcname = degradations(i, j).name;
        params = degradations(i, j).params;
        if isa(funcname, 'double')
            continue
        end
        switch funcname
            case "pad"
                if isfield(params, 'before')==0
                    params.before = 0;
                end
                if isfield(params, 'after')==0
                    params.after = 0;
                end
                func = @(aud, fs, ~, par) [zeros(fs * par.before, size(aud, 2)); aud; zeros(fs * par.after, size(aud, 2))];

            case "addSound"
                if isfield(params, 'addSoundFile')
                    if isfield(params, 'addSoundFileStart') ...
                            && isfield(params, 'addSoundFileEnd')
                        [params.addSound, params.addSoundSamplingFreq] = ...
                            mem_audioread(params.addSoundFile, [...
                                params.addSoundFileStart, ...
                                params.addSoundFileEnd ...
                            ]);
                    elseif isfield(params, 'addSoundFileStart') ...
                            && strcmpi(params.addSoundFileStart, "random")
                        [params.addSound, params.addSoundSamplingFreq] = ...
                            mem_audioread(params.addSoundFile);
                        len = size(f_audio, 1);
                        if len < size(params.addSound, 1)
                            offset = randi(max(1, size(params.addSound, 1) - len + 1));
                            params.addSound = params.addSound(offset:offset+len-1, :);
                        elseif len > size(params.addSound, 1)
                            offset = randi(size(params.addSound(1)));
                            params.addSound = circshift(params.addSound, -offset);
                        end
                    else
                        [params.addSound, params.addSoundSamplingFreq] = ...
                            mem_audioread(params.addSoundFile);
                    end
                end
                params.loadInternalSound = 0;
                func = @degradationUnit_addSound;
                
            case "applyImpulseResponse"
                if isfield(params,'impulseResponseFile')==0 ...
                        && isfield(params,'impulseResponse')==0
                    if isfield(params,'internalIR')==0
                        params.internalIR = 'GreatHall1';
                    end
                    names_internalIR = {'GreatHall1','Classroom1','Octagon1','GoogleNexusOneFrontSpeaker','GoogleNexusOneFrontMic','VinylPlayer1960'};
                    indexIR = find(strcmpi(names_internalIR,params.internalIR), 1);
                    if isempty(indexIR)
                        error('Please specify a valid preset')
                    end
                    IRfiles = adt_root + "/degradationData/" + [...
                        "RoomResponses/GreatHall_Omni_x06y06.wav",...
                        "RoomResponses/Classroom_Omni_30x20y.wav",...
                        "RoomResponses/Octagon_Omni_x06y06.wav",...
                        "PhoneResponses/IR_GoogleNexusOneFrontSpeaker.wav",...
                        "PhoneResponses/IR_GoogleNexusOneFrontMic.wav",...
                        "VinylSim/ImpulseReponseVinylPlayer1960_smoothed.wav"...
                    ];
                    params.impulseResponseFile = IRfiles(indexIR);
                end
                if isfield(params,'impulseResponseFile')
                    [params.impulseResponse, params.impulseResponseSampFreq] = ...
                        mem_audioread(file);
                end
                params.loadInternalIR = 0;
                func = @degradationUnit_applyImpulseResponse;
                
            case "adaptiveEqualizer"
                if isfield(params,'computeMagFreqRespFromAudioFile')
                    [params.computeMagFreqRespFromAudio_audioData,...
                     params.computeMagFreqRespFromAudio_sf] = ...
                        mem_audioread(params.computeMagFreqRespFromAudioFile);
                end
                func = @degradationUnit_adaptiveEqualizer;
                
            case "applyMfccMeanAdaption"
                if isfield(params,'audioDataForDestinationMfccFile')
                    [params.audioDataForDestinationMfcc,...
                     params.audioDataForDestMfcc_sf] = ...
                        mem_audioread(params.audioDataForDestinationMfccFile);
                end
                func = @degradationUnit_applyMfccMeanAdaption;
                
            case "applyMp3Compression"
                warning("applyMp3Compression is not supported. Skipping.");
                continue
                
            case "normalize"
                func = @adthelper_normalizeAudio;
                
            case ""
                continue
                
            otherwise
                func = str2func("degradationUnit_" + funcname);
        end
        
        f_audio = func(f_audio, samplerate, [], params);
    end
    
    audiowrite(output_files(i), f_audio, samplerate, 'BitsPerSample', 16);
end

end