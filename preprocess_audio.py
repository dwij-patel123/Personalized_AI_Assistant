import librosa
import soundfile as sf
import torch


# preprocessing audio data
# first write the filename in a String or Posix Path
input_dir = 'input_audios/'
pross_dir = 'processed_audio/'

def preprocess_audio_file(fileName,sampling_rate=16000,trim_threshold=20,to_mono=True):
    # using librosa library set sr(sampling rate) as 16kHz
    fileStr = fileName.split('.')
    y, sr = librosa.load(input_dir + fileName, sr=sampling_rate)

    # trim the start and end
    yt, _ = librosa.effects.trim(y, top_db=trim_threshold)
    # convert stereo to mono
    if to_mono:
        yt = librosa.to_mono(yt)

    # Write out processed audio as .mp3 file
    processed_fileName = pross_dir + fileStr[0] + '_processed.mp3'
    sf.write(processed_fileName, yt, sr)
    # Implement VAD -- ADDITIONAL REQUIREMENTS
    # for setting up VAD threshold (it is by default 0.5)
    torch.set_num_threads(1)
    # used silero-vad

    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
    (get_speech_timestamps, _, read_audio, _, _) = utils
    wav = read_audio(processed_fileName)
    speech_timestamps = get_speech_timestamps(wav, model)
    # printing speech timestamps of VAD
    print(speech_timestamps)








