# import torch
# import librosa
# import soundfile as sf
# # whisper for speech to text
# import whisper
# import os
# from dotenv import load_dotenv
# # imports for predicting input
# from langchain_huggingface import HuggingFaceEndpoint
# from langchain import PromptTemplate
# # llm for converting text to speech
# from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
# from datasets import load_dataset
# import soundfile as sf
#
#
# load_dotenv()
#
#
# # setup test directory and preprocess directory
# # all the input audios should be saved in test_audios folder
# input_dir = 'input_audios/'
# pross_dir = 'processed_audio/'
# output_dir = 'output_audios/'
# filename = 'what-are-three-words-that-describe-you-201475.mp3'
# fileStr = filename.split('.')
#
# # prerpocess Audio ans save it in 'processed_audio' directory
# def prerpocess_audio_file(fileName,sampling_rate=16000,trim_threshold=20,to_mono=True):
#     # using librosa library set sr(sampling rate) as 16kHz
#     fileStr = fileName.split('.')
#     y, sr = librosa.load(input_dir + fileName, sr=sampling_rate)
#
#     # trim the start and end
#     yt, _ = librosa.effects.trim(y, top_db=trim_threshold)
#     # convert stereo to mono
#     if to_mono:
#         yt = librosa.to_mono(yt)
#
#     # Write out processed audio as .mp3 file
#     processed_fileName = pross_dir + fileStr[0] + '_processed.mp3'
#     sf.write(processed_fileName, yt, sr, subtype='PCM_24')
#     # for setting up VAD threshold (it is by default 0.5)
#     torch.set_num_threads(1)
#     # used silero-vad
#     model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
#     (get_speech_timestamps, _, read_audio, _, _) = utils
#     wav = read_audio(processed_fileName)
#     speech_timestamps = get_speech_timestamps(wav, model)
#     # printing speech timestamps of VAD
#     print(speech_timestamps)
#
#
# # use whisper for speech to text function
# def speech_to_text(filename):
#     model = whisper.load_model("base")
#     audio = whisper.load_audio(pross_dir + filename + '_processed.mp3')
#     # for trimming audio
#     audio = whisper.pad_or_trim(audio)
#     mel = whisper.log_mel_spectrogram(audio).to(model.device)
#     # detect the spoken language
#     _, probs = model.detect_language(mel)
#     print(f"Detected language: {max(probs, key=probs.get)}")
#     options = whisper.DecodingOptions()
#     # the text is in the result.text
#     result = whisper.decode(model, mel, options)
#     return result
#
# # after getting text from speech do llm stuff
# def predict_input_text(input_text):
#     HUGGINFACE_API_TOKEN = os.getenv('HUGGINFACE_API_TOKEN')
#     repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
#
#     llm = HuggingFaceEndpoint(repo_id=repo_id, huggingfacehub_api_token=HUGGINFACE_API_TOKEN)
#     template = """You are a smart AI assistant can you please assist me with a concise answer to the following question in exactly two sentences:{question}"""
#     prompt = PromptTemplate(template=template, input_variables=["question"])
#     llm_chain = prompt | llm
#     pred_ans = llm_chain.invoke(input_text)
#     return pred_ans
#
#
# # -- TODO --
# # sampling rate is in model-config
# # try using fastspeed2 (Ony Female Voice) for reducing latency
# # use edge-tts for changing male/female voice - additional requirements
#
#
# def text_to_wav(text,filename,pitch=1,rate=16000,output_voice='Male'):
#     # load all the pretrained preprocessors
#     processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
#     model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
#     vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
#
#     inputs = processor(text=text, return_tensors="pt")
#
#     # load xvector containing speaker's voice characteristics from a dataset
#     embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
#     if output_voice == 'Female':
#         # female voice
#         speaker_embeddings = torch.tensor(embeddings_dataset[7180]["xvector"]).unsqueeze(0)
#     else:
#         # male voice
#         speaker_embeddings = torch.tensor(embeddings_dataset[5667]["xvector"]).unsqueeze(0)
#     speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
#     y_pitched = librosa.effects.pitch_shift(speech.numpy(),sr=rate,n_steps=pitch)
#     output_filePath = output_dir + filename
#     sf.write(output_filePath,data=y_pitched,samplerate=rate)
#
#
# text_to_wav(pred_ans,'test_output.mp3')
#
#


# -- TODO -- Additional Requirements
# parameters = filepath , temperature , sampling rate , speed of output ans , pitch of output ans, voice of speaker

import argparse

# create config parser
config_parser = parser = argparse.ArgumentParser(description="Parameters config file")
parser.add_argument("-c", "--config", default="configs.default_config", type=str, help="config file path (default: configs.default_config)")

# add a regular parser
parser = argparse.ArgumentParser(description="Train a FoodVision model.")

# create tunable parameters parser
group = parser.add_argument_group("Tunable Parameters")
group.add_argument(
    "--filename",
    help="Enter filename")

group.add_argument(
    "--temperature",
    type=float,
    help="Enter Temperature for LLM")

group.add_argument(
    "--sampling_rate",
    type=int,
    help="Enter Sampling Rate of Audio"
)

group.add_argument(
    "--output_speed",
    type=float,
    help="define speed for output audio"
)

group.add_argument(
    "--output_pitch",
    type=float,
    help="define output pitch of audio"
)

group.add_argument(
    "--output_voice",
    type=str,
    help="define voice of output speaker"
)

def _parse_args():
    """Parses command line arguments.
    to parse this enter filename of config file in terminal using -c
    """

    config_args, remaining = config_parser.parse_known_args()

    # for loading the default config file
    if config_args.config:
        from importlib import import_module

        # getattr() function for getting all values of config file
        config_module = getattr(import_module(config_args.config), "config")
        parser.set_defaults(**config_module.__dict__)

    # add remaining parsed input
    args = parser.parse_args(remaining)
    return args


config_args = _parse_args()



# write clean code here
from preprocess_audio import preprocess_audio_file
from models.speech_to_text import speech_to_text
from models.pred_text_model import predict_input_text
from models.text_to_speech import text_to_mp3

# only supports sampling rate (default rate 16000) 8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, and 48000

# preprocess audio function
preprocess_audio_file(config_args.filename,config_args.sampling_rate)

# speech to text
result = speech_to_text(config_args.filename)

# predict answer using llm
pred_ans = predict_input_text(result.text,config_args.temperature)

print(pred_ans)
# convert predicted text to audio and store it in output_audios directory
text_to_mp3(pred_ans,config_args.filename,config_args.output_pitch,
            config_args.sampling_rate,config_args.output_speed,config_args.output_voice)

