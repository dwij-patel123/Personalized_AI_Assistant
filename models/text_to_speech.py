import torch
import librosa
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import soundfile as sf

output_dir = 'output_audios/'

def text_to_mp3(text,filename,pitch=1,rate=16000,output_speed=1,output_voice='Male'):
    # load all the pretrained preprocessors
    fileStr = filename.split('.')

    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    inputs = processor(text=text, return_tensors="pt")

    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    # option for output voice Male/Female -- ADDITIONAL REQUIREMENTS
    if output_voice == 'Female':
        # female voice
        speaker_embeddings = torch.tensor(embeddings_dataset[7506]["xvector"]).unsqueeze(0)
    else:
        # male voice
        speaker_embeddings = torch.tensor(embeddings_dataset[5667]["xvector"]).unsqueeze(0)
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    # change the pitch of audio -- ADDITIONAL REQUIREMENTS
    y_pitched = librosa.effects.pitch_shift(speech.numpy(),sr=rate,n_steps=pitch)
    # change the pitch of audio --  ADDITIONAL REQUIREMENTS
    y_pitched_speeded = librosa.effects.time_stretch(y_pitched,rate=output_speed)
    output_filePath = output_dir + fileStr[0] + '_output.mp3'
    sf.write(output_filePath,data=y_pitched_speeded,samplerate=rate)
