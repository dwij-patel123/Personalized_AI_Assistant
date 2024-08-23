import whisper

# use whisper for speech to text function
pross_dir = 'processed_audio/'
def speech_to_text(filename):
    fileStr = filename.split('.')

    model = whisper.load_model("base")
    audio = whisper.load_audio(pross_dir + fileStr[0] + '_processed.mp3')
    # for trimming audio
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")
    options = whisper.DecodingOptions()
    # the text is in the result.text
    result = whisper.decode(model, mel, options)
    return result