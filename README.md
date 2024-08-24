# Speakly - A Speech-to-Speech AI Assistant

## Project Structure
This repository consists of implementation of Speech-to-Speech Multimodel Pipeline with constructive parts
1. **Audio Preprocessing**
2. **Voice Activity Detection**
3. **Speech to Text** using **OpenAI-Whisper**
4. **Language Model** using **Mistral AI**
5. **Text to Speech** using **Microsoft SpeechT5**

## Project Setup
Clone This Repository:

``` git clone https://github.com/dwij-patel123/Personalized_AI_Assistant.git ```

Install The Requirements:

```pip install -r requirements.txt```

Enter API key in .env file:

``` 
touch .env
```
Inside .env write the Huggingface API Key:

```
HUGGINFACE_API_TOKEN = YOUR_API_KEY
```


## File Structure
1. User should add input audio in the `input_audios` folder from the Directory
2. The output answer will be provided in `output_audios` folder
3. `processed_audio` folder contains the audios after preprocessing