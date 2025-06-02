# Speech-to-Image Live Conversion using Deep Learning

This project demonstrates a pipeline that converts **spoken audio input** into **generated images** using state-of-the-art deep learning models. It combines OpenAI's Whisper model for speech-to-text transcription with Stable Diffusion for text-to-image generation, along with sentiment analysis for enhanced text understanding.

---

## Features

- **Audio Recording:** Capture live audio input from microphone.
- **Speech-to-Text:** Transcribe spoken words using a fine-tuned Whisper model.
- **Sentiment Analysis:** Analyze the emotion or sentiment of the transcribed text.
- **Text-to-Image:** Generate an image based on the transcription using Stable Diffusion.
- **GPU Support:** Optimized to use CUDA-enabled GPU if available.

---

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Diffusers
- Librosa
- SoundDevice
- SciPy

Install dependencies with:

```bash
pip install torch transformers diffusers librosa sounddevice scipy

Usage
Set model paths in the script (functions.ipynb or your Python script):

python
Copy
Edit
WHISPER_MODEL_PATH = "path/to/your/whisper_finetuned_model"
SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"  # or your local Stable Diffusion model path
Run your script to:

Record audio from the microphone

Transcribe speech to text

Analyze sentiment (optional)

Generate and display/save images from the transcribed text

Code Overview
record_audio(duration, output_path, fs) — Records audio for a given duration and saves it.

transcribe_audio(audio_path, processor, model) — Transcribes recorded audio to text.

analyze_sentiment(text, sentiment_pipeline) — Performs sentiment analysis on text.

generate_image(text, pipe) — Generates an image from text prompt.

Notes
Make sure your Whisper fine-tuned model directory contains all required files (pytorch_model.bin, config.json, preprocessor_config.json, etc.).

Stable Diffusion can be loaded either from the Hugging Face Hub or a local directory.

GPU is recommended for faster inference but not required.

License
This project is open-source under the MIT License.
