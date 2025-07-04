*# Real-Time Voice Transcriber*



*A live voice transcription application using OpenAI's Whisper model with a user-friendly GUI.*



*## Features*



*- Real-time voice transcription*

*- GPU acceleration support*

*- Multiple Whisper model sizes*

*- Adjustable voice activity detection*

*- Audio level monitoring*

*- Clean, responsive GUI*



*## Installation*



*1. \*\*Install Python dependencies:\*\**

*```bash*

*pip install -r requirements.txt*

*```*



*2. \*\*For Windows users (PyAudio):\*\**

*If you get errors installing PyAudio, try:*

*```bash*

*pip install pipwin*

*pipwin install pyaudio*

*```*



*3. \*\*For GPU support:\*\**

*Make sure you have CUDA installed and compatible PyTorch:*

*```bash*

*pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118*

*```*



*## Usage*



*1. \*\*Run the application:\*\**

*```bash*

*python gui.py*

*```*



*2. \*\*Configure settings:\*\**

   *- Select appropriate Whisper model (smaller = faster, larger = more accurate)*

   *- Adjust pause threshold (how long to wait before processing speech)*

   *- Set minimum chunk duration*

   *- Toggle GPU acceleration if available*



*3. \*\*Start recording:\*\**

   *- Click "Start Recording"*

   *- Speak naturally - the app will detect pauses and transcribe speech segments*

   *- Transcription appears in real-time in the text area*



*## Model Recommendations*



*- \*\*tiny.en\*\*: Fastest, lowest accuracy (~32MB)*

*- \*\*base.en\*\*: Good balance of speed/accuracy (~74MB) - \*\*Recommended for most users\*\**

*- \*\*small.en\*\*: Better accuracy, moderate speed (~244MB)*

*- \*\*medium.en\*\*: High accuracy, slower (~769MB)*

*- \*\*large-v2\*\*: Highest accuracy, slowest (~1550MB) - \*\*Requires 8GB+ GPU RAM\*\**



*## Troubleshooting*



*\*\*"No audio device found":\*\**

*- Check microphone permissions*

*- Ensure microphone is not used by other applications*



*\*\*"CUDA out of memory":\*\**

*- Use a smaller model (base.en or small.en)*

*- Reduce chunk size in settings*

*- Close other GPU-intensive applications*



*\*\*Poor transcription quality:\*\**

*- Speak clearly and at moderate pace*

*- Reduce background noise*

*- Adjust silence threshold*

*- Use a better microphone*



*\*\*Application freezes:\*\**

*- Large models may take time to load initially*

*- Check console for error messages*

*- Restart the application*



*## System Requirements*



*- Python 3.8+*

*- 4GB+ RAM (8GB+ recommended for large models)*

*- Microphone access*

*- Optional: CUDA-compatible GPU for acceleration*



*## Notes*



*- First run will download the selected Whisper model*

*- GPU acceleration significantly improves performance*

*- The application processes speech in chunks based on detected pauses*

*- Transcription quality depends on audio quality and model size*

