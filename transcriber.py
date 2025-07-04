import whisper
import numpy as np
import torch
import gc
import re
import time
import warnings
import threading

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def get_gpu_capability():
    """Detect GPU capabilities and memory"""
    info = {
        'name': 'No GPU',
        'vram': 0,
        'supports_fp16': False,
        'supports_large': False,
        'recommend_gpu': False
    }
    
    if torch.cuda.is_available():
        try:
            # Get GPU information
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            vram_gb = props.total_memory / (1024 ** 3)
            
            info['name'] = props.name
            info['vram'] = vram_gb
            
            # Check large model support (>=8GB VRAM)
            info['supports_large'] = vram_gb >= 8
            
            # Check FP16 support (avoid on older cards)
            info['supports_fp16'] = props.major >= 7  # Tensor cores
            
            # Recommend GPU if VRAM >= 4GB
            info['recommend_gpu'] = vram_gb >= 4
            
            print(f"GPU Detected: {info['name']} ({vram_gb:.1f}GB VRAM)")
            
        except Exception as e:
            print(f"Error detecting GPU: {e}")
    else:
        print("No CUDA GPU available")
    
    return info

class WhisperTranscriber:
    def __init__(self, model_size="base.en", use_gpu=True):
        """
        Initialize Whisper transcriber
        
        Args:
            model_size: Whisper model size
            use_gpu: Whether to use GPU acceleration
        """
        self.model_size = model_size
        self.model = None
        self.lock = threading.Lock()  # Thread safety
        
        # Get GPU capabilities
        self.gpu_info = get_gpu_capability()
        
        # Determine compute device
        if use_gpu and torch.cuda.is_available():
            self.device = "cuda"
            # IMPORTANT: Disable FP16 for stability - this is the main fix
            self.use_fp16 = False  # Changed from self.gpu_info['supports_fp16']
            
            # Special handling for large models on limited GPUs
            if model_size == 'large-v2' and not self.gpu_info['supports_large']:
                print("Large model on limited GPU - enabling memory optimizations")
                self.use_fp16 = False  # Use FP32 for stability
                self.chunk_size = 5  # Process in 5-second chunks
            else:
                self.chunk_size = 30  # Normal processing
        else:
            self.device = "cpu"
            self.use_fp16 = False
            self.chunk_size = 10  # Smaller chunks for CPU
        
        print(f"Transcriber initialized: {self.device}, FP16: {self.use_fp16}")
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper model with optimizations"""
        try:
            print(f"Loading Whisper {self.model_size} on {self.device}...")
            start_time = time.time()
            
            # Load model
            self.model = whisper.load_model(
                self.model_size, 
                device=self.device
            )
            
            # Apply GPU optimizations
            if self.device == "cuda":
                # Move to GPU if not already there
                if next(self.model.parameters()).device.type != 'cuda':
                    self.model = self.model.cuda()
                
                # Only use half precision if explicitly enabled (now always False)
                if self.use_fp16:
                    self.model = self.model.half()
                    print("Using FP16 precision")
                else:
                    # Ensure model is in FP32
                    self.model = self.model.float()
                    print("Using FP32 precision")
                
                # Optimize for inference
                self.model.eval()
                torch.backends.cudnn.benchmark = True
            
            load_time = time.time() - start_time
            print(f"Model loaded in {load_time:.2f}s")
            
        except Exception as e:
            print(f"Failed to load Whisper model: {e}")
            raise
    
    def transcribe_chunk(self, audio_data):
        """
        Transcribe a single audio chunk
        
        Args:
            audio_data: Raw PCM audio data as bytes
            
        Returns:
            str: Transcribed text
        """
        if not self.model:
            return "[Model not loaded]"
        
        with self.lock:  # Thread safety
            try:
                # Convert bytes to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Ensure minimum length
                if len(audio_array) < 1600:  # Less than 0.1 seconds
                    return ""
                
                # Pad or trim to whisper's expected format
                audio_array = whisper.pad_or_trim(audio_array)
                
                # Handle long audio with dynamic chunking
                audio_length = len(audio_array) / 16000
                if audio_length > self.chunk_size:
                    return self._transcribe_long_audio(audio_array)
                else:
                    return self._transcribe_single(audio_array)
            
            except Exception as e:
                print(f"Transcription error: {e}")
                self._cleanup_memory()
                return f"[Error: {str(e)}]"
    
    def _transcribe_single(self, audio_array):
        """Transcribe a single audio chunk"""
        try:
            start_time = time.time()
            
            # Ensure audio is float32 for compatibility
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            
            # Convert to tensor and ensure proper device/dtype
            if self.device == "cuda":
                # Create tensor on GPU with proper dtype
                audio_tensor = torch.from_numpy(audio_array).float().cuda()
            else:
                audio_tensor = torch.from_numpy(audio_array).float()
            
            # Use the model's encode method for better control
            with torch.no_grad():
                # Get mel spectrogram
                mel = whisper.log_mel_spectrogram(audio_tensor).to(self.model.device)
                
                # Ensure mel spectrogram has correct dtype
                if self.use_fp16:
                    mel = mel.half()
                else:
                    mel = mel.float()
                
                # Detect language and decode
                _, probs = self.model.detect_language(mel)
                
                # Decode options
                options = whisper.DecodingOptions(
                    language="en",
                    temperature=0.0,
                    no_speech_threshold=0.6,
                    without_timestamps=True,
                    suppress_blank=True,
                    suppress_tokens=[-1]
                )
                
                # Decode
                result = whisper.decode(self.model, mel, options)
                
                transcribe_time = time.time() - start_time
                print(f"Transcribed {len(audio_array)/16000:.2f}s audio in {transcribe_time:.2f}s")
                
                # Clean up GPU memory
                if self.device == "cuda":
                    del audio_tensor, mel
                    self._cleanup_memory()
                
                return self._clean_transcription(result.text)
                
        except Exception as e:
            print(f"Single transcription error: {e}")
            self._cleanup_memory()
            
            # Fallback to high-level API
            try:
                print("Falling back to high-level API...")
                result = self.model.transcribe(
                    audio_array,
                    language="en",
                    temperature=0.0,
                    no_speech_threshold=0.6,
                    without_timestamps=True,
                    suppress_blank=True,
                    suppress_tokens=[-1]
                )
                return self._clean_transcription(result["text"])
            except Exception as e2:
                print(f"Fallback transcription error: {e2}")
                return f"[Error: {str(e2)}]"
    
    def _transcribe_long_audio(self, audio_array):
        """Transcribe long audio using chunking"""
        try:
            chunk_samples = int(self.chunk_size * 16000)
            total_samples = len(audio_array)
            results = []
            
            print(f"Processing long audio: {total_samples/16000:.2f}s in {self.chunk_size}s chunks")
            
            for i in range(0, total_samples, chunk_samples):
                chunk_end = min(i + chunk_samples, total_samples)
                chunk = audio_array[i:chunk_end]
                
                if len(chunk) < 1600:  # Skip very short chunks
                    continue
                
                # Pad chunk to standard size
                chunk = whisper.pad_or_trim(chunk)
                
                print(f"Processing chunk {i//chunk_samples+1}/{(total_samples+chunk_samples-1)//chunk_samples}")
                chunk_result = self._transcribe_single(chunk)
                
                if chunk_result and chunk_result.strip():
                    results.append(chunk_result)
                
                # Extra memory cleanup between chunks
                self._cleanup_memory()
            
            return " ".join(results)
            
        except Exception as e:
            print(f"Long audio transcription error: {e}")
            return f"[Error: {str(e)}]"
    
    def _clean_transcription(self, text):
        """Clean up transcription text"""
        if not text:
            return ""
        
        # Remove common artifacts
        text = text.strip()
        text = re.sub(r'\.{2,}', '.', text)  # Replace multiple dots
        text = re.sub(r'\s+', ' ', text)     # Replace multiple spaces
        
        # Remove leading/trailing non-speech characters
        text = re.sub(r'^[^\w\s]+', '', text)
        text = re.sub(r'[^\w\s.!?]+$', '', text)
        
        # Only remove completely empty or single character artifacts
        if len(text.strip()) <= 1:
            return ""
        
        # Capitalize first letter
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        
        return text
    
    def _cleanup_memory(self):
        """Release GPU memory resources"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
    
    def cleanup(self):
        """Release all resources"""
        print("Cleaning up transcriber resources...")
        with self.lock:
            if self.model:
                del self.model
                self.model = None
            
            self._cleanup_memory()
            print("Transcriber cleanup complete")