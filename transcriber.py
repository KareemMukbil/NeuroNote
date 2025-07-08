import numpy as np
import torch
import gc
import re
import time

# First try to import OpenAI's Whisper
try:
    import whisper
    from whisper import load_model as whisper_load_model
    WHISPER_TYPE = "openai"
except ImportError:
    whisper = None

# If OpenAI Whisper not found, try to use the legacy 'whisper' package
if whisper is None:
    try:
        import whisper as whisper_legacy
        whisper = whisper_legacy
        WHISPER_TYPE = "legacy"
    except ImportError:
        whisper = None

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
            
            # Check large model support (>=10GB VRAM)
            info['supports_large'] = vram_gb >= 10
            
            # Check FP16 support (not on 16-series)
            info['supports_fp16'] = "16" not in props.name
            
            # Recommend GPU if VRAM >= 4GB
            info['recommend_gpu'] = vram_gb >= 4
        except:
            pass
    
    return info

class WhisperTranscriber:
    def __init__(self, model_size="base.en", use_gpu=True):
        """
        Initialize Whisper transcriber
        
        Args:
            model_size: Whisper model size
            use_gpu: Whether to use GPU acceleration
        """
        if whisper is None:
            raise ImportError("Failed to import Whisper library. Please install it with:\n"
                             "pip install openai-whisper")
        
        self.model_size = model_size
        self.model = None
        
        # Get GPU capabilities
        self.gpu_info = get_gpu_capability()
        
        # Determine compute device
        if use_gpu and torch.cuda.is_available():
            self.device = "cuda"
            self.fp16_enabled = self.gpu_info['supports_fp16']
            
            # Special handling for large models on limited GPUs
            if model_size in ('large', 'large-v2') and not self.gpu_info['supports_large']:
                print("Large model on limited GPU - enabling memory optimizations")
                self.chunk_size = 5  # Process in 5-second chunks
            else:
                self.chunk_size = 30  # Normal processing
        else:
            self.device = "cpu"
            self.fp16_enabled = False
            self.chunk_size = 10  # Smaller chunks for CPU
        
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper model with optimizations"""
        try:
            print(f"Loading Whisper {self.model_size} on {self.device}...")
            start_time = time.time()
            
            # Load model with appropriate method
            if WHISPER_TYPE == "openai":
                self.model = whisper.load_model(
                    self.model_size, 
                    device=self.device
                )
            else:  # Legacy package
                self.model = whisper.load_model(self.model_size, device=self.device)
            
            # Apply optimizations for large models
            if self.model_size in ('large', 'large-v2') and self.device == "cuda":
                # Enable attention slicing for memory reduction
                if hasattr(self.model, 'decoder') and hasattr(self.model.decoder, 'set_attention_slice'):
                    self.model.decoder.set_attention_slice("auto")
                    print("Enabled attention slicing for memory optimization")
                
                # Use FP16 only if supported
                if self.fp16_enabled:
                    self.model.half()
                    print("Using FP16 precision")
                else:
                    print("Using FP32 precision (FP16 not supported)")
            
            load_time = time.time() - start_time
            device_str = "CPU" if self.device == "cpu" else "GPU"
            print(f"Model loaded in {load_time:.2f}s on {device_str}")
        except Exception as e:
            print(f"Failed to load Whisper model: {e}")
            # Provide installation instructions
            if "No module named 'whisper'" in str(e):
                print("\nERROR: Whisper not installed. Please install with:")
                print("pip install openai-whisper")
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
        
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            sample_rate = 16000
            audio_length = len(audio_array) / sample_rate
            
            # Handle long audio with dynamic chunking
            if audio_length > self.chunk_size:
                return self._transcribe_long_audio(audio_array, sample_rate)
            else:
                return self._transcribe_single(audio_array)
        
        except Exception as e:
            print(f"Transcription error: {e}")
            self._cleanup_memory()
            return f"[Error: {str(e)}]"
    
    def _transcribe_single(self, audio_array):
        """Transcribe a single audio chunk"""
        start_time = time.time()
        
        # Handle different Whisper API versions
        if WHISPER_TYPE == "openai":
            result = self.model.transcribe(
                audio_array,
                language="en",
                fp16=self.fp16_enabled,
                temperature=0.0,
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4,
                word_timestamps=False
            )
        else:  # Legacy package
            result = self.model.transcribe(
                audio_array,
                language="en",
                fp16=self.fp16_enabled
            )
        
        transcribe_time = time.time() - start_time
        print(f"Transcribed {len(audio_array)/16000:.2f}s audio in {transcribe_time:.2f}s")
        
        # Clean up GPU memory
        self._cleanup_memory()
        
        return self._clean_transcription(result["text"])
    
    def _transcribe_long_audio(self, audio_array, sample_rate):
        """Transcribe long audio using chunking"""
        chunk_samples = int(self.chunk_size * sample_rate)
        total_samples = len(audio_array)
        results = []
        
        for start in range(0, total_samples, chunk_samples):
            end = min(start + chunk_samples, total_samples)
            chunk = audio_array[start:end]
            
            if len(chunk) < 500:  # Skip very short chunks
                continue
                
            print(f"Processing chunk {start//chunk_samples+1}/{(total_samples+chunk_samples-1)//chunk_samples}")
            chunk_result = self._transcribe_single(chunk)
            results.append(chunk_result)
            
            # Extra memory cleanup between chunks
            self._cleanup_memory()
        
        return " ".join(results)
    
    def _clean_transcription(self, text):
        """Clean up transcription text"""
        # Remove common artifacts
        text = text.replace('...', '').strip()
        text = text.replace('..', '.').strip()
        
        # Remove leading/trailing non-speech characters
        text = re.sub(r'^[^a-zA-Z0-9]+', '', text)
        text = re.sub(r'[^a-zA-Z0-9]+$', '', text)
        
        # Capitalize first letter
        if text:
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
        if self.model:
            del self.model
            self.model = None
        
        self._cleanup_memory()
