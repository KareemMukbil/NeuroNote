import pyaudio
import numpy as np
import threading
import time
import queue
from collections import deque
import wave
import tempfile
import os

class AudioRecorder:
    def __init__(self, sample_rate=16000, chunk_size=1024, channels=1):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format = pyaudio.paInt16
        
        # Recording state
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.processed_chunks = queue.Queue()
        
        # Audio processing parameters
        self.pause_threshold = 1.5  # seconds of silence to trigger chunk
        self.min_chunk_duration = 1.0  # minimum chunk length in seconds
        self.silence_threshold = 300  # RMS threshold for silence detection
        
        # Buffers and timing
        self.audio_buffer = deque()
        self.silence_duration = 0
        self.current_chunk_duration = 0
        self.last_audio_time = 0
        
        # Audio level tracking
        self.current_level = 0
        self.level_history = deque(maxlen=10)  # Smooth level changes
        
        # PyAudio instance
        self.pa = None
        self.stream = None
        
        # Threading
        self.record_thread = None
        self.process_thread = None
        self.running = False
        
        # Voice activity detection
        self.noise_level = 0
        self.voice_detected = False
        
    def start_recording(self):
        """Start the audio recording process"""
        if self.is_recording:
            return
            
        try:
            # Initialize PyAudio
            self.pa = pyaudio.PyAudio()
            
            # Find default input device
            default_device = self.pa.get_default_input_device_info()
            print(f"Using audio device: {default_device['name']}")
            
            # Open audio stream
            self.stream = self.pa.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback,
                input_device_index=default_device['index']
            )
            
            # Reset state
            self.audio_buffer.clear()
            self.silence_duration = 0
            self.current_chunk_duration = 0
            self.last_audio_time = time.time()
            self.noise_level = 0
            self.voice_detected = False
            self.level_history.clear()
            
            # Start recording
            self.is_recording = True
            self.running = True
            self.stream.start_stream()
            
            # Start processing thread
            self.process_thread = threading.Thread(target=self._process_audio, daemon=True)
            self.process_thread.start()
            
            print("Recording started")
            
        except Exception as e:
            print(f"Failed to start recording: {e}")
            self.cleanup()
            raise
    
    def stop_recording(self):
        """Stop the audio recording process"""
        if not self.is_recording:
            return
            
        self.is_recording = False
        self.running = False
        
        # Finalize any remaining audio in buffer
        if len(self.audio_buffer) > 0:
            self._finalize_current_chunk()
        
        print("Recording stopped")
    
    def cleanup(self):
        """Clean up audio resources"""
        self.running = False
        self.is_recording = False
        
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
            self.stream = None
            
        if self.pa:
            try:
                self.pa.terminate()
            except:
                pass
            self.pa = None
            
        # Clear queues
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except:
                break
                
        while not self.processed_chunks.empty():
            try:
                self.processed_chunks.get_nowait()
            except:
                break
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for incoming audio data"""
        if status:
            print(f"Audio callback status: {status}")
        
        if self.is_recording and self.running:
            try:
                self.audio_queue.put(in_data, block=False)
            except queue.Full:
                print("Audio queue full, dropping frame")
                
        return (None, pyaudio.paContinue)
    
    def _process_audio(self):
        """Main audio processing loop"""
        print("Audio processing thread started")
        
        while self.running:
            try:
                # Get audio data with timeout
                try:
                    audio_data = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                self._process_audio_chunk(audio_data)
                
            except Exception as e:
                print(f"Audio processing error: {e}")
                time.sleep(0.1)
        
        print("Audio processing thread stopped")
    
    def _process_audio_chunk(self, audio_data):
        """Process a single audio chunk"""
        try:
            # Convert to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Calculate RMS (audio level)
            rms = np.sqrt(np.mean(audio_array.astype(np.float32)**2))
            
            # Update noise level (adaptive threshold)
            if len(self.audio_buffer) == 0:  # First chunk or new recording
                self.noise_level = max(self.noise_level * 0.95, rms * 0.1)
            
            # Determine if this is speech or silence
            is_speech = rms > max(self.silence_threshold, self.noise_level * 3)
            
            # Update audio level for UI (smoothed)
            self.level_history.append(min(rms / 2000.0, 1.0))
            self.current_level = sum(self.level_history) / len(self.level_history)
            
            # Add to buffer
            self.audio_buffer.append(audio_data)
            
            # Update timing
            chunk_duration = len(audio_array) / self.sample_rate
            self.current_chunk_duration += chunk_duration
            
            # Voice activity detection
            if is_speech:
                self.voice_detected = True
                self.silence_duration = 0
                self.last_audio_time = time.time()
            else:
                self.silence_duration += chunk_duration
            
            # Decide whether to finalize current chunk
            should_finalize = False
            
            # Only finalize if we detected voice activity first
            if self.voice_detected:
                # Finalize if we have enough silence after speech
                if self.silence_duration >= self.pause_threshold:
                    if self.current_chunk_duration >= self.min_chunk_duration:
                        should_finalize = True
                
                # Finalize if chunk is getting too long (prevent memory issues)
                elif self.current_chunk_duration >= 30.0:  # Max 30 seconds
                    should_finalize = True
            
            # Don't accumulate too much silence without speech
            elif self.current_chunk_duration > 5.0:
                self._reset_buffer()
            
            if should_finalize:
                self._finalize_current_chunk()
                
        except Exception as e:
            print(f"Error processing audio chunk: {e}")
    
    def _reset_buffer(self):
        """Reset audio buffer without finalizing"""
        self.audio_buffer.clear()
        self.current_chunk_duration = 0
        self.silence_duration = 0
        self.voice_detected = False
    
    def _finalize_current_chunk(self):
        """Finalize the current audio chunk for transcription"""
        if len(self.audio_buffer) == 0:
            return
        
        try:
            # Combine all audio data
            combined_audio = b''.join(self.audio_buffer)
            
            # Convert to numpy array for processing
            audio_array = np.frombuffer(combined_audio, dtype=np.int16)
            
            # Apply audio processing
            processed_audio = self._process_audio_for_transcription(audio_array)
            
            # Only send if we have meaningful audio
            if len(processed_audio) > 1600:  # At least 0.1 seconds
                pcm_data = processed_audio.tobytes()
                self.processed_chunks.put(pcm_data)
                
                duration = len(processed_audio) / self.sample_rate
                print(f"Chunk finalized: {duration:.1f}s, RMS: {np.sqrt(np.mean(processed_audio.astype(np.float32)**2)):.1f}")
        
        except Exception as e:
            print(f"Error finalizing chunk: {e}")
        
        finally:
            # Reset for next chunk
            self._reset_buffer()
    
    def _process_audio_for_transcription(self, audio_array):
        """Process audio for better transcription quality"""
        try:
            # Convert to float for processing
            audio_float = audio_array.astype(np.float32)
            
            # Normalize audio
            max_val = np.max(np.abs(audio_float))
            if max_val > 0:
                audio_float = audio_float / max_val * 16000
            
            # Simple noise reduction - high-pass filter
            if len(audio_float) > 1:
                # Calculate difference (high-pass effect)
                diff = np.diff(audio_float)
                # Combine original with high-pass
                filtered = np.concatenate(([audio_float[0]], diff * 0.3 + audio_float[1:] * 0.7))
                audio_float = filtered
            
            # Remove DC offset
            audio_float = audio_float - np.mean(audio_float)
            
            # Apply gentle compression to even out volume
            compressed = np.sign(audio_float) * np.power(np.abs(audio_float) / 32768.0, 0.7) * 32768.0
            
            # Convert back to int16
            audio_processed = np.clip(compressed, -32768, 32767).astype(np.int16)
            
            return audio_processed
            
        except Exception as e:
            print(f"Error processing audio: {e}")
            return audio_array  # Return original if processing fails
    
    def get_audio_chunk(self):
        """Get the next processed audio chunk"""
        try:
            return self.processed_chunks.get(timeout=0.5)
        except queue.Empty:
            return None
    
    def get_audio_level(self):
        """Get current audio level (0.0 to 1.0)"""
        return self.current_level
    
    def has_audio_chunks(self):
        """Check if there are audio chunks available"""
        return not self.processed_chunks.empty()
    
    def get_recording_stats(self):
        """Get recording statistics"""
        return {
            'is_recording': self.is_recording,
            'chunk_duration': self.current_chunk_duration,
            'silence_duration': self.silence_duration,
            'voice_detected': self.voice_detected,
            'audio_level': self.current_level,
            'noise_level': self.noise_level,
            'buffer_size': len(self.audio_buffer)
        }