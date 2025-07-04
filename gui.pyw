import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import queue
from recorder import AudioRecorder
from transcriber import WhisperTranscriber, get_gpu_capability
import sys
import time

class ModernStyle:
    """Modern dark red/gold color scheme with enhanced styling"""
    
    # Color palette
    COLORS = {
        'bg_primary': '#1a0f0f',      # Very dark red-brown
        'bg_secondary': '#2d1b1b',    # Dark red-brown
        'bg_tertiary': '#3d2424',     # Medium red-brown
        'bg_card': '#342020',         # Card background
        'accent_gold': '#d4af37',     # Gold
        'accent_gold_dark': '#b8941f', # Darker gold
        'accent_gold_light': '#f4d03f', # Lighter gold
        'accent_red': '#8b0000',      # Dark red
        'accent_red_light': '#a00000', # Light red
        'text_primary': '#f5f5f5',    # Light text
        'text_secondary': '#c0c0c0',  # Secondary text
        'text_muted': '#999999',      # Muted text
        'text_accent': '#d4af37',     # Gold text
        'success': '#2ecc71',         # Modern green
        'warning': '#f39c12',         # Modern orange
        'error': '#e74c3c',           # Modern red
        'border': '#4a3333',          # Border color
        'border_light': '#5a4040',    # Light border
        'highlight': '#5a3f3f',       # Highlight color
        'shadow': '#0f0808',          # Shadow color
    }
    
    @staticmethod
    def setup_style():
        """Configure ttk styles with modern dark theme"""
        style = ttk.Style()
        
        # Configure main theme
        style.theme_use('clam')
        
        # Main frame styling
        style.configure('Main.TFrame', 
                       background=ModernStyle.COLORS['bg_primary'])
        
        # Card frame styling
        style.configure('Card.TFrame',
                       background=ModernStyle.COLORS['bg_card'],
                       relief='flat',
                       borderwidth=1)
        
        # Title styling with gradient effect
        style.configure('Title.TLabel',
                       background=ModernStyle.COLORS['bg_primary'],
                       foreground=ModernStyle.COLORS['accent_gold'],
                       font=('Segoe UI', 28, 'bold'))
        
        style.configure('Subtitle.TLabel',
                       background=ModernStyle.COLORS['bg_primary'],
                       foreground=ModernStyle.COLORS['text_muted'],
                       font=('Segoe UI', 11))
        
        style.configure('Modern.TLabel',
                       background=ModernStyle.COLORS['bg_primary'],
                       foreground=ModernStyle.COLORS['text_primary'],
                       font=('Segoe UI', 10))
        
        style.configure('Card.TLabel',
                       background=ModernStyle.COLORS['bg_card'],
                       foreground=ModernStyle.COLORS['text_primary'],
                       font=('Segoe UI', 10))
        
        style.configure('Gold.TLabel',
                       background=ModernStyle.COLORS['bg_primary'],
                       foreground=ModernStyle.COLORS['accent_gold'],
                       font=('Segoe UI', 10, 'bold'))
        
        style.configure('GoldCard.TLabel',
                       background=ModernStyle.COLORS['bg_card'],
                       foreground=ModernStyle.COLORS['accent_gold'],
                       font=('Segoe UI', 10, 'bold'))
        
        style.configure('Status.TLabel',
                       background=ModernStyle.COLORS['bg_primary'],
                       foreground=ModernStyle.COLORS['text_accent'],
                       font=('Segoe UI', 11, 'bold'))
        
        # Enhanced button styling
        style.configure('Modern.TButton',
                       background=ModernStyle.COLORS['accent_gold'],
                       foreground=ModernStyle.COLORS['bg_primary'],
                       borderwidth=0,
                       focuscolor='none',
                       font=('Segoe UI', 11, 'bold'),
                       padding=(25, 15))
        
        style.map('Modern.TButton',
                 background=[('active', ModernStyle.COLORS['accent_gold_light']),
                            ('pressed', ModernStyle.COLORS['accent_gold_dark'])])
        
        # Record button with enhanced styling
        style.configure('Record.TButton',
                       background=ModernStyle.COLORS['accent_red'],
                       foreground=ModernStyle.COLORS['text_primary'],
                       borderwidth=0,
                       focuscolor='none',
                       font=('Segoe UI', 13, 'bold'),
                       padding=(35, 18))
        
        style.map('Record.TButton',
                 background=[('active', ModernStyle.COLORS['accent_red_light']),
                            ('pressed', '#b30000')])
        
        # Secondary button styling
        style.configure('Secondary.TButton',
                       background=ModernStyle.COLORS['bg_tertiary'],
                       foreground=ModernStyle.COLORS['text_primary'],
                       borderwidth=1,
                       focuscolor='none',
                       font=('Segoe UI', 10, 'bold'),
                       padding=(20, 12))
        
        style.map('Secondary.TButton',
                 background=[('active', ModernStyle.COLORS['highlight']),
                            ('pressed', ModernStyle.COLORS['bg_secondary'])])
        
        # Enhanced combobox styling
        style.configure('Modern.TCombobox',
                       fieldbackground=ModernStyle.COLORS['bg_tertiary'],
                       background=ModernStyle.COLORS['bg_tertiary'],
                       foreground=ModernStyle.COLORS['text_primary'],
                       borderwidth=1,
                       relief='flat',
                       bordercolor=ModernStyle.COLORS['border_light'],
                       font=('Segoe UI', 10))
        
        # Enhanced progressbar styling
        style.configure('Modern.Horizontal.TProgressbar',
                       background=ModernStyle.COLORS['accent_gold'],
                       troughcolor=ModernStyle.COLORS['bg_tertiary'],
                       borderwidth=0,
                       lightcolor=ModernStyle.COLORS['accent_gold'],
                       darkcolor=ModernStyle.COLORS['accent_gold'])
        
        # Enhanced checkbutton styling
        style.configure('Modern.TCheckbutton',
                       background=ModernStyle.COLORS['bg_card'],
                       foreground=ModernStyle.COLORS['text_primary'],
                       focuscolor='none',
                       font=('Segoe UI', 10))
        
        # Enhanced spinbox styling
        style.configure('Modern.TSpinbox',
                       fieldbackground=ModernStyle.COLORS['bg_tertiary'],
                       background=ModernStyle.COLORS['bg_tertiary'],
                       foreground=ModernStyle.COLORS['text_primary'],
                       borderwidth=1,
                       relief='flat',
                       bordercolor=ModernStyle.COLORS['border_light'],
                       font=('Segoe UI', 10))
        
        # Enhanced labelframe styling
        style.configure('Modern.TLabelframe',
                       background=ModernStyle.COLORS['bg_card'],
                       borderwidth=1,
                       relief='flat',
                       bordercolor=ModernStyle.COLORS['border_light'])
        
        style.configure('Modern.TLabelframe.Label',
                       background=ModernStyle.COLORS['bg_card'],
                       foreground=ModernStyle.COLORS['accent_gold'],
                       font=('Segoe UI', 11, 'bold'))

class TranscriptionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("NeuroNote")
        self.root.geometry("1200x850")
        self.root.configure(bg=ModernStyle.COLORS['bg_primary'])
        
        # Setup modern styling
        ModernStyle.setup_style()
        
        # Detect GPU capability
        self.gpu_info = get_gpu_capability()
        
        # Initialize components
        self.recorder = None
        self.transcriber = None
        
        # Threading and communication
        self.transcription_queue = queue.Queue()
        self.is_recording = False
        self.transcription_thread = None
        
        # Track last transcription time to avoid UI flooding
        self.last_transcription_time = 0
        
        self.setup_gui()
        self.start_queue_monitor()
        
    def setup_gui(self):
        # Main container with padding
        main_container = ttk.Frame(self.root, style='Main.TFrame')
        main_container.pack(fill=tk.BOTH, expand=True, padx=25, pady=25)
        
        # Header section
        header_frame = ttk.Frame(main_container, style='Main.TFrame')
        header_frame.pack(fill=tk.X, pady=(0, 30))
        
        # Title and subtitle
        title_label = ttk.Label(header_frame, text="🎙️ NEURONOTE", 
                               style='Title.TLabel')
        title_label.pack(anchor=tk.W)
        
        subtitle_label = ttk.Label(header_frame, text="Real-time speech-to-text with AI precision", 
                                  style='Subtitle.TLabel')
        subtitle_label.pack(anchor=tk.W, pady=(5, 0))
        
        # GPU info badge
        gpu_frame = ttk.Frame(header_frame, style='Main.TFrame')
        gpu_frame.pack(anchor=tk.E, pady=(10, 0))
        
        gpu_text = f"⚡ {self.gpu_info['name']} • {self.gpu_info['vram']:.1f}GB VRAM"
        gpu_label = ttk.Label(gpu_frame, text=gpu_text, style='Gold.TLabel')
        gpu_label.pack()
        
        # Main content area
        content_frame = ttk.Frame(main_container, style='Main.TFrame')
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel for controls
        left_panel = ttk.Frame(content_frame, style='Main.TFrame')
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        
        # Control cards
        self.create_control_card(left_panel)
        self.create_settings_card(left_panel)
        self.create_status_card(left_panel)
        
        # Right panel for transcription
        right_panel = ttk.Frame(content_frame, style='Main.TFrame')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.create_transcription_panel(right_panel)
        
        # Bottom status bar
        self.create_bottom_status_bar(main_container)
        
    def create_control_card(self, parent):
        """Create main control card"""
        card = ttk.LabelFrame(parent, text="🎯 CONTROLS", 
                             style='Modern.TLabelframe', padding="20")
        card.pack(fill=tk.X, pady=(0, 20))
        
        # Primary record button
        self.record_button = ttk.Button(card, text="● START RECORDING", 
                                       command=self.toggle_recording,
                                       style='Record.TButton')
        self.record_button.pack(fill=tk.X, pady=(0, 15))
        
        # Secondary buttons in a grid
        button_frame = ttk.Frame(card, style='Card.TFrame')
        button_frame.pack(fill=tk.X)
        
        clear_button = ttk.Button(button_frame, text="🗑️ CLEAR", 
                                 command=self.clear_text,
                                 style='Secondary.TButton')
        clear_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        copy_button = ttk.Button(button_frame, text="📋 COPY", 
                                command=self.copy_transcript,
                                style='Secondary.TButton')
        copy_button.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Model selection
        model_frame = ttk.Frame(card, style='Card.TFrame')
        model_frame.pack(fill=tk.X, pady=(20, 0))
        
        ttk.Label(model_frame, text="AI MODEL", style='GoldCard.TLabel').pack(anchor=tk.W)
        
        self.model_var = tk.StringVar(value="medium.en")
        
        # Determine available models based on GPU
        model_options = ['tiny.en', 'base.en', 'small.en', 'medium.en']
        if self.gpu_info['supports_large']:
            model_options.extend(['large-v2'])
            
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, 
                                  state="readonly", style='Modern.TCombobox')
        model_combo['values'] = model_options
        model_combo.pack(fill=tk.X, pady=(8, 0))
        
        # Bind model change
        self.model_var.trace_add("write", self.on_model_change)
        
    def create_settings_card(self, parent):
        """Create settings card"""
        card = ttk.LabelFrame(parent, text="⚙️ SETTINGS", 
                             style='Modern.TLabelframe', padding="20")
        card.pack(fill=tk.X, pady=(0, 20))
        
        # Settings grid
        settings_grid = ttk.Frame(card, style='Card.TFrame')
        settings_grid.pack(fill=tk.X)
        
        # Pause threshold
        ttk.Label(settings_grid, text="Pause Threshold", 
                 style='Card.TLabel').grid(row=0, column=0, sticky="w", pady=(0, 10))
        self.pause_var = tk.DoubleVar(value=1.5)
        pause_spin = ttk.Spinbox(settings_grid, from_=0.5, to=5.0, increment=0.5, 
                                width=10, textvariable=self.pause_var,
                                style='Modern.TSpinbox')
        pause_spin.grid(row=0, column=1, sticky="ew", padx=(10, 0), pady=(0, 10))
        
        # Min chunk duration
        ttk.Label(settings_grid, text="Min Chunk Duration", 
                 style='Card.TLabel').grid(row=1, column=0, sticky="w", pady=(0, 10))
        self.min_chunk_var = tk.DoubleVar(value=1.0)
        chunk_spin = ttk.Spinbox(settings_grid, from_=0.5, to=3.0, increment=0.5, 
                                width=10, textvariable=self.min_chunk_var,
                                style='Modern.TSpinbox')
        chunk_spin.grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=(0, 10))
        
        settings_grid.columnconfigure(1, weight=1)
        
        # GPU acceleration toggle
        gpu_frame = ttk.Frame(card, style='Card.TFrame')
        gpu_frame.pack(fill=tk.X, pady=(15, 0))
        
        self.gpu_var = tk.BooleanVar(value=self.gpu_info['recommend_gpu'])
        gpu_check = ttk.Checkbutton(gpu_frame, text="🚀 GPU Acceleration", 
                                   variable=self.gpu_var,
                                   style='Modern.TCheckbutton')
        gpu_check.pack(anchor=tk.W)
        
    def create_status_card(self, parent):
        """Create status card"""
        card = ttk.LabelFrame(parent, text="📊 STATUS", 
                             style='Modern.TLabelframe', padding="20")
        card.pack(fill=tk.X, pady=(0, 20))
        
        # Status indicator
        self.status_label = ttk.Label(card, text="🟢 Ready to record", 
                                     style='Status.TLabel')
        self.status_label.pack(anchor=tk.W, pady=(0, 15))
        
        # Audio level section
        level_label = ttk.Label(card, text="Audio Level", style='GoldCard.TLabel')
        level_label.pack(anchor=tk.W, pady=(0, 8))
        
        self.level_var = tk.IntVar()
        self.level_bar = ttk.Progressbar(card, variable=self.level_var, 
                                        maximum=100, style='Modern.Horizontal.TProgressbar')
        self.level_bar.pack(fill=tk.X, pady=(0, 10))
        
        # Level percentage
        self.level_text = ttk.Label(card, text="0%", style='Card.TLabel')
        self.level_text.pack(anchor=tk.W)
        
        # Warning label
        self.warning_label = ttk.Label(card, text="", 
                                      background=ModernStyle.COLORS['bg_card'],
                                      foreground=ModernStyle.COLORS['error'],
                                      font=('Segoe UI', 9, 'bold'))
        self.warning_label.pack(anchor=tk.W, pady=(15, 0))
        
    def create_transcription_panel(self, parent):
        """Create transcription panel"""
        panel = ttk.LabelFrame(parent, text="📝 LIVE TRANSCRIPTION", 
                              style='Modern.TLabelframe', padding="20")
        panel.pack(fill=tk.BOTH, expand=True)
        
        # Transcription display
        self.text_display = scrolledtext.ScrolledText(
            panel, 
            wrap=tk.WORD,
            font=('Segoe UI', 12),
            bg=ModernStyle.COLORS['bg_tertiary'],
            fg=ModernStyle.COLORS['text_primary'],
            insertbackground=ModernStyle.COLORS['accent_gold'],
            selectbackground=ModernStyle.COLORS['accent_gold_dark'],
            selectforeground=ModernStyle.COLORS['bg_primary'],
            borderwidth=0,
            relief='flat',
            padx=20,
            pady=20
        )
        self.text_display.pack(fill=tk.BOTH, expand=True)
        
        # Placeholder text
        placeholder_text = "🎤 Click 'START RECORDING' to begin transcription...\n\nYour speech will appear here in real-time."
        self.text_display.insert(tk.END, placeholder_text)
        self.text_display.config(state=tk.DISABLED)
        
    def create_bottom_status_bar(self, parent):
        """Create bottom status bar"""
        status_bar = ttk.Frame(parent, style='Main.TFrame')
        status_bar.pack(fill=tk.X, pady=(20, 0))
        
        # Left side - stats
        stats_frame = ttk.Frame(status_bar, style='Main.TFrame')
        stats_frame.pack(side=tk.LEFT)
        
        self.word_count_label = ttk.Label(stats_frame, text="Words: 0", 
                                         style='Modern.TLabel')
        self.word_count_label.pack(side=tk.LEFT, padx=(0, 20))
        
        self.char_count_label = ttk.Label(stats_frame, text="Characters: 0", 
                                         style='Modern.TLabel')
        self.char_count_label.pack(side=tk.LEFT)
        
        # Right side - version
        version_label = ttk.Label(status_bar, text="NeuroNote", 
                                 style='Modern.TLabel')
        version_label.pack(side=tk.RIGHT)
        
    def update_stats(self):
        """Update word and character counts"""
        try:
            text = self.text_display.get(1.0, tk.END).strip()
            if text and not text.startswith("🎤"):  # Ignore placeholder
                words = len(text.split())
                chars = len(text)
                self.word_count_label.config(text=f"Words: {words:,}")
                self.char_count_label.config(text=f"Characters: {chars:,}")
            else:
                self.word_count_label.config(text="Words: 0")
                self.char_count_label.config(text="Characters: 0")
        except:
            pass
    
    def on_model_change(self, *args):
        """Show warning when selecting large models on limited hardware"""
        model = self.model_var.get()
        if model == 'large-v2' and not self.gpu_info['supports_large']:
            warning = ("⚠️ Large models may be slow on your hardware.\nConsider using medium.en for better performance.")
            self.warning_label.config(text=warning)
        else:
            self.warning_label.config(text="")
    
    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        try:
            # Clear placeholder text
            self.text_display.config(state=tk.NORMAL)
            self.text_display.delete(1.0, tk.END)
            self.text_display.config(state=tk.DISABLED)
            
            # Initialize recorder
            self.recorder = AudioRecorder()
            self.recorder.pause_threshold = self.pause_var.get()
            self.recorder.min_chunk_duration = self.min_chunk_var.get()
            
            # Initialize transcriber with selected model and GPU setting
            model_size = self.model_var.get()
            use_gpu = self.gpu_var.get()
            
            # Update status during model loading
            self.status_label.config(text="🔄 Loading AI model...")
            self.root.update()
            
            self.transcriber = WhisperTranscriber(model_size, use_gpu=use_gpu)
            
            # Start recording
            self.recorder.start_recording()
            self.is_recording = True
            
            # Start transcription thread
            self.transcription_thread = threading.Thread(target=self.transcription_worker, 
                                                        daemon=True)
            self.transcription_thread.start()
            
            # Update UI
            self.record_button.config(text="⏹️ STOP RECORDING")
            self.status_label.config(text="🔴 Recording active...")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start recording: {str(e)}")
            self.cleanup_recording()
    
    def stop_recording(self):
        self.is_recording = False
        
        if self.recorder:
            self.recorder.stop_recording()
        
        # Update UI
        self.record_button.config(text="● START RECORDING")
        self.status_label.config(text="🟡 Recording stopped")
        self.level_var.set(0)
        self.level_text.config(text="0%")
        
        # Clean up resources
        self.cleanup_recording()
    
    def cleanup_recording(self):
        """Clean up recording resources"""
        if self.recorder:
            self.recorder.cleanup()
            self.recorder = None
        
        if self.transcriber:
            self.transcriber.cleanup()
            self.transcriber = None
    
    def transcription_worker(self):
        """Background thread for processing audio chunks"""
        while self.is_recording:
            try:
                if not self.recorder:
                    time.sleep(0.1)
                    continue
                
                # Get audio chunk from recorder
                audio_data = self.recorder.get_audio_chunk()
                if audio_data is not None:
                    # Get audio level for visualization
                    level = self.recorder.get_audio_level()
                    self.root.after(0, self.update_audio_level, level)
                    
                    # Transcribe audio
                    self.root.after(0, lambda: self.status_label.config(text="🔵 AI processing..."))
                    
                    if self.transcriber:
                        transcription = self.transcriber.transcribe_chunk(audio_data)
                        
                        if transcription and transcription.strip():
                            # Add to queue for GUI update
                            self.transcription_queue.put(transcription)
                else:
                    time.sleep(0.1)  # No audio data, wait a bit
                        
            except Exception as e:
                print(f"Transcription error: {e}")
                self.transcription_queue.put(f"[Error: {str(e)}]")
                time.sleep(0.5)  # Wait before retrying
        
        self.root.after(0, lambda: self.status_label.config(text="🟡 Recording stopped"))
    
    def update_audio_level(self, level):
        """Update audio level indicator"""
        level_percent = int(level * 100)
        self.level_var.set(level_percent)
        self.level_text.config(text=f"{level_percent}%")
    
    def start_queue_monitor(self):
        """Monitor the transcription queue and update GUI"""
        try:
            while True:
                transcription = self.transcription_queue.get_nowait()
                self.add_transcription(transcription)
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.start_queue_monitor)
    
    def add_transcription(self, text):
        """Add transcription to the text display"""
        current_time = time.time()
        # Avoid flooding the UI with updates
        if current_time - self.last_transcription_time > 0.1:
            self.text_display.config(state=tk.NORMAL)
            self.text_display.insert(tk.END, text + " ")
            self.text_display.see(tk.END)
            self.text_display.config(state=tk.DISABLED)
            self.last_transcription_time = current_time
            
            # Update stats
            self.update_stats()
            
            # Update status
            if self.is_recording:
                self.status_label.config(text="🔴 Recording active...")
    
    def clear_text(self):
        """Clear the transcription display"""
        self.text_display.config(state=tk.NORMAL)
        self.text_display.delete(1.0, tk.END)
        self.text_display.config(state=tk.DISABLED)
        self.update_stats()
    
    def copy_transcript(self):
        """Copy the current transcript to clipboard"""
        try:
            # Get all text from the display
            transcript_text = self.text_display.get(1.0, tk.END).strip()
            
            if transcript_text and not transcript_text.startswith("🎤"):
                # Copy to clipboard
                self.root.clipboard_clear()
                self.root.clipboard_append(transcript_text)
                self.root.update()  # Keep clipboard content
                
                # Show confirmation
                messagebox.showinfo("📋 Transcript Copied", 
                                  f"Successfully copied to clipboard! ✨\n\n"
                                  f"📊 Length: {len(transcript_text):,} characters\n"
                                  f"📝 Words: {len(transcript_text.split()):,}")
            else:
                messagebox.showwarning("📋 Copy Transcript", 
                                     "No transcript content available to copy.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy transcript: {str(e)}")
    
    def on_closing(self):
        """Handle window closing"""
        if self.is_recording:
            self.stop_recording()
        
        # Clean up
        self.cleanup_recording()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = TranscriptionGUI(root)
    
    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nShutting down...")
        app.on_closing()

if __name__ == "__main__":
    main()