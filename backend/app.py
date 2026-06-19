import os
import gc
import re
import uuid
import shutil
import time
import tempfile
import subprocess
import unicodedata
import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

app = FastAPI(
    title="Khmer Automatic Speech Recognition (ASR) API",
    description="ASR using metythorn/whisper-large-v3 with CTranslate2 backend (Optimized)",
    version="2.0.0"
)

# Enable CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pre-compile regex patterns at the module level for performance
FORMATTING_CHARS_PATTERN = re.compile(r'[\u200b\u200c\u200d\u200e\u200f]+')
KHMER_SPACE_SANDWICH_PATTERN = re.compile(r'(?<=[\u1780-\u17FF\u19E0-\u19FF])\s+(?=[\u1780-\u17FF\u19E0-\u19FF])')

# Expanded phonetic slip correction dictionary
# Maps common model hallucination misspellings to their correct Khmer forms
PHONETIC_SLIP_DICT = {
    "ច្រឹសរ": "ជ្រើសរើស",
    "ច្រើសរើស": "ជ្រើសរើស",
    "ប្រើស": "ប្រើ",
    "សួស្ដី": "សួស្តី",
    "កម្ពុជ្ជា": "កម្ពុជា",
    "អរគុនណ": "អរគុណ",
    "សរសេរ": "សរសេរ",
    "សំរាប់": "សម្រាប់",
    "បន្ថែន": "បន្ថែម",
    "កំពុត": "កំពត",
}

# Optimized CPU INT8 configuration
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
CPU_THREADS = 4
LOCAL_CT2_PATH = "./metythorn_whisper_large_v3_ct2"

model = None

def initialize_model():
    """
    Startup model initialization with CPU thread tuning.
    Checks for local CTranslate2 directory first,
    falling back to cache loading or conversion from Hugging Face format.
    """
    global model

    # Check if local CTranslate2 directory exists
    if os.path.exists(LOCAL_CT2_PATH):
        print(f"Loading local CTranslate2 model from '{LOCAL_CT2_PATH}'...")
        try:
            model = WhisperModel(
                LOCAL_CT2_PATH,
                device=DEVICE,
                compute_type=COMPUTE_TYPE,
                cpu_threads=CPU_THREADS,
            )
            print(f"Local CTranslate2 model loaded successfully (cpu_threads={CPU_THREADS}).")
        except Exception as e:
            print(f"Failed to load local model: {e}")
            model = None

    if model is None:
        print("Falling back to loading 'metythorn/whisper-large-v3' directly via Hugging Face cache...")
        try:
            model = WhisperModel(
                "metythorn/whisper-large-v3",
                device=DEVICE,
                compute_type=COMPUTE_TYPE,
                cpu_threads=CPU_THREADS,
            )
            print("Model loaded from Hugging Face cache successfully.")
        except Exception as e:
            print(f"Failed to load directly from Hugging Face: {e}")
            print("Attempting to convert PyTorch model from Hugging Face to CTranslate2...")
            try:
                from ctranslate2.converters import TransformersConverter
                converter = TransformersConverter(
                    model_name_or_path="metythorn/whisper-large-v3",
                    copy_files=["tokenizer.json", "preprocessor_config.json"]
                )
                converter.convert(output_dir=LOCAL_CT2_PATH, quantization=COMPUTE_TYPE, force=True)
                model = WhisperModel(
                    LOCAL_CT2_PATH,
                    device=DEVICE,
                    compute_type=COMPUTE_TYPE,
                    cpu_threads=CPU_THREADS,
                )
                print("Model converted and loaded successfully.")
            except Exception as conv_err:
                print(f"Conversion/load failed: {conv_err}")
                model = None

# Initialize Whisper model
initialize_model()

def clean_and_normalize_khmer_text(text: str) -> str:
    """
    Linguistic post-processing module to ensure clean and continuous Khmer text
    while preserving spacing around non-Khmer words.
    """
    # Step A: Unconditionally strip hidden formatting characters
    text = FORMATTING_CHARS_PATTERN.sub("", text)

    # Step B: Strip spaces ONLY when tightly sandwiched between Khmer characters
    text = KHMER_SPACE_SANDWICH_PATTERN.sub("", text)

    # Step C: Phonetic slip dictionary mapping
    for slip, correction in PHONETIC_SLIP_DICT.items():
        text = text.replace(slip, correction)

    # Step D: Apply NFC Unicode normalization and strip leading/trailing whitespace
    return unicodedata.normalize("NFC", text).strip()

@app.get("/health")
def health():
    """
    Health check endpoint exposing model state and configuration details.
    """
    return {
        "status": "ok" if model is not None else "error",
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "cpu_threads": CPU_THREADS,
        "vad_enabled": True,
        "language_lock": "km",
    }

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    initial_prompt: str = Form("ការសន្ទនាជាភាសាខ្មែរ។")
):
    """
    ASR transcription endpoint with:
    - FFmpeg signal enhancement (anlmdn + loudnorm)
    - Silero VAD pre-filtering for speed
    - Anti-hallucination decoding parameters
    - Khmer language lock
    - Aggressive resource cleanup
    """
    if model is None:
        raise HTTPException(status_code=500, detail="ASR Model is not loaded on the backend")

    start_time = time.time()

    # Generate unique paths for temporary files
    input_suffix = os.path.splitext(file.filename)[1] if file.filename else ".tmp"
    if not input_suffix:
        input_suffix = ".tmp"

    temp_dir = tempfile.gettempdir()
    unique_id = uuid.uuid4().hex
    input_path = os.path.join(temp_dir, f"input_{unique_id}{input_suffix}")
    output_wav_path = os.path.join(temp_dir, f"output_{unique_id}.wav")

    try:
        # Write the uploaded file to disk
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Transcode audio using FFmpeg with signal enhancement filters (16kHz mono PCM)
        # anlmdn = noise reduction, loudnorm = volume normalization
        command = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-af", "anlmdn,loudnorm",
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            output_wav_path
        ]

        try:
            subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg failed with exit code {e.returncode}")
            print(f"FFmpeg stderr: {e.stderr}")
            raise HTTPException(
                status_code=400,
                detail=f"Audio transcoding failed: {e.stderr.strip()}"
            )
        except FileNotFoundError:
            raise HTTPException(
                status_code=500,
                detail="FFmpeg executable not found on the system path."
            )

        # Confirm transcoded file was created and is not empty
        if not os.path.exists(output_wav_path) or os.path.getsize(output_wav_path) == 0:
            raise HTTPException(
                status_code=400,
                detail="Audio preprocessing failed. FFmpeg did not produce a valid WAV output."
            )

        # Load enhanced audio directly using soundfile (lightweight, no librosa overhead)
        audio, sample_rate = sf.read(output_wav_path, dtype="float32")
        duration_ms = int((len(audio) / sample_rate) * 1000)

        # Establish prompt context
        prompt_text = initial_prompt.strip() if (initial_prompt and initial_prompt.strip()) else "ការសន្ទនាជាភាសាខ្មែរ។"

        # Transcribe with maximum precision + speed optimizations
        segments, info = model.transcribe(
            audio,
            language="km",                      # Lock to Khmer — skip auto-detection
            beam_size=5,
            temperature=0.0,
            patience=2.0,
            length_penalty=1.0,
            initial_prompt=prompt_text,
            condition_on_previous_text=False,    # Prevent hallucination feedback loops
            no_speech_threshold=0.6,             # Aggressively filter silent segments
            log_prob_threshold=-0.5,             # Discard low-confidence gibberish
            repetition_penalty=1.1,              # Penalize repeated words/phrases
            vad_filter=True,                     # Enable Silero VAD
            vad_parameters=dict(
                threshold=0.6,                   # Higher = more selective speech detection
                min_silence_duration_ms=500,      # Segment on shorter silences
                speech_pad_ms=200,                # Less noise at segment edges
            ),
        )

        # Concatenate text from transcribed segments
        transcription = "".join([segment.text for segment in segments])

        # Clean up and normalize the Khmer text
        normalized_text = clean_and_normalize_khmer_text(transcription)

        elapsed_ms = int((time.time() - start_time) * 1000)

        # Log transcription stats (safe for non-UTF-8 consoles)
        print(f"[ASR] {len(normalized_text)} chars | audio={duration_ms}ms | processing={elapsed_ms}ms")

        return {
            "text": normalized_text,
            "duration_ms": duration_ms,
            "processing_ms": elapsed_ms,
        }

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        print(f"Exception during transcription: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal Server Error: {str(e)}"
        )

    finally:
        # Guarantee removal of temporary files from disk
        if os.path.exists(input_path):
            try:
                os.remove(input_path)
            except Exception as e:
                print(f"Failed to delete temp input file: {e}")
        if os.path.exists(output_wav_path):
            try:
                os.remove(output_wav_path)
            except Exception as e:
                print(f"Failed to delete temp output file: {e}")

        # Explicitly run garbage collection to reclaim memory immediately
        gc.collect()
