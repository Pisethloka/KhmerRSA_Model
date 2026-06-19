import io
import os
import re
import time
import math
import subprocess
import unicodedata
import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

app = FastAPI(
    title="Khmer Automatic Speech Recognition (ASR) API",
    description="ASR using metythorn/whisper-large-v3 with CTranslate2 backend (Optimized v3.1)",
    version="3.1.0"
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
    "សូមស្វាគម": "សូមស្វាគមន៍",
    "បំលែង": "បំប្លែង",
}

# Optimized CPU INT8 configuration with dynamic threads
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
CPU_THREADS = os.cpu_count() or 4
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
    keywords: str = Form(None)
):
    """
    ASR transcription endpoint with:
    - Zero-I/O in-memory processing via FFmpeg piping directly to soundfile
    - FFmpeg signal enhancement (anlmdn with custom settings + loudnorm)
    - Dynamic spelling hints prompt injection (keywords)
    - Silero VAD pre-filtering for speed and syllable preservation
    - Segment-level average log probability confidence calculation
    - Anti-hallucination decoding parameters
    """
    if model is None:
        raise HTTPException(status_code=500, detail="ASR Model is not loaded on the backend")

    start_time = time.time()

    try:
        # Read the raw uploaded bytes directly into memory
        raw_audio_bytes = await file.read()
        if not raw_audio_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Transcode audio in-memory using FFmpeg (from pipe:0 to pipe:1)
        # anlmdn=s=7 (denoising with strength 7), loudnorm (loudness normalization)
        # Output is 16kHz mono PCM WAV
        command = [
            "ffmpeg", "-y",
            "-i", "pipe:0",
            "-af", "anlmdn=s=7,loudnorm",
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            "-f", "wav",
            "pipe:1"
        ]

        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            wav_bytes, stderr_bytes = process.communicate(input=raw_audio_bytes)
        except FileNotFoundError:
            raise HTTPException(
                status_code=500,
                detail="FFmpeg executable not found on the system path."
            )

        if process.returncode != 0:
            stderr_text = stderr_bytes.decode("utf-8", errors="ignore")
            print(f"FFmpeg failed with exit code {process.returncode}")
            print(f"FFmpeg stderr: {stderr_text}")
            raise HTTPException(
                status_code=400,
                detail=f"Audio transcoding failed: {stderr_text.strip()}"
            )

        if not wav_bytes:
            raise HTTPException(
                status_code=400,
                detail="Audio preprocessing failed. FFmpeg did not produce a valid WAV output."
            )

        # Load enhanced audio directly from memory using soundfile
        try:
            audio, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        except Exception as sf_err:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to decode audio bytes: {str(sf_err)}"
            )

        duration_ms = int((len(audio) / sample_rate) * 1000)
        if duration_ms == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded audio contains no playable audio data."
            )

        # Construct dynamic prompt using spelling hints
        base_prompt = "នេះគឺជាការបកប្រែសំឡេងភាសាខ្មែរសម្រាប់ប្រព័ន្ធស្វ័យប្រវត្តិ។ សូមសរសេរឱ្យបានត្រឹមត្រូវតាមវេយ្យាករណ៍ មានសញ្ញាប្រយុត្តិគតិយុត្ត និងបន្សែងពាក្យឱ្យបានត្រឹមត្រូវ។"
        if keywords and keywords.strip():
            prompt_text = f"{base_prompt} ពាក្យគន្លឹះ៖ {keywords.strip()}."
        else:
            prompt_text = base_prompt

        # Transcribe with maximum precision + speed optimizations
        segments, info = model.transcribe(
            audio,
            language="km",                      # Lock to Khmer — skip auto-detection
            beam_size=3,                        # High-accuracy beam search, balanced with speed
            temperature=[0.0, 0.2],             # Greedy decoding with temperature fallback if low confidence
            patience=1.5,                       # Reduced patient search for speed
            length_penalty=1.0,
            initial_prompt=prompt_text,
            condition_on_previous_text=False,    # Prevent hallucination feedback loops
            no_speech_threshold=0.6,             # Aggressively filter silent segments
            log_prob_threshold=-0.5,             # Discard low-confidence gibberish
            repetition_penalty=1.15,             # Stronger repetition loop prevention
            vad_filter=True,                     # Enable Silero VAD
            vad_parameters=dict(
                threshold=0.6,                   # Higher = more selective speech detection
                min_silence_duration_ms=500,      # Segment on shorter silences
                speech_pad_ms=200,                # Less noise at segment edges
            ),
        )

        # Iterate and concatenate text from transcribed segments, keeping count
        segments_list = list(segments)
        transcription = "".join([segment.text for segment in segments_list])
        num_segments = len(segments_list)

        # Calculate actual transcription confidence (exponential of averaged segment log probs)
        if num_segments > 0:
            avg_logprob = sum(seg.avg_logprob for seg in segments_list) / num_segments
            avg_confidence = round(math.exp(avg_logprob) * 100)
            avg_confidence = max(0, min(100, avg_confidence))
        else:
            avg_confidence = 100

        # Clean up and normalize the Khmer text
        normalized_text = clean_and_normalize_khmer_text(transcription)

        elapsed_ms = int((time.time() - start_time) * 1000)

        # Log transcription stats safely
        print(f"[ASR] {len(normalized_text)} chars | segments={num_segments} | confidence={avg_confidence}% | audio={duration_ms}ms | processing={elapsed_ms}ms")

        return {
            "text": normalized_text,
            "duration_ms": duration_ms,
            "processing_ms": elapsed_ms,
            "num_segments": num_segments,
            "confidence": avg_confidence,
            "word_count": len(normalized_text.split()),
            "char_count": len(normalized_text),
        }

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        print(f"Exception during transcription: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal Server Error: {str(e)}"
        )
