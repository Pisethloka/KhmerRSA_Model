import os
import re
import uuid
import tempfile
import shutil
import subprocess
import unicodedata
import torch
import librosa
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from transformers import WhisperProcessor, WhisperForConditionalGeneration

app = FastAPI(
    title="Khmer Automatic Speech Recognition (ASR) API",
    description="ASR using metythorn/whisper-large-v3",
    version="1.0.0"
)

# Enable CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model loading (once at startup)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if DEVICE == "cuda" else torch.float32
print(f"Loading Whisper model on {DEVICE} in {torch_dtype}...")

try:
    processor = WhisperProcessor.from_pretrained("metythorn/whisper-large-v3")
    model = WhisperForConditionalGeneration.from_pretrained(
        "metythorn/whisper-large-v3",
        torch_dtype=torch_dtype
    ).to(DEVICE)
    model.eval()
    print("ASR model loaded and set to eval mode.")
except Exception as e:
    print(f"Error loading model: {e}")
    processor = None
    model = None

@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "error",
        "device": DEVICE
    }

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    initial_prompt: str = Form(None)
):
    if model is None or processor is None:
        raise HTTPException(status_code=500, detail="ASR Model is not loaded on the backend")

    # Audio preprocessing
    input_suffix = os.path.splitext(file.filename)[1] if file.filename else ".tmp"
    if not input_suffix:
        input_suffix = ".tmp"

    temp_dir = tempfile.gettempdir()
    unique_id = uuid.uuid4().hex
    input_path = os.path.join(temp_dir, f"input_{unique_id}{input_suffix}")
    output_wav_path = os.path.join(temp_dir, f"output_{unique_id}.wav")

    try:
        # Save the uploaded file to input_path
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Convert using FFmpeg with filters: anlmdn, loudnorm
        command = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-af", "anlmdn,loudnorm",
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            output_wav_path
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Assert output file exists and has non-zero size
        if not os.path.exists(output_wav_path) or os.path.getsize(output_wav_path) == 0:
            print(f"FFmpeg stdout: {result.stdout}")
            print(f"FFmpeg stderr: {result.stderr}")
            raise HTTPException(
                status_code=400,
                detail="Audio preprocessing failed. FFmpeg did not produce a valid WAV output."
            )

        # Establish prompt context
        prompt_text = initial_prompt.strip() if (initial_prompt and initial_prompt.strip()) else "ការសន្ទនាជាភាសាខ្មែរ។"
        prompt_ids = processor.get_prompt_ids(prompt_text, return_tensors="pt").to(DEVICE)

        # Apply generation config
        model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="km", task="transcribe")
        model.generation_config.num_beams = 5
        model.generation_config.temperature = 0.0
        model.generation_config.do_sample = False
        # Strict repetition penalties can break natural Khmer repeating auxiliary tokens (like ការ, ភាព, ដដែលៗ)
        # causing the model to break word-spelling rules, so we keep repetition_penalty at 1.0.
        model.generation_config.repetition_penalty = 1.0

        # Inference
        # Read the WAV using librosa.load(path, sr=16000)
        audio, sample_rate = librosa.load(output_wav_path, sr=16000)
        duration_ms = int((len(audio) / sample_rate) * 1000)

        # Pass it to processor and move to device
        input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(DEVICE)

        # Call model.generate
        with torch.no_grad():
            output_ids = model.generate(
                input_features,
                prompt_ids=prompt_ids
            )

        # Decode with processor.batch_decode
        transcription = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

        # Post-processing:
        # 1. Strip zero-width spaces (\u200b) everywhere
        clean_text = re.sub(r'\u200b', '', transcription)
        # 2. Strip whitespaces only between Khmer characters to prevent breaking English words in mixed conversations
        clean_text = re.sub(r'(?<=[\u1780-\u17FF\u19E0-\u19FF])\s+(?=[\u1780-\u17FF\u19E0-\u19FF])', '', clean_text)

        # Apply NFC Unicode normalization and strip
        normalized_text = unicodedata.normalize("NFC", clean_text).strip()

        # Debug log
        print(f"[ASR] raw tokens: {output_ids}")

        return {
            "text": normalized_text,
            "duration_ms": duration_ms
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
        # Delete both temp files to prevent disk leaks
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

        # Clear GPU cache to prevent memory bottlenecks on large-v3 model
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Failed to clear CUDA cache: {e}")
