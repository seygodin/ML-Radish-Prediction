"""VQA + STT for the radish demo.

Provides voice-question -> answer over a selected image:
  - whisper-base STT (no ffmpeg: audio is decoded to a numpy float32 mono 16k
    array and passed to the HF pipeline as {"array", "sampling_rate"}).
  - SmolVLM-500M-Instruct VQA over a PIL image.

Both models are lazily loaded ONCE on first use and cached in module globals so
the demo server stays fast to start.

Public API:
    transcribe(wav_bytes) -> str
    vqa(pil_image, question) -> str
    answer(pil_image, question_text=None, wav_bytes=None) -> dict
"""
from __future__ import annotations

import io
import threading
from typing import Optional

import numpy as np
from PIL import Image

# Lazily-imported heavy deps (torch/transformers) are referenced inside the
# loaders so importing this module is cheap.

_ASR = None
_VLM_PROC = None
_VLM_MODEL = None
_LOCK = threading.Lock()

WHISPER_MODEL_ID = "openai/whisper-base"
SMOLVLM_MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"


def _device_index() -> int:
    """cuda 디바이스 인덱스 정수 반환(pipeline device 인자용)."""
    import torch
    return 0 if torch.cuda.is_available() else -1


def _get_asr():
    """Lazy-load the whisper-base ASR pipeline (cached)."""
    global _ASR
    if _ASR is not None:
        return _ASR
    with _LOCK:
        if _ASR is None:
            from transformers import pipeline
            _ASR = pipeline(
                "automatic-speech-recognition",
                model=WHISPER_MODEL_ID,
                device=_device_index(),
            )
    return _ASR


def _get_vlm():
    """Lazy-load the SmolVLM processor + model (cached)."""
    global _VLM_PROC, _VLM_MODEL
    if _VLM_MODEL is not None:
        return _VLM_PROC, _VLM_MODEL
    with _LOCK:
        if _VLM_MODEL is None:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
            cuda = torch.cuda.is_available()
            proc = AutoProcessor.from_pretrained(SMOLVLM_MODEL_ID)
            model = AutoModelForImageTextToText.from_pretrained(
                SMOLVLM_MODEL_ID,
                dtype=torch.bfloat16 if cuda else torch.float32,
            )
            model = model.to("cuda:0" if cuda else "cpu").eval()
            _VLM_PROC, _VLM_MODEL = proc, model
    return _VLM_PROC, _VLM_MODEL


def _decode_wav_to_16k_mono(wav_bytes: bytes) -> np.ndarray:
    """Decode audio bytes to a float32 mono 16kHz numpy array (no ffmpeg).

    Supports any format libsndfile reads: WAV / MP3 / OGG / FLAC / AIFF, etc.
    Compressed formats needing ffmpeg (m4a/AAC, webm/opus) are NOT supported and
    raise a clear ValueError so the API can return a helpful message.
    """
    import soundfile as sf
    try:
        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    except Exception as e:  # noqa: BLE001
        raise ValueError(
            "오디오 디코딩 실패 — 지원 형식은 wav/mp3/ogg/flac 입니다 "
            "(m4a·webm 등은 미지원). 원본 오류: " + str(e)
        ) from e
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = np.asarray(data, dtype=np.float32)
    if sr != 16000:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
        data = np.asarray(data, dtype=np.float32)
    return data


def transcribe(wav_bytes: bytes) -> str:
    """Transcribe WAV audio bytes to text via whisper-base."""
    arr = _decode_wav_to_16k_mono(wav_bytes)
    if arr.size == 0:
        return ""
    asr = _get_asr()
    out = asr({"array": arr, "sampling_rate": 16000})
    return (out.get("text") or "").strip()


def vqa(pil_image: Image.Image, question: str) -> str:
    """Answer a question about a PIL image via SmolVLM."""
    import torch
    proc, model = _get_vlm()
    img = pil_image.convert("RGB")
    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": question}],
    }]
    prompt = proc.apply_chat_template(messages, add_generation_prompt=True)
    inputs = proc(text=prompt, images=[img], return_tensors="pt")
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=256)
    ans = proc.batch_decode(ids, skip_special_tokens=True)[0]
    return ans.split("Assistant:")[-1].strip()


def answer(
    pil_image: Image.Image,
    question_text: Optional[str] = None,
    wav_bytes: Optional[bytes] = None,
) -> dict:
    """Run the full voice/text VQA flow.

    If wav_bytes is given, transcribe it to obtain the question; otherwise use
    question_text. Returns {transcript, question, answer}.
    """
    transcript = ""
    if wav_bytes:
        transcript = transcribe(wav_bytes)
        question = transcript or (question_text or "")
    else:
        question = (question_text or "").strip()

    if not question:
        return {
            "transcript": transcript,
            "question": "",
            "answer": "(질문이 비어 있습니다. 음성이 인식되지 않았거나 텍스트가 없습니다.)",
        }

    ans = vqa(pil_image, question)
    return {"transcript": transcript, "question": question, "answer": ans}
