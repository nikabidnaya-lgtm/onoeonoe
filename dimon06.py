# -*- coding: utf-8 -*-
import json
import os
import re
import time
import subprocess
import atexit
import shutil
# Важно: задаём кэши HF ДО импортов huggingface/transformers,
# иначе на Windows они могут зафиксироваться в C:\Users\...\.cache.
_EARLY_WORK_DIR = r"D:\pinokio"
os.environ.setdefault("HF_HOME", os.path.join(_EARLY_WORK_DIR, "hf_cache"))
os.environ.setdefault("TORCH_HOME", os.path.join(_EARLY_WORK_DIR, "torch_cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
if 'RVC_INFER_PYTHON' not in os.environ:
    os.environ['RVC_INFER_PYTHON'] = r'C:\Users\BAand\Miniconda3\envs\rvc_infer\python.exe'
from huggingface_hub import snapshot_download
import wave
from pathlib import Path
import traceback
import inspect
import importlib
import threading
import urllib.request
import glob
import numpy as np
import torch
import whisper
from moviepy import VideoFileClip
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
import sys
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["LANG"] = "en_US.UTF-8"
os.environ["LC_ALL"] = "en_US.UTF-8"
os.environ["HF_HUB_USER_AGENT"] = "huggingface_hub/1.5.0 python-requests/2.31.0"
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
os.environ["HF_TOKEN"] = "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"
# os.environ["HF_TOKEN"] = "<PUT_YOUR_TOKEN_HERE>"  # задайте в окружении
_torch_load_original = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load_original(*args, **kwargs)
torch.load = _torch_load_compat

SOURCE_LANGUAGE = "ru"
TARGET_LANGUAGE = "en"
MAX_MARIAN_INPUT_TOKENS = 450
MAX_XTTS_REFERENCE_SECONDS = 30
OUTPUT_SAMPLE_RATE = 16000
MIN_STRETCH_RATIO = 0.80
MAX_STRETCH_RATIO = 1.25
ULTRA_SAFE_MODE = True
ULTRA_SAFE_MIN_RATIO = 0.92
ULTRA_SAFE_MAX_RATIO = 1.08
MAX_XTTS_TEXT_CHARS = 240
MAX_EXPRESSIVE_SEGMENT_SECONDS = 3.8
MIN_EXPRESSIVE_SEGMENT_SECONDS = 0.35
TTS_BACKEND = "qwen_local"
QWEN_MODEL_ID_06 = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
QWEN_MODEL_ID_17 = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
ENABLE_QWEN_SIZE = str(os.environ.get("ENABLE_QWEN_SIZE", "0.6")).strip().lower()
if ENABLE_QWEN_SIZE in {"1.7", "17", "1_7", "1-7", "1,7"}:
    _qwen_model_by_flag = QWEN_MODEL_ID_17
elif ENABLE_QWEN_SIZE in {"0.6", "06", "0_6", "0-6", "0,6"}:
    _qwen_model_by_flag = QWEN_MODEL_ID_06
else:
    print(f"[LOCAL][WARN] Некорректный ENABLE_QWEN_SIZE={ENABLE_QWEN_SIZE!r}, используем 0.6B")
    _qwen_model_by_flag = QWEN_MODEL_ID_06
QWEN_LOCAL_MODEL_ID = os.environ.get("QWEN_LOCAL_MODEL_ID", _qwen_model_by_flag)
QWEN_LOCAL_DEVICE = -1
QWEN_LOCAL_TRUST_REMOTE_CODE = True
QWEN_LOAD_AT_START = False
QWEN_LOCAL_DTYPE = os.environ.get("QWEN_LOCAL_DTYPE", "float16")
QWEN_LOW_CPU_MEM_USAGE = True
QWEN_OUTPUT_STYLE_PROMPT = "Speak in natural modern American English, emotionally rich, human-like, with realistic prosody and clean articulation."

RVC_ENABLED = True
RVC_AUTO_TRAIN = True
RVC_REPO_DIR = os.path.join(_EARLY_WORK_DIR, "rvc")
RVC_WORK_DIR = os.path.join(_EARLY_WORK_DIR, "rvc_workspace")
RVC_MODELS_DIR = os.path.join(RVC_WORK_DIR, "models")
RVC_DATASET_DIR = os.path.join(RVC_WORK_DIR, "datasets")
RVC_TMP_DIR = os.path.join(RVC_WORK_DIR, "tmp")
RVC_TRAIN_SCRIPT = os.environ.get("RVC_TRAIN_SCRIPT", os.path.join(RVC_REPO_DIR, "train_rvc.py"))
RVC_INFER_SCRIPT = os.environ.get("RVC_INFER_SCRIPT", os.path.join(RVC_REPO_DIR, "infer_rvc.py"))
RVC_EPOCHS = int(os.environ.get("RVC_EPOCHS", "250"))
RVC_F0_METHOD = os.environ.get("RVC_F0_METHOD", "rmvpe")
RVC_PYTHON = os.environ.get("RVC_PYTHON", sys.executable)
RVC_INFER_PYTHON = os.environ.get("RVC_INFER_PYTHON", RVC_PYTHON)
RVC_PITCH_SHIFT = int(os.environ.get("RVC_PITCH_SHIFT", "0"))
RVC_INDEX_RATE = float(os.environ.get("RVC_INDEX_RATE", "0.85"))
RVC_FILTER_RADIUS = int(os.environ.get("RVC_FILTER_RADIUS", "7"))
RVC_PROTECT = float(os.environ.get("RVC_PROTECT", "0.35"))
RVC_AUTOTUNE = os.environ.get("RVC_AUTOTUNE", "false").lower() == "true"
RVC_AUTO_INSTALL_SCRIPTS = os.environ.get("RVC_AUTO_INSTALL_SCRIPTS", "true").lower() == "true"
RVC_GIT_REPO = os.environ.get("RVC_GIT_REPO", "https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git")
RVC_GIT_BRANCH = os.environ.get("RVC_GIT_BRANCH", "main")
RVC_TRAIN_SCRIPT_URL = os.environ.get("RVC_TRAIN_SCRIPT_URL", "")
RVC_INFER_SCRIPT_URL = os.environ.get("RVC_INFER_SCRIPT_URL", "")
RVC_SAMPLE_RATE = os.environ.get("RVC_SAMPLE_RATE", "40k")
RVC_MODEL_VERSION = os.environ.get("RVC_MODEL_VERSION", "v2")
RVC_BATCH_SIZE = int(os.environ.get("RVC_BATCH_SIZE", "4"))
RVC_SAVE_EVERY_EPOCH = int(os.environ.get("RVC_SAVE_EVERY_EPOCH", "10"))
RVC_IF_F0 = int(os.environ.get("RVC_IF_F0", "1"))
RVC_IF_LATEST = int(os.environ.get("RVC_IF_LATEST", "1"))
RVC_IF_CACHE_GPU = int(os.environ.get("RVC_IF_CACHE_GPU", "0"))
RVC_TRAIN_GPUS = os.environ.get("RVC_TRAIN_GPUS", "0")
RVC_GLOO_SOCKET_IFNAME = os.environ.get("RVC_GLOO_SOCKET_IFNAME", "").strip()
RVC_TRAIN_REQUIRED = os.environ.get("RVC_TRAIN_REQUIRED", "false").lower() == "true"
RVC_TRAIN_MAX_ATTEMPTS = int(os.environ.get("RVC_TRAIN_MAX_ATTEMPTS", "2"))
RVC_STRICT_INFER_DEPS = os.environ.get("RVC_STRICT_INFER_DEPS", "true").lower() == "true"
CURRENT_RVC_MODEL_PATH = None
CURRENT_RVC_INDEX_PATH = None
CLAUSE_MIN_CHARS = 10
MICRO_PAUSE_COMMA_MS = 90
MICRO_PAUSE_SEMICOLON_MS = 130
MICRO_PAUSE_SENTENCE_MS = 180
REF_PAD_SECONDS = 0.25
REF_MIN_SECONDS = 2.2
REF_MAX_SECONDS = 8.0
GLOSSARY_RU_EN = {
    "алготрейдинг": "algo trading",
    "робот": "trading bot",
}
WORK_DIR = r"D:\pinokio"
VIDEO_FOLDER = r"D:\!!Dima\01"
os.environ["HF_HOME"] = os.path.join(WORK_DIR, "hf_cache")
os.environ["TORCH_HOME"] = os.path.join(WORK_DIR, "torch_cache")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["IMAGEIO_FFMPEG_EXE"] = r"C:\ProgramData\chocolatey\lib\ffmpeg\tools\ffmpeg\bin\ffmpeg.exe"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
torch.hub.set_dir(os.path.join(WORK_DIR, "torch_hub"))
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(VIDEO_FOLDER, exist_ok=True)
os.makedirs(RVC_WORK_DIR, exist_ok=True)
os.makedirs(RVC_MODELS_DIR, exist_ok=True)
os.makedirs(RVC_DATASET_DIR, exist_ok=True)
os.makedirs(RVC_TMP_DIR, exist_ok=True)
os.environ["TMP"] = RVC_TMP_DIR
os.environ["TEMP"] = RVC_TMP_DIR
os.environ["TMPDIR"] = RVC_TMP_DIR

# === ГЛОБАЛЬНЫЙ ФИКС ДЛЯ WINDOWS + RVC (самый важный блок) ===
os.environ["USE_LIBUV"] = "0"
os.environ["TORCH_USE_LIBUV"] = "0"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29501"
os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"

def _ensure_sox_in_path():
    if shutil.which("sox"):
        print(f"[OK] sox найден в PATH: {shutil.which('sox')}")
        return
    candidate_dirs = [
        r"C:\Program Files (x86)\sox-14-4-2",
        r"C:\Program Files\sox-14-4-2",
    ]
    for d in candidate_dirs:
        exe = os.path.join(d, "sox.exe")
        if os.path.exists(exe):
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            print(f"[OK] sox добавлен в PATH на время процесса: {exe}")
            return
    print("[WARN] SoX не найден (ни в PATH, ни в стандартных путях).")

def _check_runtime_tools():
    ffmpeg_path = shutil.which("ffmpeg") or os.environ.get("IMAGEIO_FFMPEG_EXE")
    ffprobe_path = shutil.which("ffprobe")
    sox_path = shutil.which("sox")
    print(f"[CHECK] ffmpeg: {ffmpeg_path if ffmpeg_path else 'NOT FOUND'}")
    print(f"[CHECK] ffprobe: {ffprobe_path if ffprobe_path else 'NOT FOUND'}")
    print(f"[CHECK] sox: {sox_path if sox_path else 'NOT FOUND'}")

def _resolve_rvc_infer_python() -> str:
    """Пытается подобрать Python 3.10 для RVC infer (fairseq на py311 часто ломается)."""
    explicit = os.environ.get("RVC_INFER_PYTHON", "").strip().strip('"').strip("'")
    if explicit:
        if os.path.exists(explicit):
            print(f"[RVC] Используем RVC_INFER_PYTHON из окружения: {explicit}")
            return explicit
        print(f"[RVC][WARN] RVC_INFER_PYTHON задан, но путь не существует: {explicit}")

    current = RVC_PYTHON
    if sys.version_info < (3, 11):
        return current

    candidates = []

    # 1) py launcher (Windows): py -3.10
    try:
        probe = subprocess.run(
            ["py", "-3.10", "-c", "import sys;print(sys.executable)"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if probe.returncode == 0:
            exe = (probe.stdout or "").strip()
            if exe and os.path.exists(exe):
                candidates.append(exe)
    except Exception:
        pass

    # 2) python3.10 в PATH
    for name in ("python3.10", "python310"):
        exe = shutil.which(name)
        if exe and os.path.exists(exe):
            candidates.append(exe)

    # 3) Поиск в conda envs рядом с текущим python
    try:
        cur = Path(sys.executable)
        conda_root = cur.parent.parent  # .../envs/<env>
        envs_dir = conda_root.parent
        if envs_dir.name.lower() == "envs" and envs_dir.exists():
            for env in envs_dir.iterdir():
                if not env.is_dir():
                    continue
                nm = env.name.lower()
                if "310" in nm or "py310" in nm or "python310" in nm or nm in {"rvc_infer", "rvc-py310", "rvc_py310"}:
                    py_exe = env / "python.exe"
                    if py_exe.exists():
                        candidates.append(str(py_exe))
    except Exception:
        pass

    # 4) Явный приоритет env с именем rvc_infer рядом с текущим conda root
    try:
        cur = Path(sys.executable)
        conda_root = cur.parent.parent
        envs_dir = conda_root.parent
        direct = envs_dir / "rvc_infer" / "python.exe"
        if direct.exists():
            candidates.append(str(direct))
    except Exception:
        pass

    # уникализируем
    uniq = []
    seen = set()
    for c in candidates:
        key = os.path.normcase(os.path.abspath(c))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)

    for exe in uniq:
        if os.path.normcase(os.path.abspath(exe)) != os.path.normcase(os.path.abspath(current)):
            print(f"[RVC] Автовыбор python для infer: {exe}")
            return exe

    print("[RVC][WARN] Python 3.10 для RVC infer не найден автоматически; будет использован текущий Python.")
    print("[RVC][WARN] Рекомендуется задать RVC_INFER_PYTHON на py310 интерпретатор.")
    return current

def load_pcm16_wav(path: str, max_seconds: float | None = None, offset_seconds: float = 0.0):
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        total_frames = wf.getnframes()
        start_frame = int(max(0.0, offset_seconds) * sample_rate)
        if start_frame > total_frames:
            start_frame = total_frames
        wf.setpos(start_frame)
        if max_seconds is None:
            frames_to_read = total_frames - start_frame
        else:
            frames_to_read = min(int(max_seconds * sample_rate), total_frames - start_frame)
        raw = wf.readframes(frames_to_read)
    if sample_width != 2:
        raise RuntimeError(f"Ожидался WAV 16-bit PCM, но sample_width={sample_width}")
    audio_int16 = np.frombuffer(raw, dtype=np.int16)
    if audio_int16.size == 0:
        return np.zeros(1, dtype=np.float32), sample_rate
    audio = audio_int16.astype(np.float32) / 32768.0
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)
    return audio, sample_rate

def save_pcm16_wav(path: str, audio: np.ndarray, sample_rate: int):
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())

def _build_atempo_filter(speed: float) -> str:
    speed = max(0.05, float(speed))
    stages = []
    while speed < 0.5:
        stages.append(0.5)
        speed /= 0.5
    while speed > 2.0:
        stages.append(2.0)
        speed /= 2.0
    stages.append(speed)
    return ",".join(f"atempo={v:.6f}" for v in stages)

def _time_stretch_ffmpeg_preserve_pitch(audio: np.ndarray, target_samples: int) -> np.ndarray:
    if audio.size <= 1 or target_samples <= 1:
        return np.zeros(max(1, target_samples), dtype=np.float32)
    in_path = Path(WORK_DIR) / "tmp_timestretch_in.wav"
    out_path = Path(WORK_DIR) / "tmp_timestretch_out.wav"
    save_pcm16_wav(str(in_path), audio, OUTPUT_SAMPLE_RATE)
    speed = len(audio) / float(target_samples)
    atempo = _build_atempo_filter(speed)
    ffmpeg_bin = os.environ.get("IMAGEIO_FFMPEG_EXE", "ffmpeg")
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(in_path),
        "-filter:a",
        atempo,
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        stretched, sr = load_pcm16_wav(str(out_path))
        stretched = resample_audio_np(stretched, sr, OUTPUT_SAMPLE_RATE)
        if len(stretched) < target_samples:
            stretched = np.pad(stretched, (0, target_samples - len(stretched)), mode="constant")
        elif len(stretched) > target_samples:
            stretched = stretched[:target_samples]
        return stretched.astype(np.float32)
    except Exception:
        if len(audio) < target_samples:
            return np.pad(audio.astype(np.float32), (0, target_samples - len(audio)), mode="constant")
        return audio[:target_samples].astype(np.float32)

def resample_audio_np(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size <= 1:
        return np.zeros(max(1, int(audio.size)), dtype=np.float32)
    if src_sr <= 0 or dst_sr <= 0:
        return audio.astype(np.float32)
    if src_sr == dst_sr:
        return audio.astype(np.float32)
    tensor = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    new_length = max(1, int(round(audio.shape[0] * float(dst_sr) / float(src_sr))))
    out = torch.nn.functional.interpolate(tensor, size=new_length, mode="linear", align_corners=False)
    return out.squeeze(0).squeeze(0).numpy().astype(np.float32)

def preprocess_for_asr(audio: np.ndarray) -> np.ndarray:
    audio = audio - np.mean(audio)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = audio / peak * 0.92
    gate = 0.008
    audio[np.abs(audio) < gate] = 0.0
    return audio.astype(np.float32)

def ultra_safe_fit_to_samples(audio: np.ndarray, target_samples: int) -> np.ndarray:
    if target_samples <= 1:
        return np.zeros(1, dtype=np.float32)
    if audio.size <= 1:
        return np.zeros(target_samples, dtype=np.float32)
    cur = len(audio)
    ratio = cur / float(target_samples)
    if ULTRA_SAFE_MIN_RATIO <= ratio <= ULTRA_SAFE_MAX_RATIO:
        return _time_stretch_ffmpeg_preserve_pitch(audio.astype(np.float32), target_samples)
    if cur < target_samples:
        pad = target_samples - cur
        return np.pad(audio.astype(np.float32), (0, pad), mode="constant")
    out = audio[:target_samples].astype(np.float32)
    fade = min(256, len(out))
    if fade > 1:
        out[-fade:] *= np.linspace(1.0, 0.0, num=fade, dtype=np.float32)
    return out

def stretch_to_samples(audio: np.ndarray, target_samples: int) -> np.ndarray:
    if target_samples <= 1:
        return np.zeros(1, dtype=np.float32)
    if audio.size <= 1:
        return np.zeros(target_samples, dtype=np.float32)
    if len(audio) == target_samples:
        return audio.astype(np.float32)
    if ULTRA_SAFE_MODE:
        return ultra_safe_fit_to_samples(audio.astype(np.float32), target_samples)
    return _time_stretch_ffmpeg_preserve_pitch(audio.astype(np.float32), target_samples)

def select_best_reference_slice(audio: np.ndarray, sample_rate: int, max_seconds: int = MAX_XTTS_REFERENCE_SECONDS) -> np.ndarray:
    target = min(len(audio), sample_rate * max_seconds)
    if len(audio) <= target:
        return audio
    step = sample_rate
    best_score = -1.0
    best_start = 0
    for start in range(0, len(audio) - target + 1, step):
        chunk = audio[start : start + target]
        rms = float(np.sqrt(np.mean(chunk * chunk) + 1e-9))
        silence_ratio = float(np.mean(np.abs(chunk) < 0.01))
        score = rms * (1.0 - silence_ratio)
        if score > best_score:
            best_score = score
            best_start = start
    return audio[best_start : best_start + target]

def load_wav_for_whisper(path: str, target_sr: int = OUTPUT_SAMPLE_RATE) -> np.ndarray:
    audio, sr = load_pcm16_wav(path)
    audio = resample_audio_np(audio, sr, target_sr)
    return preprocess_for_asr(audio)

def polish_for_american_speech(text: str) -> str:
    replacements = {
        " do not ": " don't ",
        " does not ": " doesn't ",
        " did not ": " didn't ",
        " cannot ": " can't ",
        " will not ": " won't ",
        " I am ": " I'm ",
        " we are ": " we're ",
        " they are ": " they're ",
        " it is ": " it's ",
        " that is ": " that's ",
    }
    out = f" {text.strip()} "
    for src, dst in replacements.items():
        out = re.sub(re.escape(src), dst, out, flags=re.IGNORECASE)
    out = re.sub(r"\s+", " ", out).strip()
    out = re.sub(r"([.!?]){2,}", r"\1", out)
    return out

def apply_glossary(text: str) -> str:
    out = text
    for ru, en in GLOSSARY_RU_EN.items():
        out = re.sub(re.escape(ru), en, out, flags=re.IGNORECASE)
    return out

def split_long_text_for_marian(text: str, tokenizer, max_tokens: int = MAX_MARIAN_INPUT_TOKENS):
    words = text.split()
    if not words:
        return []
    chunks, current_words = [], []
    for word in words:
        candidate = " ".join(current_words + [word])
        token_count = len(tokenizer(candidate, add_special_tokens=False)["input_ids"])
        if token_count <= max_tokens:
            current_words.append(word)
        elif current_words:
            chunks.append(" ".join(current_words))
            current_words = [word]
        else:
            chunks.append(word)
    if current_words:
        chunks.append(" ".join(current_words))
    return chunks

def translate_text_ru_to_en(text: str, tokenizer, model) -> str:
    text = apply_glossary(text)
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s", text)
    translated_parts = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        for chunk in split_long_text_for_marian(sent, tokenizer):
            inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=MAX_MARIAN_INPUT_TOKENS)
            translated = model.generate(**inputs, max_new_tokens=256, num_beams=4)
            translated_parts.append(tokenizer.decode(translated[0], skip_special_tokens=True))
    return " ".join(translated_parts).strip()

def clean_segment_text_en(text: str) -> str:
    text = re.sub(r"[♪♫]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    cleaned = []
    for w in words:
        if len(cleaned) >= 3 and cleaned[-1].lower() == cleaned[-2].lower() == cleaned[-3].lower() == w.lower():
            continue
        cleaned.append(w)
    return " ".join(cleaned).strip()

def split_text_with_punctuation(text: str):
    text = clean_segment_text_en(text)
    if not text:
        return []
    parts = [p.strip() for p in re.split(r"(?<=[,;:.!?])\s+", text) if p.strip()]
    return parts if parts else [text]

def refine_segments_for_expression(segments):
    refined = []
    for seg in segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        text_en = clean_segment_text_en(seg.get("text_en", ""))
        text_ru = (seg.get("text_ru", "") or "").strip()
        duration = max(0.0, end - start)
        if not text_en:
            refined.append({"start": start, "end": end, "text_ru": text_ru, "text_en": ""})
            continue
        needs_split = duration > MAX_EXPRESSIVE_SEGMENT_SECONDS or len(text_en) > 240
        if not needs_split:
            refined.append({"start": start, "end": end, "text_ru": text_ru, "text_en": text_en})
            continue
        parts = split_text_with_punctuation(text_en)
        if len(parts) <= 1:
            refined.append({"start": start, "end": end, "text_ru": text_ru, "text_en": text_en})
            continue
        weights = [max(1, len(p)) for p in parts]
        total_w = sum(weights)
        cur = start
        for i, part in enumerate(parts):
            chunk_dur = duration * (weights[i] / total_w) if total_w else (duration / len(parts))
            chunk_dur = max(MIN_EXPRESSIVE_SEGMENT_SECONDS, chunk_dur)
            nxt = min(end, cur + chunk_dur)
            if i == len(parts) - 1:
                nxt = end
            refined.append(
                {
                    "start": cur,
                    "end": max(cur + 0.05, nxt),
                    "text_ru": text_ru if i == 0 else "",
                    "text_en": part,
                }
            )
            cur = nxt
    for i in range(1, len(refined)):
        if refined[i]["start"] < refined[i - 1]["end"]:
            refined[i]["start"] = refined[i - 1]["end"]
        if refined[i]["end"] <= refined[i]["start"]:
            refined[i]["end"] = refined[i]["start"] + 0.05
    return refined

def translate_segments_ru_to_en(segments_ru, tokenizer, model):
    out = []
    prev_ru = ""
    for seg in tqdm(segments_ru, desc="Перевод сегментов", unit="seg", leave=False):
        ru_text = (seg.get("text") or "").strip()
        if ru_text:
            ru_with_context = f"{prev_ru}. {ru_text}".strip(" .") if prev_ru else ru_text
            translated = translate_text_ru_to_en(ru_with_context, tokenizer, model)
            translated_parts = re.split(r"(?<=[.!?])\s+", translated)
            en_text = translated_parts[-1] if translated_parts else translated
            en_text = polish_for_american_speech(en_text)
            prev_ru = ru_text
        else:
            en_text = ""
        out.append(
            {
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text_ru": ru_text,
                "text_en": clean_segment_text_en(en_text),
            }
        )
    return out

def micro_pause_samples_for_clause(clause: str) -> int:
    clause = clause.strip()
    if not clause:
        return 0
    if clause.endswith(","):
        return int(OUTPUT_SAMPLE_RATE * MICRO_PAUSE_COMMA_MS / 1000)
    if clause.endswith(";") or clause.endswith(":"):
        return int(OUTPUT_SAMPLE_RATE * MICRO_PAUSE_SEMICOLON_MS / 1000)
    if clause.endswith(".") or clause.endswith("?") or clause.endswith("!"):
        return int(OUTPUT_SAMPLE_RATE * MICRO_PAUSE_SENTENCE_MS / 1000)
    return int(OUTPUT_SAMPLE_RATE * MICRO_PAUSE_COMMA_MS / 1000)

def split_text_for_tts(text: str):
    parts = [p.strip() for p in re.split(r"(?<=[,;:.!?])\s+", text) if p.strip()]
    return parts if parts else [text.strip()]

Qwen3TTS_MODEL = None
Qwen3TTS_TOKENIZER = None

def _snapshot_download_windows_no_symlink(repo_id: str, cache_dir: str, token: str | None):
    """HF snapshot download без symlink для Windows (WinError 1314)."""
    safe_repo = repo_id.replace("/", "--")
    local_dir = os.path.join(cache_dir, "no_symlink_snapshots", safe_repo)
    os.makedirs(local_dir, exist_ok=True)
    kwargs = dict(
        repo_id=repo_id,
        cache_dir=cache_dir,
        local_dir=local_dir,
        local_files_only=False,
    )
    if token:
        kwargs["token"] = token
    return snapshot_download(**kwargs)

def _resolve_qwen_model_for_runtime() -> str:
    """Returns model id selected by ENABLE_QWEN_SIZE / QWEN_LOCAL_MODEL_ID."""
    model_id = QWEN_LOCAL_MODEL_ID
    has_cuda = bool(torch.cuda.is_available())
    if not has_cuda and "1.7B-Base" in model_id:
        print("[LOCAL][WARN] Выбрана 1.7B на CPU — это может быть очень медленно или нестабильно.")
        print("[LOCAL][WARN] Для более стабильной работы задайте ENABLE_QWEN_SIZE=0.6")
    return model_id


def load_local_qwen_tts():
    global Qwen3TTS_MODEL, Qwen3TTS_TOKENIZER
    if Qwen3TTS_MODEL is not None:
        return Qwen3TTS_MODEL, Qwen3TTS_TOKENIZER
    token = os.environ.get("HF_TOKEN")
    try:
        qwen_tts_mod = importlib.import_module("qwen_tts")
        Qwen3TTSModel = qwen_tts_mod.Qwen3TTSModel
        Qwen3TTSTokenizer = qwen_tts_mod.Qwen3TTSTokenizer
        runtime_model_id = _resolve_qwen_model_for_runtime()
        print(f"[LOCAL] Загружаем Qwen3-TTS: {runtime_model_id}")
        cache_dir = os.environ.get("HF_HOME", os.path.join(WORK_DIR, "hf_cache"))
        os.makedirs(cache_dir, exist_ok=True)
        print(f"[LOCAL] cache dir: {cache_dir}")
        print("[LOCAL] start resolve model path...")
        t_snap = time.time()
        if token:
            model_local_path = _snapshot_download_windows_no_symlink(
                repo_id=runtime_model_id,
                cache_dir=cache_dir,
                token=token,
            )
            print(f"[LOCAL] snapshot_download(model) done in {time.time() - t_snap:.1f} сек")
        else:
            model_local_path = _find_local_hf_snapshot(cache_dir, runtime_model_id)
            if model_local_path:
                print(f"[LOCAL] HF_TOKEN не задан, используем локальный snapshot модели ({time.time() - t_snap:.1f} сек)")
            else:
                print("[LOCAL] HF_TOKEN не задан, пробуем скачать модель анонимно...")
                model_local_path = _snapshot_download_windows_no_symlink(
                    repo_id=runtime_model_id,
                    cache_dir=cache_dir,
                    token=None,
                )
                print(f"[LOCAL] snapshot_download(model, anonymous) done in {time.time() - t_snap:.1f} сек")
        print(f"[LOCAL] model local path: {model_local_path}")
        t0 = time.time()
        print("[LOCAL] start model.from_pretrained(...)")
        _dtype_map = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        selected_dtype = _dtype_map.get(QWEN_LOCAL_DTYPE.lower(), torch.float16)
        has_cuda = bool(torch.cuda.is_available())
        device_map = "auto" if has_cuda else None
        low_cpu_mem_usage = QWEN_LOW_CPU_MEM_USAGE if has_cuda else False
        if not has_cuda and selected_dtype in (torch.float16, torch.bfloat16):
            print(
                f"[LOCAL][WARN] dtype={selected_dtype} на CPU может вызывать краш/выход процесса; "
                "переключаемся на torch.float32"
            )
            selected_dtype = torch.float32
        if not has_cuda:
            try:
                torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
            except Exception:
                pass
        print(f"[LOCAL] cuda={has_cuda}, device_map={device_map}, dtype={selected_dtype}, low_cpu_mem_usage={low_cpu_mem_usage}")
        model_kwargs = dict(
            token=token,
            dtype=selected_dtype,
            trust_remote_code=QWEN_LOCAL_TRUST_REMOTE_CODE,
            local_files_only=True,
            low_cpu_mem_usage=low_cpu_mem_usage,
            attn_implementation="eager",
        )
        if device_map is not None:
            model_kwargs["device_map"] = device_map
        model = Qwen3TTSModel.from_pretrained(
            model_local_path,
            **model_kwargs,
        )
        print(f"[LOCAL] model loaded in {time.time() - t0:.1f} сек")
        t1 = time.time()
        print("[LOCAL] start tokenizer.from_pretrained(...)")
        tokenizer_local_id = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
        print("[LOCAL] start resolve tokenizer path...")
        t_tok_snap = time.time()
        if token:
            tokenizer_local_path = _snapshot_download_windows_no_symlink(
                repo_id=tokenizer_local_id,
                cache_dir=cache_dir,
                token=token,
            )
            print(f"[LOCAL] snapshot_download(tokenizer) done in {time.time() - t_tok_snap:.1f} сек")
        else:
            tokenizer_local_path = _find_local_hf_snapshot(cache_dir, tokenizer_local_id)
            if tokenizer_local_path:
                print(f"[LOCAL] HF_TOKEN не задан, используем локальный snapshot токенайзера ({time.time() - t_tok_snap:.1f} сек)")
            else:
                print("[LOCAL] HF_TOKEN не задан, пробуем скачать токенайзер анонимно...")
                tokenizer_local_path = _snapshot_download_windows_no_symlink(
                    repo_id=tokenizer_local_id,
                    cache_dir=cache_dir,
                    token=None,
                )
                print(f"[LOCAL] snapshot_download(tokenizer, anonymous) done in {time.time() - t_tok_snap:.1f} сек")
        tokenizer = Qwen3TTSTokenizer.from_pretrained(
            tokenizer_local_path,
            token=token,
            local_files_only=True,
        )
        print(f"[LOCAL] tokenizer loaded in {time.time() - t1:.1f} сек")
        Qwen3TTS_MODEL = model
        Qwen3TTS_TOKENIZER = tokenizer
        print("[OK] Qwen3-TTS успешно загружен")
        return model, tokenizer
    except Exception as e:
        print("[LOCAL][ERROR] traceback:")
        traceback.print_exc()
        raise RuntimeError(
            f"Ошибка загрузки Qwen3-TTS: {e}\n"
            "Проверьте: 1) pip install -U qwen-tts\n"
            "2) HF_TOKEN\n"
            "3) интернет\n"
            "4) место на диске\n"
            "5) sox/ffmpeg/ffprobe в PATH\n"
            "6) проверьте, что HF cache не уходит в C:\\Users\\...\\.cache"
        )

def _find_local_hf_snapshot(cache_dir: str, repo_id: str) -> str | None:
    """Ищет локальный snapshot HF-модели без сети/токена."""
    safe_repo = repo_id.replace("/", "--")
    root = os.path.join(cache_dir, f"models--{safe_repo}", "snapshots")
    if not os.path.exists(root):
        return None
    candidates = []
    for name in os.listdir(root):
        pth = os.path.join(root, name)
        if os.path.isdir(pth):
            candidates.append((os.path.getmtime(pth), pth))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]

def _run_cmd_live(cmd: list[str], title: str, cwd: str | None = None, env: dict | None = None):
    print(f"[CMD] {title}")
    print("[CMD] " + " ".join(cmd))
    if cwd:
        print(f"[CMD] cwd={cwd}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        cwd=cwd,
        env=env,
    )
    assert proc.stdout is not None
    tail_lines = []
    for line in proc.stdout:
        clean = line.rstrip()
        print(clean)
        tail_lines.append(clean)
        if len(tail_lines) > 120:
            tail_lines.pop(0)
    code = proc.wait()
    if code != 0:
        tail = "\n".join(tail_lines[-40:])
        raise RuntimeError(f"Команда завершилась с кодом {code}: {title}\n{tail}")

def _rvc_paths_for_name(voice_name: str):
    voice_dir = os.path.join(RVC_MODELS_DIR, voice_name)
    model_path = os.path.join(voice_dir, f"{voice_name}.pth")
    index_path = os.path.join(voice_dir, f"{voice_name}.index")
    return voice_dir, model_path, index_path

def _download_file(url: str, dst_path: str):
    print(f"[RVC] Скачиваем: {url} -> {dst_path}")
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with urllib.request.urlopen(url) as resp, open(dst_path, "wb") as f:
        f.write(resp.read())

def _find_train_like_scripts() -> list[str]:
    out = []
    if not os.path.exists(RVC_REPO_DIR):
        return out
    for root, _, files in os.walk(RVC_REPO_DIR):
        for fn in files:
            low = fn.lower()
            if low.endswith('.py') and ('train' in low):
                out.append(os.path.join(root, fn))
    return sorted(out)

def _find_infer_like_scripts() -> list[str]:
    out = []
    if not os.path.exists(RVC_REPO_DIR):
        return out
    for root, _, files in os.walk(RVC_REPO_DIR):
        for fn in files:
            low = fn.lower()
            if low.endswith('.py') and ('infer' in low):
                out.append(os.path.join(root, fn))
    return sorted(out)

def ensure_rvc_scripts_ready():
    os.makedirs(RVC_REPO_DIR, exist_ok=True)
    train_script = _discover_rvc_train_script()
    infer_script = _discover_rvc_infer_script()
    if train_script and infer_script:
        print(f"[RVC] scripts готовы: train={train_script}, infer={infer_script}")
        return train_script, infer_script
    if not RVC_AUTO_INSTALL_SCRIPTS:
        raise RuntimeError(
            "RVC scripts не найдены, а авто-установка отключена.\n"
            f"Ожидается train script (например {RVC_TRAIN_SCRIPT}) и infer script (например {RVC_INFER_SCRIPT})."
        )
    print("[RVC] Скрипты не найдены. Пытаемся скачать RVC-репозиторий...")
    git_dir = os.path.join(RVC_REPO_DIR, ".git")
    if not os.path.exists(git_dir):
        _run_cmd_live([
            "git", "clone", "--depth", "1", "--branch", RVC_GIT_BRANCH, RVC_GIT_REPO, RVC_REPO_DIR
        ], "RVC git clone")
    else:
        _run_cmd_live(["git", "-C", RVC_REPO_DIR, "fetch", "origin", RVC_GIT_BRANCH], "RVC git fetch")
        _run_cmd_live(["git", "-C", RVC_REPO_DIR, "checkout", RVC_GIT_BRANCH], "RVC git checkout")
        _run_cmd_live(["git", "-C", RVC_REPO_DIR, "pull", "origin", RVC_GIT_BRANCH], "RVC git pull")
    train_script = _discover_rvc_train_script()
    infer_script = _discover_rvc_infer_script()
    if (not train_script or not infer_script) and (RVC_TRAIN_SCRIPT_URL or RVC_INFER_SCRIPT_URL):
        try:
            if not train_script and RVC_TRAIN_SCRIPT_URL:
                _download_file(RVC_TRAIN_SCRIPT_URL, RVC_TRAIN_SCRIPT)
            if not infer_script and RVC_INFER_SCRIPT_URL:
                _download_file(RVC_INFER_SCRIPT_URL, RVC_INFER_SCRIPT)
        except Exception as e:
            print(f"[RVC][WARN] Не удалось скачать custom RVC scripts: {e}")
        train_script = _discover_rvc_train_script()
        infer_script = _discover_rvc_infer_script()
    if not train_script or not infer_script:
        train_like = _find_train_like_scripts()[:40]
        infer_like = _find_infer_like_scripts()[:40]
        raise RuntimeError(
            "После авто-установки RVC нужные скрипты не найдены.\n"
            f"train: {train_script}\n"
            f"infer: {infer_script}\n\n"
            "Вариант 1 (рекомендуется): положите совместимые entrypoint-скрипты в:\n"
            f"  - {RVC_TRAIN_SCRIPT}\n"
            f"  - {RVC_INFER_SCRIPT}\n\n"
            "Вариант 2: задайте URL для авто-скачивания:\n"
            "  - RVC_TRAIN_SCRIPT_URL\n"
            "  - RVC_INFER_SCRIPT_URL\n\n"
            f"Найденные train-like файлы (до 40): {train_like}\n"
            f"Найденные infer-like файлы (до 40): {infer_like}"
        )
    print(f"[RVC] scripts готовы после установки: train={train_script}, infer={infer_script}")
    print("[RVC] Если train script не является CLI-скриптом, задайте явный путь через RVC_TRAIN_SCRIPT.")
    return train_script, infer_script

def _discover_rvc_train_script() -> str | None:
    candidates = [
        RVC_TRAIN_SCRIPT,
        os.path.join(RVC_REPO_DIR, "train.py"),
        os.path.join(RVC_REPO_DIR, "train", "train.py"),
        os.path.join(RVC_REPO_DIR, "scripts", "train_rvc.py"),
        os.path.join(RVC_REPO_DIR, "tools", "train_rvc.py"),
        os.path.join(RVC_REPO_DIR, "infer", "modules", "train", "train.py"),
        os.path.join(RVC_REPO_DIR, "tools", "infer", "train-index.py"),
        os.path.join(RVC_REPO_DIR, "tools", "infer", "train-index-v2.py"),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None

def _discover_rvc_infer_script() -> str | None:
    candidates = [
        RVC_INFER_SCRIPT,
        os.path.join(RVC_REPO_DIR, "infer.py"),
        os.path.join(RVC_REPO_DIR, "infer", "infer.py"),
        os.path.join(RVC_REPO_DIR, "scripts", "infer_rvc.py"),
        os.path.join(RVC_REPO_DIR, "tools", "infer_cli.py"),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None

def _find_latest_file(root: str, exts: tuple[str, ...], name_hint: str = "") -> str | None:
    candidates = []
    if not os.path.exists(root):
        return None
    for r, _, files in os.walk(root):
        for fn in files:
            low = fn.lower()
            if low.endswith(exts):
                p = os.path.join(r, fn)
                score = 1 if (name_hint and name_hint.lower() in low) else 0
                mtime = os.path.getmtime(p)
                candidates.append((score, mtime, p))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][2]

def _find_rvc_config_candidates() -> list[str]:
    patterns = [
        os.path.join(RVC_REPO_DIR, "configs", "**", "*.json"),
        os.path.join(RVC_REPO_DIR, "infer", "modules", "train", "configs", "**", "*.json"),
    ]
    out = []
    for pat in patterns:
        out.extend(glob.glob(pat, recursive=True))
    seen = set()
    uniq = []
    for x in out:
        nx = os.path.normpath(x)
        if nx not in seen:
            seen.add(nx)
            uniq.append(nx)
    return uniq

def _find_rvc_config_template(sample_rate: str, version: str) -> str | None:
    sr = sample_rate.lower().strip()
    ver = version.lower().strip()
    preferred = [
        os.path.join(RVC_REPO_DIR, "configs", ver, f"{sr}.json"),
        os.path.join(RVC_REPO_DIR, "configs", f"{sr}.json"),
        os.path.join(RVC_REPO_DIR, "infer", "modules", "train", "configs", ver, f"{sr}.json"),
        os.path.join(RVC_REPO_DIR, "infer", "modules", "train", "configs", f"{sr}.json"),
    ]
    for pth in preferred:
        if os.path.exists(pth):
            return pth
    all_cfg = _find_rvc_config_candidates()
    for pth in all_cfg:
        low = pth.lower()
        if sr in low and ver in low:
            return pth
    for pth in all_cfg:
        if sr in pth.lower():
            return pth
    return None

def _ensure_rvc_filelist(exp_dir: str, dataset_wav: str):
    """Создаёт корректный filelist.txt для RVC train.py (5 колонок)."""
    os.makedirs(exp_dir, exist_ok=True)
    filelist_path = os.path.join(exp_dir, "filelist.txt")
    wav_norm = os.path.normpath(dataset_wav)

    pitch_path = os.path.join(exp_dir, "dummy_pitch.npy")
    pitchf_path = os.path.join(exp_dir, "dummy_pitchf.npy")
    if not os.path.exists(pitch_path):
        np.save(pitch_path, np.zeros(16, dtype=np.int64))
    if not os.path.exists(pitchf_path):
        np.save(pitchf_path, np.zeros(16, dtype=np.float32))

    def _build_lines():
        # Формат RVC (ожидается 5 значений при распаковке):
        # audiopath|text|pitch.npy|pitchf.npy|speaker_id
        return [
            f"{wav_norm}|a|{os.path.normpath(pitch_path)}|{os.path.normpath(pitchf_path)}|0",
            f"{wav_norm}|b|{os.path.normpath(pitch_path)}|{os.path.normpath(pitchf_path)}|0",
        ]

    need_rewrite = True
    if os.path.exists(filelist_path):
        try:
            with open(filelist_path, "r", encoding="utf-8") as f:
                rows = [x.strip() for x in f if x.strip()]
            if rows and all(len(r.split("|")) >= 5 for r in rows):
                need_rewrite = False
        except Exception:
            need_rewrite = True

    if need_rewrite:
        lines = _build_lines()
        with open(filelist_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[RVC] filelist создан/обновлён: {filelist_path} (entries={len(lines)}, cols=5)")
    else:
        print(f"[RVC] filelist уже существует и валиден: {filelist_path}")

def _prepare_rvc_train_experiment_dir(voice_name: str) -> str:
    exp_dir = os.path.join(RVC_WORK_DIR, "logs", voice_name)
    os.makedirs(exp_dir, exist_ok=True)
    config_save_path = os.path.join(exp_dir, "config.json")
    if os.path.exists(config_save_path):
        return exp_dir
    tpl = _find_rvc_config_template(RVC_SAMPLE_RATE, RVC_MODEL_VERSION)
    if not tpl:
        found = _find_rvc_config_candidates()[:80]
        raise RuntimeError(
            "RVC train требует config.json в experiment_dir, но шаблон конфига не найден.\n"
            f"Ожидали sample_rate={RVC_SAMPLE_RATE}, version={RVC_MODEL_VERSION}.\n"
            f"Создайте вручную: {config_save_path}\n"
            f"или положите config-шаблоны в {os.path.join(RVC_REPO_DIR, 'configs')}.\n"
            f"Найденные config-файлы (до 80): {found}"
        )
    shutil.copy2(tpl, config_save_path)
    print(f"[RVC] config template: {tpl} -> {config_save_path}")
    return exp_dir

def _sync_rvc_artifacts_to_voice_dir(voice_name: str, voice_dir: str):
    os.makedirs(voice_dir, exist_ok=True)
    found_pth = _find_latest_file(RVC_WORK_DIR, (".pth",), voice_name)
    found_index = _find_latest_file(RVC_WORK_DIR, (".index",), voice_name)
    if found_pth:
        dst_pth = os.path.join(voice_dir, f"{voice_name}.pth")
        if os.path.abspath(found_pth) != os.path.abspath(dst_pth):
            shutil.copy2(found_pth, dst_pth)
        print(f"[RVC] model artifact: {found_pth} -> {dst_pth}")
    if found_index:
        dst_idx = os.path.join(voice_dir, f"{voice_name}.index")
        if os.path.abspath(found_index) != os.path.abspath(dst_idx):
            shutil.copy2(found_index, dst_idx)
        print(f"[RVC] index artifact: {found_index} -> {dst_idx}")

# ======================== ИСПРАВЛЕННЫЕ ФУНКЦИИ ДЛЯ RVC ========================
def _prepare_rvc_train_env():
    env = os.environ.copy()
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = "29501"
    env["WORLD_SIZE"] = "1"
    env["RANK"] = "0"
    env["LOCAL_RANK"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = RVC_TRAIN_GPUS if str(RVC_TRAIN_GPUS).strip() else "-1"
    env["USE_LIBUV"] = "0"
    env["TORCH_USE_LIBUV"] = "0"
    env["C10D_USE_LIBUV"] = "0"
    env["TORCH_DISTRIBUTED_DEBUG"] = "OFF"
    # ВАЖНО: на Windows GLOO иногда не может выбрать сетевое устройство автоматически.
    # Можно задать интерфейс явно через RVC_GLOO_SOCKET_IFNAME (например Ethernet/Wi-Fi).
    env.pop("GLOO_DEVICE_TRANSPORT", None)
    if RVC_GLOO_SOCKET_IFNAME:
        env["GLOO_SOCKET_IFNAME"] = RVC_GLOO_SOCKET_IFNAME
    else:
        env.pop("GLOO_SOCKET_IFNAME", None)
    env["NCCL_IB_DISABLE"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    print(f"[RVC] RVC_TRAIN_GPUS={RVC_TRAIN_GPUS} | CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
    print(f"[RVC] USE_LIBUV = 0 | TORCH_USE_LIBUV = 0")
    print(f"[RVC] MASTER_ADDR={env['MASTER_ADDR']}, GLOO_SOCKET_IFNAME={env.get('GLOO_SOCKET_IFNAME', '<auto>')}")
    return env

def _build_rvc_train_cmd(train_script: str, dataset_wav: str, voice_name: str):
    """Один стабильный путь запуска на Windows (без fallback).

    ВАЖНО: запускаем train.py напрямую БЕЗ torch.distributed.run,
    т.к. torchrun на некоторых Windows-сборках PyTorch всё равно
    пытается создать TCPStore с libuv и падает.
    """
    exp_dir = _prepare_rvc_train_experiment_dir(voice_name)
    return [
        RVC_PYTHON,
        train_script,
        "-se", str(RVC_SAVE_EVERY_EPOCH),
        "-te", str(RVC_EPOCHS),
        "-bs", str(RVC_BATCH_SIZE),
        "-e", exp_dir,
        "-sr", RVC_SAMPLE_RATE,
        "-v", RVC_MODEL_VERSION,
        "-g", RVC_TRAIN_GPUS,
        "-f0", str(RVC_IF_F0),
        "-l", str(RVC_IF_LATEST),
        "-c", str(RVC_IF_CACHE_GPU),
    ]


def _run_rvc_train_with_windows_retries(cmd: list[str], cwd: str, base_env: dict, voice_name: str):
    """Windows retry loop for ProcessGroupGloo init + проверка артефактов.

    Некоторые сборки PyTorch на Windows падают с
    `makeDeviceForHostname(): unsupported gloo device`.
    Делаем ограниченный перебор MASTER_ADDR/GLOO_SOCKET_IFNAME.
    Важно: train.py может завершиться кодом 0, но дочерний процесс упасть,
    поэтому после каждой попытки обязательно проверяем наличие .pth.
    """
    master_candidates = [
        base_env.get("MASTER_ADDR", "127.0.0.1"),
        "127.0.0.1",
        "localhost",
    ]
    master_candidates = list(dict.fromkeys(master_candidates))

    gloo_candidates = []
    forced = (base_env.get("GLOO_SOCKET_IFNAME") or "").strip()
    if forced:
        gloo_candidates.append(forced)
    gloo_candidates.extend([
        None,
        "Loopback Pseudo-Interface 1",
        "Ethernet",
        "Wi-Fi",
        "vEthernet (Default Switch)",
    ])
    gloo_candidates = list(dict.fromkeys(gloo_candidates))

    attempts_all = [(master, gloo_if) for master in master_candidates for gloo_if in gloo_candidates]
    max_attempts = max(1, int(RVC_TRAIN_MAX_ATTEMPTS))
    attempts = attempts_all[:max_attempts]

    last_err = None
    for i, (master, gloo_if) in enumerate(attempts, 1):
        env = base_env.copy()
        env["MASTER_ADDR"] = master
        if gloo_if:
            env["GLOO_SOCKET_IFNAME"] = gloo_if
        else:
            env.pop("GLOO_SOCKET_IFNAME", None)

        print(f"[RVC] train attempt {i}/{len(attempts)}: MASTER_ADDR={master}, GLOO_SOCKET_IFNAME={env.get('GLOO_SOCKET_IFNAME', '<auto>')}")
        try:
            _run_cmd_live(cmd, f"RVC train [attempt {i}]", cwd=cwd, env=env)

            # Проверка фактического результата обучения (не только кода возврата).
            found_pth = _find_latest_file(RVC_WORK_DIR, (".pth",), voice_name)
            if found_pth and os.path.exists(found_pth):
                print(f"[RVC] train attempt succeeded, model artifact found: {found_pth}")
                return

            last_err = RuntimeError("train process exited without model artifact (.pth)")
            print(f"[RVC][WARN] train attempt {i} finished but no .pth artifact found")
        except Exception as e:
            last_err = e
            print(f"[RVC][WARN] train attempt {i} failed: {e}")
            continue

    raise RuntimeError(f"RVC train failed after {len(attempts)} attempts. Last error: {last_err}")
# ======================== КОНЕЦ ИСПРАВЛЕНИЙ ========================

def train_rvc_voice_model(voice_name: str, source_wav_path: str):
    global CURRENT_RVC_MODEL_PATH, CURRENT_RVC_INDEX_PATH, RVC_ENABLED
    voice_dir, model_path, index_path = _rvc_paths_for_name(voice_name)
    os.makedirs(voice_dir, exist_ok=True)
    if os.path.exists(model_path):
        print(f"[RVC] ✓ Модель уже существует: {model_path}")
        CURRENT_RVC_MODEL_PATH = model_path
        CURRENT_RVC_INDEX_PATH = index_path if os.path.exists(index_path) else ""
        return model_path, index_path
    if not RVC_AUTO_TRAIN:
        raise RuntimeError("RVC_AUTO_TRAIN=False и модель не найдена.")
    print("[RVC] Подготовка датасета и запуск обучения...")
    dataset_dir = os.path.join(RVC_DATASET_DIR, voice_name)
    os.makedirs(dataset_dir, exist_ok=True)
    dataset_wav = os.path.join(dataset_dir, f"{voice_name}_source.wav")
    if not os.path.exists(dataset_wav):
        subprocess.run([
            os.environ.get("IMAGEIO_FFMPEG_EXE", "ffmpeg"),
            "-y", "-i", source_wav_path,
            "-ar", "40000", "-ac", "1",
            dataset_wav
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    train_script, _ = ensure_rvc_scripts_ready()
    rvc_env = _prepare_rvc_train_env()
    cmd = _build_rvc_train_cmd(train_script, dataset_wav, voice_name)
    exp_dir = os.path.join(RVC_WORK_DIR, "logs", voice_name)
    _ensure_rvc_filelist(exp_dir, dataset_wav)
    print("[RVC] Режим запуска train: direct python script (без torchrun)")
    try:
        _run_rvc_train_with_windows_retries(cmd, cwd=RVC_REPO_DIR, base_env=rvc_env, voice_name=voice_name)
    except Exception as e:
        print("[RVC][ERROR] Обучение не удалось после всех попыток.")
        print(f"[RVC][ERROR] {e}")
        if RVC_TRAIN_REQUIRED:
            raise
        print("[RVC][WARN] Продолжаем БЕЗ RVC-конверсии (чистый Qwen3-TTS), т.к. RVC_TRAIN_REQUIRED=false")
        RVC_ENABLED = False
        CURRENT_RVC_MODEL_PATH = None
        CURRENT_RVC_INDEX_PATH = None
        return "", ""
    _sync_rvc_artifacts_to_voice_dir(voice_name, voice_dir)
    if not os.path.exists(model_path):
        weights_dir = os.path.join(RVC_WORK_DIR, "logs", voice_name, "weights")
        latest_pth = _find_latest_file(weights_dir, (".pth",), voice_name)
        if latest_pth:
            shutil.copy2(latest_pth, model_path)
            print(f"[RVC] Модель найдена в weights/ и скопирована")
    if not os.path.exists(model_path):
        raise RuntimeError(f"RVC model не создан: {model_path}")
    CURRENT_RVC_MODEL_PATH = model_path
    CURRENT_RVC_INDEX_PATH = index_path if os.path.exists(index_path) else ""
    print(f"[RVC] ✓ Обучение завершено → {model_path}")
    return model_path, index_path

def _ensure_rvc_py311_compat_dir() -> str:
    """Создаёт sitecustomize.py с фиксом urllib.quote для старых зависимостей на py3."""
    compat_dir = os.path.join(RVC_TMP_DIR, "py_compat")
    os.makedirs(compat_dir, exist_ok=True)
    sitecustomize_path = os.path.join(compat_dir, "sitecustomize.py")
    content = (
        "# auto-generated for RVC compatibility on py311\n"
        "import urllib\n"
        "import urllib.parse\n"
        "if not hasattr(urllib, 'quote'):\n"
        "    urllib.quote = urllib.parse.quote\n"
    )
    if not os.path.exists(sitecustomize_path) or open(sitecustomize_path, 'r', encoding='utf-8').read() != content:
        with open(sitecustomize_path, 'w', encoding='utf-8') as f:
            f.write(content)
    return compat_dir


def _check_rvc_infer_dependencies(rvc_env: dict) -> tuple[list[str], list[str]]:
    """Проверяет зависимости RVC infer: что реально не установлено и что падает при import."""
    required = [
        ("dotenv", "python-dotenv"),
        ("ffmpeg", "ffmpeg-python"),
        ("av", "av"),
        ("fairseq", "fairseq"),
        ("parselmouth", "praat-parselmouth"),
        ("faiss", "faiss-cpu"),
        ("librosa", "librosa"),
        ("soundfile", "soundfile"),
    ]
    missing = []
    import_issues = []
    for mod, pip_name in required:
        # 1) Проверяем наличие пакета (без выполнения импорта)
        spec_cmd = (
            "import importlib.util,sys; "
            f"sys.exit(0 if importlib.util.find_spec('{mod}') else 1)"
        )
        spec_chk = subprocess.run(
            [RVC_INFER_PYTHON, "-c", spec_cmd],
            capture_output=True,
            text=True,
            env=rvc_env,
        )
        has_spec = spec_chk.returncode == 0

        # 2) Пытаемся импортировать (может упасть по runtime-конфликту)
        imp_chk = subprocess.run(
            [RVC_INFER_PYTHON, "-c", f"import {mod}"],
            capture_output=True,
            text=True,
            env=rvc_env,
        )
        can_import = imp_chk.returncode == 0

        if not has_spec:
            missing.append(f"{mod} (pip install {pip_name})")
            continue

        if has_spec and not can_import:
            err = (imp_chk.stdout or "") + (imp_chk.stderr or "")
            err = err.strip().splitlines()
            tail = err[-1] if err else "unknown import error"
            import_issues.append(f"{mod}: {tail}")

    return missing, import_issues


def _build_rvc_infer_cmd(infer_script: str, input_wav: str, output_wav: str, env: dict) -> list[str]:
    """Собирает команду infer_cli под конкретную версию RVC (новый/старый CLI)."""
    base = [RVC_INFER_PYTHON, infer_script]
    help_text = ""
    try:
        h = subprocess.run(base + ["-h"], capture_output=True, text=True, env=env)
        help_text = (h.stdout or "") + "\n" + (h.stderr or "")
    except Exception:
        help_text = ""

    # Вариант 1: наш текущий CLI с --input_wav/--output_wav/--model
    if "--input_wav" in help_text and "--output_wav" in help_text and "--model" in help_text:
        cmd = [
            *base,
            "--input_wav", input_wav,
            "--output_wav", output_wav,
            "--model", CURRENT_RVC_MODEL_PATH,
            "--f0_method", RVC_F0_METHOD,
            "--pitch", str(RVC_PITCH_SHIFT),
            "--index_rate", str(RVC_INDEX_RATE),
            "--filter_radius", str(RVC_FILTER_RADIUS),
            "--protect", str(RVC_PROTECT),
            "--autotune", str(RVC_AUTOTUNE).lower(),
        ]
        if CURRENT_RVC_INDEX_PATH:
            cmd += ["--index", CURRENT_RVC_INDEX_PATH]
        return cmd

    # Вариант 2: классический RVC CLI с --input_path/--opt_path/--model_name
    if "--input_path" in help_text and "--opt_path" in help_text and "--model_name" in help_text:
        cmd = [
            *base,
            "--input_path", input_wav,
            "--opt_path", output_wav,
            "--model_name", CURRENT_RVC_MODEL_PATH,
            "--f0method", RVC_F0_METHOD,
            "--f0up_key", str(RVC_PITCH_SHIFT),
            "--index_rate", str(RVC_INDEX_RATE),
            "--filter_radius", str(RVC_FILTER_RADIUS),
            "--protect", str(RVC_PROTECT),
        ]
        if CURRENT_RVC_INDEX_PATH:
            cmd += ["--index_path", CURRENT_RVC_INDEX_PATH]
        return cmd

    # Фолбэк: старый формат из этого скрипта
    cmd = [
        *base,
        "--input_wav", input_wav,
        "--output_wav", output_wav,
        "--model", CURRENT_RVC_MODEL_PATH,
        "--f0_method", RVC_F0_METHOD,
        "--pitch", str(RVC_PITCH_SHIFT),
        "--index_rate", str(RVC_INDEX_RATE),
        "--filter_radius", str(RVC_FILTER_RADIUS),
        "--protect", str(RVC_PROTECT),
        "--autotune", str(RVC_AUTOTUNE).lower(),
    ]
    if CURRENT_RVC_INDEX_PATH:
        cmd += ["--index", CURRENT_RVC_INDEX_PATH]
    return cmd


def apply_rvc_conversion(input_wav: str, output_wav: str):
    global RVC_ENABLED
    if not RVC_ENABLED:
        raise RuntimeError("RVC_ENABLED=false, но запрошена RVC-конверсия. Включите RVC или отключите вызов apply_rvc_conversion.")
    if not CURRENT_RVC_MODEL_PATH or not os.path.exists(CURRENT_RVC_MODEL_PATH):
        raise RuntimeError("RVC-модель не инициализирована. Сначала вызовите train_rvc_voice_model(...).")

    compat_dir = _ensure_rvc_py311_compat_dir()
    rvc_env = os.environ.copy()
    old_pp = rvc_env.get("PYTHONPATH", "")
    pp_items = [compat_dir, RVC_REPO_DIR]
    if old_pp:
        pp_items.append(old_pp)
    rvc_env["PYTHONPATH"] = os.pathsep.join(pp_items)

    missing, import_issues = _check_rvc_infer_dependencies(rvc_env)
    if missing:
        msg = "[RVC][ERROR] Не хватает модулей для RVC infer в выбранном Python:\n- " + "\n- ".join(missing)
        if RVC_STRICT_INFER_DEPS:
            raise RuntimeError(msg)
        print(msg)
    if import_issues:
        print("[RVC][WARN] Обнаружены проблемы runtime-импорта в env RVC (пакет есть, но import падает):")
        for item in import_issues:
            print(f"[RVC][WARN] - {item}")

        issues_blob = "\n".join(import_issues).lower()
        if (
            "winerror 1455" in issues_blob
            or "paging file is too small" in issues_blob
            or "llvmlite.dll" in issues_blob
        ):
            raise RuntimeError(
                "[RVC][ERROR] В env RVC не хватает виртуальной памяти Windows (WinError 1455), "
                "из-за этого не грузится llvmlite/numba/librosa для infer.\n"
                "Сделайте: System Properties -> Performance -> Advanced -> Virtual memory и выставьте "
                "System managed size или вручную минимум 16-32 GB, затем перезагрузите Windows.\n"
                "После перезагрузки проверьте в rvc_infer env:\n"
                "python -c \"import llvmlite, numba, librosa; print('OK')\""
            )


    _, infer_script = ensure_rvc_scripts_ready()
    print(f"[RVC] infer script: {infer_script}")
    cmd = _build_rvc_infer_cmd(
        infer_script=infer_script,
        input_wav=input_wav,
        output_wav=output_wav,
        env=rvc_env,
    )
    try:
        _run_cmd_live(cmd, "RVC infer", cwd=RVC_REPO_DIR, env=rvc_env)
    except Exception as e:
        emsg = str(e)
        if (
            "cannot import name 'quote' from 'urllib'" in emsg
            or "ImportError: cannot import name 'quote' from 'urllib'" in emsg
            or "No module named 'parselmouth'" in emsg
            or 'No module named "parselmouth"' in emsg
        ):
            raise RuntimeError(
                "[RVC][ERROR] RVC infer упал на parselmouth. Сделайте в этом же env команды:\n"
                "1) pip uninstall -y parselmouth\n"
                "2) pip uninstall -y praat-parselmouth\n"
                "3) pip install -U praat-parselmouth\n"
                "4) python -c \"import parselmouth; print(parselmouth.__file__)\"\n"
                "Если после этого снова ошибка urllib.quote — используйте Python 3.10 env для RVC.\n\n"
                + emsg
            )
        if (
            "DLL load failed while importing parselmouth" in emsg
            and "paging file is too small" in emsg.lower()
        ):
            raise RuntimeError(
                "[RVC][ERROR] parselmouth не загрузился из-за нехватки виртуальной памяти Windows (paging file too small).\n"
                "Увеличьте файл подкачки (System managed или вручную >= 16-32 GB), перезагрузите Windows и повторите запуск.\n"
                "Проверка после перезагрузки:\n"
                "python -c \"import parselmouth; print('OK')\"\n\n"
                + emsg
            )

        if (
            "fairseq.dataclass.configs.CommonConfig" in emsg
            or "mutable default <class 'fairseq.dataclass.configs.CommonConfig'>" in emsg
        ):
            raise RuntimeError(
                "[RVC][ERROR] RVC infer падает из-за несовместимости fairseq с Python 3.11.\n"
                "Запустите ИМЕННО infer через Python 3.10 и задайте переменную:\n"
                "set RVC_INFER_PYTHON=C:\\path\\to\\py310\\python.exe\n"
                "В py310 env установите: fairseq, av, ffmpeg-python, praat-parselmouth.\n\n"
                + emsg
            )
        raise

def _call_with_supported_kwargs(fn, kwargs: dict):
    """Вызывает функцию, передавая только поддерживаемые kwargs."""
    try:
        sig = inspect.signature(fn)
        accepted = set(sig.parameters.keys())
        filtered = {k: v for k, v in kwargs.items() if k in accepted}
    except Exception:
        filtered = kwargs
    return fn(**filtered)


def _normalize_qwen_language(lang: str) -> str:
    """Qwen language normalization (e.g. en->english, ru->russian)."""
    m = {
        "en": "english",
        "ru": "russian",
        "zh": "chinese",
        "ja": "japanese",
        "ko": "korean",
        "de": "german",
        "fr": "french",
        "it": "italian",
        "pt": "portuguese",
        "es": "spanish",
        "auto": "auto",
    }
    key = (lang or "").strip().lower()
    return m.get(key, key or "english")


def _extract_audio_and_sr_from_qwen_output(output):
    """Нормализует различные форматы ответа qwen_tts в (audio_np_float32, sr)."""
    # tuple/list: (audio, sr)
    if isinstance(output, (tuple, list)) and len(output) >= 2:
        audio, sr = output[0], int(output[1])
    elif isinstance(output, dict):
        audio = output.get("audio") or output.get("wav") or output.get("speech")
        sr = int(output.get("sr") or output.get("sample_rate") or 24000)
    else:
        audio = getattr(output, "audio", None)
        sr = int(getattr(output, "sample_rate", 24000))

    if audio is None:
        raise RuntimeError("Qwen output не содержит audio")

    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    elif hasattr(audio, "cpu") and hasattr(audio, "numpy"):
        audio = audio.cpu().numpy()

    audio = np.asarray(audio, dtype=np.float32)
    return audio, sr


def _qwen_generate_any(model, text: str, ref_audio_path: str | None = None, ref_text: str | None = None):
    """Пробует API-методы Qwen3-TTS в порядке совместимости.

    Для Base-модели приоритет — generate_voice_clone(ref_audio=...).
    """
    is_base_model = "base" in QWEN_LOCAL_MODEL_ID.lower()

    # Базовые kwargs
    common_kwargs = {
        "text": text,
        "language": _normalize_qwen_language("en"),
        "instruct": QWEN_OUTPUT_STYLE_PROMPT,
        "speaker": None,
    }

    clone_kwargs_xvec = {
        "text": text,
        "language": _normalize_qwen_language("en"),
        "ref_audio": ref_audio_path,
        "x_vector_only_mode": True,
    }
    clone_kwargs_with_text = {
        "text": text,
        "language": _normalize_qwen_language("en"),
        "ref_audio": ref_audio_path,
    }
    if ref_text:
        clone_kwargs_with_text["ref_text"] = ref_text

    attempts = []

    # 1) Для Base сначала пробуем x_vector_only_mode (устойчиво без корректного ref_text),
    # затем вариант с ref_text при наличии.
    if is_base_model and hasattr(model, "generate_voice_clone"):
        attempts.append(("generate_voice_clone[xvector]", model.generate_voice_clone, clone_kwargs_xvec))
        attempts.append(("generate_voice_clone", model.generate_voice_clone, clone_kwargs_with_text))

    # 2) Остальные варианты (на случай другой версии API).
    if hasattr(model, "generate_custom_voice"):
        attempts.append(("generate_custom_voice", model.generate_custom_voice, common_kwargs))
    if hasattr(model, "generate"):
        attempts.append(("generate", model.generate, common_kwargs))
    if hasattr(model, "tts"):
        attempts.append(("tts", model.tts, common_kwargs))
    if hasattr(model, "synthesize"):
        attempts.append(("synthesize", model.synthesize, common_kwargs))
    if hasattr(model, "infer"):
        attempts.append(("infer", model.infer, common_kwargs))

    # remove duplicates preserving order
    seen = set()
    uniq_attempts = []
    for name, fn, kwargs in attempts:
        key = (name, id(fn))
        if key in seen:
            continue
        seen.add(key)
        uniq_attempts.append((name, fn, kwargs))

    last_error = None

    def _run_with_progress(fn, kwargs, mode_name):
        done = threading.Event()

        def _progress_worker():
            with tqdm(total=100, desc=f"Qwen {mode_name}", unit="%", leave=False) as p:
                while not done.wait(0.2):
                    p.update(1)
                    if p.n >= 100:
                        p.n = 0
                        p.refresh()
                p.n = 100
                p.refresh()

        t = threading.Thread(target=_progress_worker, daemon=True)
        t.start()
        try:
            return _call_with_supported_kwargs(fn, kwargs)
        finally:
            done.set()
            t.join(timeout=1.0)

    for name, fn, kwargs in uniq_attempts:
        # Для clone-режима ref_audio обязателен
        if name.startswith("generate_voice_clone") and not kwargs.get("ref_audio"):
            print("[LOCAL][WARN] skip generate_voice_clone: ref_audio отсутствует")
            continue
        try:
            out = _run_with_progress(fn, kwargs, name)
            print(f"[LOCAL] Qwen mode: {name} | ref_audio={'yes' if ref_audio_path else 'no'}")
            return out
        except Exception as e:
            last_error = e
            print(f"[LOCAL][WARN] {name} failed: {e}")

    methods = [m for m in dir(model) if not m.startswith("_")]
    raise RuntimeError(
        "Не удалось вызвать ни один поддерживаемый метод TTS у Qwen3TTSModel. "
        f"Последняя ошибка: {last_error}. Методы модели: {methods}. "
        "Для Base-модели используйте generate_voice_clone с ref_audio (3-10 сек чистой речи)."
    )


def synthesize_text_local_qwen(text: str, out_wav_path: str, ref_audio_path: str | None = None):
    model, tokenizer = load_local_qwen_tts()
    clean_text = clean_segment_text_en(text)
    output = _qwen_generate_any(model, clean_text, ref_audio_path=ref_audio_path, ref_text=None)
    audio, sr = _extract_audio_and_sr_from_qwen_output(output)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    save_pcm16_wav(out_wav_path, audio, sr)


def _is_nonempty_wav(path: str) -> bool:
    if not os.path.exists(path) or os.path.getsize(path) < 128:
        return False
    try:
        audio, _ = load_pcm16_wav(path)
        return audio.size > 1 and float(np.max(np.abs(audio))) > 1e-6
    except Exception:
        return False


def synthesize_segment_qwen_local(text: str, tmp_dir: Path, seg_id: int, target_samples: int, ref_audio_path: str | None = None, qwen_cache_dir: Path | None = None) -> np.ndarray:
    clauses = [c for c in split_text_with_punctuation(text) if len(c) >= CLAUSE_MIN_CHARS] or split_text_with_punctuation(text)
    if not clauses:
        return np.zeros(target_samples, dtype=np.float32)
    weights = [max(1, len(c)) for c in clauses]
    total = sum(weights)
    out_parts = []
    for j, clause in enumerate(clauses):
        qwen_base_dir = qwen_cache_dir or tmp_dir
        qwen_base_dir.mkdir(parents=True, exist_ok=True)
        qwen_out = str(qwen_base_dir / f"seg_{seg_id:05d}_qwen_{j:02d}.wav")
        qwen_rvc_out = str(tmp_dir / f"seg_{seg_id:05d}_qwen_rvc_{j:02d}.wav")
        if _is_nonempty_wav(qwen_out):
            print(f"[QWEN][CACHE] reuse: {qwen_out}")
        else:
            synthesize_text_local_qwen(clause, qwen_out, ref_audio_path=ref_audio_path)
        if RVC_ENABLED:
            apply_rvc_conversion(qwen_out, qwen_rvc_out)
            clause_audio, clause_sr = load_pcm16_wav(qwen_rvc_out)
        else:
            clause_audio, clause_sr = load_pcm16_wav(qwen_out)
        clause_audio = resample_audio_np(clause_audio, clause_sr, OUTPUT_SAMPLE_RATE)
        target_clause = max(1, int(round(target_samples * (weights[j] / total))))
        clause_audio = stretch_to_samples(clause_audio, target_clause)
        out_parts.append(clause_audio)
        if j < len(clauses) - 1:
            pause_n = micro_pause_samples_for_clause(clause)
            out_parts.append(np.zeros(pause_n, dtype=np.float32))
    merged = np.concatenate(out_parts).astype(np.float32)
    return stretch_to_samples(merged, target_samples)

def extract_reference_for_segment(source_audio: np.ndarray, sr: int, start: float, end: float, out_path: str):
    center_start = max(0.0, start - REF_PAD_SECONDS)
    center_end = min(len(source_audio) / sr, end + REF_PAD_SECONDS)
    if center_end <= center_start:
        center_end = min(len(source_audio) / sr, center_start + REF_MIN_SECONDS)
    seg_audio = source_audio[int(center_start * sr) : int(center_end * sr)]
    min_samples = int(REF_MIN_SECONDS * sr)
    max_samples = int(REF_MAX_SECONDS * sr)
    if seg_audio.size < min_samples:
        needed = min_samples - seg_audio.size
        left_pad = needed // 2
        right_pad = needed - left_pad
        seg_audio = np.pad(seg_audio, (left_pad, right_pad), mode="constant")
    if seg_audio.size > max_samples:
        mid = seg_audio.size // 2
        half = max_samples // 2
        seg_audio = seg_audio[max(0, mid - half) : max(0, mid - half) + max_samples]
    save_pcm16_wav(out_path, seg_audio.astype(np.float32), sr)

def finalize_master(audio: np.ndarray) -> np.ndarray:
    threshold = 0.65
    ratio = 3.0
    abs_audio = np.abs(audio)
    over = abs_audio > threshold
    compressed = audio.copy()
    compressed[over] = np.sign(audio[over]) * (threshold + (abs_audio[over] - threshold) / ratio)
    peak = float(np.max(np.abs(compressed))) if compressed.size else 0.0
    if peak > 0.98:
        compressed = compressed * (0.98 / peak)
    return compressed.astype(np.float32)

def synthesize_aligned_track(translated_segments, out_path: str, total_duration_s: float, source_audio: np.ndarray, source_sr: int):
    total_samples = max(1, int(round(total_duration_s * OUTPUT_SAMPLE_RATE)))
    timeline = np.zeros(total_samples, dtype=np.float32)
    tmp_dir = Path(WORK_DIR) / "tmp_tts_segments"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    project_qwen_cache_dir = Path(out_path).parent / f"{Path(out_path).stem}_qwen_cache"
    project_qwen_cache_dir.mkdir(parents=True, exist_ok=True)
    pbar = tqdm(total=len(translated_segments), desc="Qwen3-TTS синтез", unit="seg")
    for i, seg in enumerate(translated_segments):
        text = (seg.get("text_en") or "").strip()
        if not text:
            pbar.update(1)
            continue
        start = max(0.0, float(seg.get("start", 0.0)))
        end = max(start + 0.05, float(seg.get("end", start + 0.05)))
        seg_target_len = max(1, int(round((end - start) * OUTPUT_SAMPLE_RATE)))
        ref_path = str(tmp_dir / f"seg_{i:05d}_ref.wav")
        extract_reference_for_segment(source_audio, source_sr, start, end, ref_path)
        seg_audio = synthesize_segment_qwen_local(
            text,
            tmp_dir,
            i,
            seg_target_len,
            ref_audio_path=ref_path,
            qwen_cache_dir=project_qwen_cache_dir,
        )
        start_idx = int(round(start * OUTPUT_SAMPLE_RATE))
        end_idx = min(total_samples, start_idx + seg_audio.shape[0])
        if end_idx > start_idx:
            chunk = seg_audio[: end_idx - start_idx]
            timeline[start_idx:end_idx] += chunk
        pbar.update(1)
    pbar.close()
    timeline = finalize_master(timeline)
    save_pcm16_wav(out_path, timeline, OUTPUT_SAMPLE_RATE)

def main():
    global RVC_INFER_PYTHON
    RVC_INFER_PYTHON = _resolve_rvc_infer_python()
    _ensure_sox_in_path()
    _check_runtime_tools()
    print("=== Запуск обработки видео ===")
    print(f"Рабочая папка: {VIDEO_FOLDER}")
    print("\nЗагрузка моделей... (первый запуск может занять 10–40 мин на скачивание)")
    start_time = time.time()
    print("• Whisper large-v3 ", end="", flush=True)
    whisper_model = whisper.load_model("large-v3", device="cpu", download_root=os.path.join(WORK_DIR, "whisper_models"))
    print(f" — OK ({time.time() - start_time:.1f} сек)")
    print("• MarianMT (ru → en) ", end="", flush=True)
    marian_local_path = r"D:\pinokio\hf_cache\models--Helsinki-NLP--opus-mt-ru-en\snapshots\main"
    tokenizer = MarianTokenizer.from_pretrained(marian_local_path, local_files_only=True)
    model = MarianMTModel.from_pretrained(marian_local_path, local_files_only=True)
    print(f" — OK ({time.time() - start_time:.1f} сек)")
    print(f"• Qwen3-TTS local backend — {QWEN_LOCAL_MODEL_ID}")
    print(f"  ENABLE_QWEN_SIZE: {ENABLE_QWEN_SIZE}")
    print(f"  preload at start: {QWEN_LOAD_AT_START}")
    print(f"• RVC train required: {RVC_TRAIN_REQUIRED}")
    print(f"• RVC infer python: {RVC_INFER_PYTHON}")
    if RVC_ENABLED:
        print("[RVC] Проверка/подготовка train+infer скриптов...")
        ensure_rvc_scripts_ready()
    if QWEN_LOAD_AT_START:
        try:
            load_local_qwen_tts()
        except Exception as e:
            print("\n" + "="*70)
            print("ОШИБКА ЗАГРУЗКИ Qwen3-TTS:")
            print(e)
            print("Возможные причины:")
            print("1. Не установлен пакет: pip install -U qwen-tts")
            print("2. Неверный HF_TOKEN")
            print("3. Нет интернета / места на диске")
            print("4. SoX/ffmpeg/ffprobe недоступны в PATH")
            print("="*70)
            raise
    else:
        print("[INFO] Предзагрузка Qwen отключена: модель загрузится при первом TTS-сегменте.")
    print(f"Все модели загружены за {time.time() - start_time:.1f} сек\n")
    print(f"Режим тайм-выравнивания: {'ULTRA-SAFE' if ULTRA_SAFE_MODE else 'NORMAL'}")
    video_files = [f for f in os.listdir(VIDEO_FOLDER) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
    if not video_files:
        print("Видео не найдено в папке!")
        raise SystemExit(0)
    print(f"Найдено видео: {len(video_files)} шт.")
    print(", ".join(video_files), "\n")
    for filename in tqdm(video_files, desc="Обработка видео", unit="file"):
        print(f"\n{'=' * 60}")
        print(f"Обрабатываем: {filename}")
        print(f"{'-' * 60}")
        video_path = os.path.join(VIDEO_FOLDER, filename)
        base_name = os.path.splitext(filename)[0]
        audio_path = os.path.join(VIDEO_FOLDER, f"{base_name}.wav")
        transcribed_file = os.path.join(VIDEO_FOLDER, f"{base_name}_transcribed.txt")
        transcribed_segments_file = os.path.join(VIDEO_FOLDER, f"{base_name}_transcribed_segments.json")
        translated_file = os.path.join(VIDEO_FOLDER, f"{base_name}_translated_en.txt")
        translated_segments_file = os.path.join(VIDEO_FOLDER, f"{base_name}_translated_segments_en.json")
        expressive_segments_file = os.path.join(VIDEO_FOLDER, f"{base_name}_expressive_segments_en.json")
        eng_audio_path = os.path.join(VIDEO_FOLDER, f"{base_name}_eng.wav")
        if os.path.exists(audio_path):
            print("Аудио уже существует → пропуск")
        else:
            print("Извлечение аудио...")
            start = time.time()
            video = VideoFileClip(video_path)
            audio = video.audio
            audio.write_audiofile(audio_path, codec="pcm_s16le", fps=OUTPUT_SAMPLE_RATE, ffmpeg_params=["-ac", "1"], logger=None)
            video.close()
            print(f"Аудио сохранено ({time.time() - start:.1f} сек): {audio_path}")
        source_audio, source_sr = load_pcm16_wav(audio_path)
        source_audio = resample_audio_np(source_audio, source_sr, OUTPUT_SAMPLE_RATE)
        source_duration = len(source_audio) / OUTPUT_SAMPLE_RATE
        need_asr = not (os.path.exists(transcribed_file) and os.path.exists(transcribed_segments_file))
        if need_asr:
            print(f"Распознавание речи (Whisper large-v3, язык={SOURCE_LANGUAGE})...")
            start = time.time()
            whisper_audio = load_wav_for_whisper(audio_path)
            whisper_result = whisper_model.transcribe(whisper_audio, language=SOURCE_LANGUAGE, verbose=False, fp16=False)
            transcribed_text = whisper_result["text"].strip()
            segments_ru = whisper_result.get("segments", [])
            print(f"Распознано ({time.time() - start:.1f} сек)")
            with open(transcribed_file, "w", encoding="utf-8") as f:
                f.write(transcribed_text)
            with open(transcribed_segments_file, "w", encoding="utf-8") as f:
                json.dump(segments_ru, f, ensure_ascii=False, indent=2)
        else:
            print("Распознанный русский текст + сегменты уже существуют → начинаем с перевода")
            with open(transcribed_file, "r", encoding="utf-8") as f:
                transcribed_text = f.read().strip()
            with open(transcribed_segments_file, "r", encoding="utf-8") as f:
                segments_ru = json.load(f)
        if os.path.exists(translated_file) and os.path.exists(translated_segments_file):
            print("Английский перевод + сегменты уже существует → пропуск")
            with open(translated_file, "r", encoding="utf-8") as f:
                translated_text = f.read().strip()
            with open(translated_segments_file, "r", encoding="utf-8") as f:
                translated_segments = json.load(f)
            for seg in translated_segments:
                seg["text_en"] = clean_segment_text_en(seg.get("text_en", ""))
        else:
            print("Перевод текста/сегментов на английский...")
            start = time.time()
            translated_segments = translate_segments_ru_to_en(segments_ru, tokenizer, model)
            translated_text = " ".join(seg.get("text_en", "") for seg in translated_segments).strip()
            with open(translated_file, "w", encoding="utf-8") as f:
                f.write(translated_text)
            with open(translated_segments_file, "w", encoding="utf-8") as f:
                json.dump(translated_segments, f, ensure_ascii=False, indent=2)
            print(f"Перевод завершён ({time.time() - start:.1f} сек)")
        expressive_segments = refine_segments_for_expression(translated_segments)
        with open(expressive_segments_file, "w", encoding="utf-8") as f:
            json.dump(expressive_segments, f, ensure_ascii=False, indent=2)
        if os.path.exists(eng_audio_path):
            print("Английское аудио уже существует → пропуск")
        else:
            print("Синтез английского аудио с тайм-выравниванием сегментов...")
            print(f"TTS backend: {TTS_BACKEND}")
            print(f"Экспрессивных сегментов: {len(expressive_segments)}")
            if RVC_ENABLED:
                print("[RVC] Подготовка/обучение голосовой модели на исходном русском аудио...")
                train_rvc_voice_model(base_name, audio_path)
            start = time.time()
            synthesize_aligned_track(expressive_segments, eng_audio_path, source_duration, source_audio, OUTPUT_SAMPLE_RATE)
            print(f"Аудио создано ({time.time() - start:.1f} сек): {eng_audio_path}")
    print("\n" + "=" * 70)
    print("Все файлы обработаны!")
    print("Готовые файлы в папке:")
    print("• .wav — оригинальное аудио")
    print("• _transcribed.txt — русский текст")
    print("• _transcribed_segments.json — русский текст с таймкодами")
    print("• _translated_en.txt — английский перевод")
    print("• _translated_segments_en.json — перевод с таймкодами")
    print("• _expressive_segments_en.json — логическая нарезка для интонации")
    print("• _eng.wav — английская озвучка с выравниванием")
    print("=" * 70)

if __name__ == "__main__":
    main()