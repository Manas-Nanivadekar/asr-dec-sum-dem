import os
import re
from pathlib import Path

os.environ["PYTORCH_NVFUSER_DISABLE"] = "1"
os.environ["PYTORCH_JIT_USE_NNC_NOT_NVFUSER"] = "1"

import torch

torch._C._jit_set_nvfuser_enabled(False)
torch._C._jit_override_can_fuse_on_gpu(False)
torch._C._jit_set_texpr_fuser_enabled(True)

import soundfile as sf
from scipy.signal import resample_poly
from dotenv import load_dotenv
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from diarizen.pipelines.inference import DiariZenPipeline

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

SUMMARIZATION_CONFIG = {
    "input_path": "../Track2_ASR/outputs/ASR/fa_asr_predictions",
    "gt_path": "./data/Track_4_DS_DevData_1/Hindi/GT/gt_dev_data_summarization.csv",
    "columns": {
        "rec_id": "rec_id",
        "gt_summary": "gt_summary",
    },
    "model": {
        "name": "meta-llama/Llama-3.2-3B-Instruct",
        "torch_dtype": "float16",
        "device": "cuda",
        "device_map": "auto",
        "hf_token": "<---------------Paste-Your-HF-Token-for-LLAMA-3.2-3B-Instruct-Model-Here--------------->",
        "low_cpu_mem_usage": True,
    },
    "inference": {
        "max_new_tokens": 512,
        "do_sample": False,
        "top_k": 50,
        "repetition_penalty": 1.0,
        "num_beams": 1,
        "use_cache": True,
    },
    "optimization": {
        "torch_compile": False,
        "batch_size": 1,
    },
    "output": {
        "summarization_dir": "./outputs/Summarization",
        "metrics_dir": "./outputs/metrics",
        "summary_pattern": "{rec_id}_summary.txt",
        "asr_pattern": "{rec_id}_fullaudio_transcription.txt",
    },
    "logging": {
        "level": "INFO",
        "format": "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        "date_format": "%Y-%m-%d %H:%M:%S",
        "log_dir": "./outputs/logs",
    },
}

Path(SUMMARIZATION_CONFIG["output"]["summarization_dir"]).mkdir(parents=True, exist_ok=True)
Path(SUMMARIZATION_CONFIG["output"]["metrics_dir"]).mkdir(parents=True, exist_ok=True)
Path(SUMMARIZATION_CONFIG["logging"]["log_dir"]).mkdir(parents=True, exist_ok=True)

# -------------------------------
# Load models at startup
# -------------------------------

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

print("Loading ASR model...")

asr_model = AutoModel.from_pretrained(
    "ai4bharat/indic-conformer-600m-multilingual",
    trust_remote_code=True,
    token=HF_TOKEN,
)
asr_model = asr_model.to(device)

print("Loading diarization model...")

diar_pipeline = DiariZenPipeline.from_pretrained(
    "BUT-FIT/diarizen-wavlm-large-s80-md"
).to(device)

diar_pipeline.embedding_batch_size = 16
diar_pipeline.segmentation_batch_size = 16

# -------------------------------
# Pipeline Class
# -------------------------------


class DiCoIndicPipeline:

    def __init__(self, asr_model, diarization_pipeline, device):
        self.asr_model = asr_model
        self.diarization_pipeline = diarization_pipeline
        self.device = device

    def decode_audio(self, wav_file, rttm_segments, target_sr=16000):

        audio, sr = sf.read(wav_file, dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != target_sr:
            audio = resample_poly(audio, target_sr, sr)
            sr = target_sr

        transcripts = []
        self.asr_model.eval()

        for idx, seg in enumerate(rttm_segments, start=1):

            start = max(0, int(seg["start"] * sr) - int(0.1 * sr))
            end = min(len(audio), int(seg["end"] * sr) + int(0.1 * sr))

            segment = audio[start:end]

            if len(segment) == 0:
                continue

            waveform = torch.from_numpy(segment).unsqueeze(0).to(self.device)

            min_length = 512
            if waveform.shape[1] < min_length:
                padding = min_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))

            with torch.no_grad():
                transcription = self.asr_model(waveform, "hi", "rnnt")

            transcription_text = (
                transcription[0]
                if isinstance(transcription, (list, tuple))
                else str(transcription)
            )

            transcripts.append(
                {
                    "speaker": seg["speaker"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": transcription_text,
                }
            )

        return transcripts

    def transcribe(self, audio_path):

        diar_output = self.diarization_pipeline(audio_path)

        segments = []
        for speaker in diar_output.labels():
            timeline = diar_output.label_timeline(speaker)
            for segment in timeline:
                segments.append(
                    {"speaker": speaker, "start": segment.start, "end": segment.end}
                )

        segments = sorted(segments, key=lambda x: x["start"])

        transcripts = self.decode_audio(audio_path, segments)

        return transcripts


pipeline = DiCoIndicPipeline(
    asr_model, diarization_pipeline=diar_pipeline, device=device
)

# -------------------------------
# Summarization Model (Llama-3.2-3B-Instruct)
# -------------------------------

print(f"Loading summarization model ({SUMMARIZATION_CONFIG['model']['name']})...")

_sum_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
_sum_model_token = HF_TOKEN
if not _sum_model_token:
    cfg_token = SUMMARIZATION_CONFIG["model"]["hf_token"]
    if "Paste-Your-HF-Token" not in cfg_token:
        _sum_model_token = cfg_token

sum_tokenizer = AutoTokenizer.from_pretrained(
    SUMMARIZATION_CONFIG["model"]["name"],
    token=_sum_model_token,
)
_sum_model_kwargs = {
    "token": _sum_model_token,
    "torch_dtype": _sum_dtype,
    "low_cpu_mem_usage": SUMMARIZATION_CONFIG["model"]["low_cpu_mem_usage"],
}
if torch.cuda.is_available():
    _sum_model_kwargs["device_map"] = SUMMARIZATION_CONFIG["model"]["device_map"]

sum_model = AutoModelForCausalLM.from_pretrained(
    SUMMARIZATION_CONFIG["model"]["name"],
    **_sum_model_kwargs,
)
sum_model.eval()

print("Summarization model ready.")

_SUMMARIZE_SYSTEM = (
    "You are a transcript summarization assistant. "
    "Produce clean, factual Hindi/English-agnostic summaries from ASR transcripts. "
    "Output ONLY the required XML-like tags, with no extra words before or after them."
)

_MAX_TRANSCRIPT_CHARS = 4000  # ~1 000 tokens of source text
_SUMMARY_TAGS = ("SUMMARY", "TOPIC", "CONCLUSION")
_SUMMARY_TEMPLATE = (
    "Transcript:\n\n{transcript}\n\n"
    "Return exactly this structure and nothing else:\n"
    "<SUMMARY>One concise paragraph summary.</SUMMARY>\n"
    "<TOPIC>Comma-separated key topics.</TOPIC>\n"
    "<CONCLUSION>Main outcome/actionable conclusion.</CONCLUSION>"
)


def _extract_tag(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return " ".join(match.group(1).strip().split())


def _clean_structured_summary(raw_text: str) -> tuple[str, dict[str, str]]:
    sections = {tag: _extract_tag(raw_text, tag) for tag in _SUMMARY_TAGS}
    if not sections["SUMMARY"]:
        # Fallback: salvage model text into SUMMARY if tags are missing.
        cleaned = " ".join(raw_text.strip().split())
        sections["SUMMARY"] = cleaned
    sections["TOPIC"] = sections["TOPIC"] or "NA"
    sections["CONCLUSION"] = sections["CONCLUSION"] or "NA"

    tagged = "\n".join(
        f"<{tag}>{sections[tag]}</{tag}>"
        for tag in _SUMMARY_TAGS
    )
    return tagged, sections


def summarize_transcript(transcript_text: str) -> tuple[str, dict[str, str]]:
    if len(transcript_text) > _MAX_TRANSCRIPT_CHARS:
        transcript_text = transcript_text[:_MAX_TRANSCRIPT_CHARS] + "…"

    messages = [
        {"role": "system", "content": _SUMMARIZE_SYSTEM},
        {"role": "user", "content": _SUMMARY_TEMPLATE.format(transcript=transcript_text)},
    ]

    formatted = sum_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    model_device = next(sum_model.parameters()).device
    input_ids = sum_tokenizer(formatted, return_tensors="pt").input_ids.to(model_device)

    with torch.no_grad():
        output_ids = sum_model.generate(
            input_ids,
            max_new_tokens=SUMMARIZATION_CONFIG["inference"]["max_new_tokens"],
            do_sample=SUMMARIZATION_CONFIG["inference"]["do_sample"],
            top_k=SUMMARIZATION_CONFIG["inference"]["top_k"],
            repetition_penalty=SUMMARIZATION_CONFIG["inference"]["repetition_penalty"],
            num_beams=SUMMARIZATION_CONFIG["inference"]["num_beams"],
            use_cache=SUMMARIZATION_CONFIG["inference"]["use_cache"],
            pad_token_id=sum_tokenizer.eos_token_id,
        )

    generated = output_ids[0][input_ids.shape[1] :]
    raw = sum_tokenizer.decode(generated, skip_special_tokens=True).strip()
    return _clean_structured_summary(raw)


if __name__ == "__main__":
    audio_file = "179729.wav"
    result = pipeline.transcribe(audio_file)
    print(result)
