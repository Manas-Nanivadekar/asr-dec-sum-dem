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
_SUMMARY_TAGS = (
    "SUMMARY",
    "TOPIC",
    "CONCLUSION",
    "DOMAIN",
    "DOCTOR_ADVICE",
    "CLINICAL_INDICATORS",
)
_SUMMARY_TEMPLATE = (
    "Transcript:\n\n{transcript}\n\n"
    "Task:\n"
    "1) Decide if this is a medical/clinical conversation.\n"
    "2) Always provide SUMMARY, TOPIC, and CONCLUSION.\n"
    "3) If and only if medical, also provide DOCTOR_ADVICE and CLINICAL_INDICATORS.\n\n"
    "Rules:\n"
    "- DOMAIN must be either MEDICAL or GENERAL.\n"
    "- If DOMAIN is GENERAL, set DOCTOR_ADVICE to NA and CLINICAL_INDICATORS to NA.\n"
    "- CLINICAL_INDICATORS should include vital signs/lab values/measurements when present (BP, sugar, SPO2, etc.), else NA.\n"
    "- Output only the tags below, nothing else.\n\n"
    "Return exactly this structure:\n"
    "<SUMMARY>One concise paragraph summary.</SUMMARY>\n"
    "<TOPIC>Comma-separated key topics.</TOPIC>\n"
    "<CONCLUSION>Main outcome/actionable conclusion.</CONCLUSION>\n"
    "<DOMAIN>MEDICAL or GENERAL</DOMAIN>\n"
    "<DOCTOR_ADVICE>Doctor's advice if medical, else NA</DOCTOR_ADVICE>\n"
    "<CLINICAL_INDICATORS>Clinical measurements/findings if medical and present, else NA</CLINICAL_INDICATORS>"
)


def _extract_tag(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return " ".join(match.group(1).strip().split())


_MEDICAL_KEYWORDS = {
    # English
    "doctor", "hospital", "clinic", "medicine", "medication", "tablet", "dose",
    "diagnosis", "symptom", "treatment", "prescription", "bp", "blood pressure",
    "sugar", "glucose", "diabetes", "fever", "pain", "headache", "menstruation",
    "period", "pregnancy", "hemoglobin", "spo2", "pulse", "heart rate",
    # Hindi
    "डॉक्टर", "अस्पताल", "दवा", "इलाज", "जांच", "टैबलेट", "बुखार", "दर्द",
    "ब्लड प्रेशर", "शुगर", "मधुमेह", "मासिक", "पीरियड", "गर्भ", "नाड़ी",
}


def _is_medical_transcript(text: str) -> bool:
    low = text.lower()
    if any(k in low for k in _MEDICAL_KEYWORDS):
        return True
    clinical_pattern = re.compile(
        r"\b("
        r"bp|blood pressure|sugar|glucose|hba1c|spo2|oxygen saturation|pulse|heart rate|"
        r"mmhg|mg/dl|bpm|mmol/l|hemoglobin|hb"
        r")\b",
        flags=re.IGNORECASE,
    )
    return bool(clinical_pattern.search(text))


def _trim_transcript(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n...\n" + text[-half:]


def _extract_clinical_indicators(text: str) -> str:
    # Capture common metric patterns discussed in clinical conversations.
    patterns = [
        r"\b(?:bp|blood pressure)\s*[:\-]?\s*\d{2,3}\s*/\s*\d{2,3}\b",
        r"\b(?:sugar|glucose)\s*[:\-]?\s*\d{2,3}(?:\.\d+)?\s*(?:mg/dl|mmol/l)?\b",
        r"\b(?:spo2|oxygen saturation)\s*[:\-]?\s*\d{2,3}\s*%?\b",
        r"\b(?:pulse|heart rate)\s*[:\-]?\s*\d{2,3}\s*(?:bpm)?\b",
        r"\b(?:hb|hemoglobin)\s*[:\-]?\s*\d{1,2}(?:\.\d+)?\b",
    ]
    hits = []
    for p in patterns:
        hits.extend(re.findall(p, text, flags=re.IGNORECASE))
    cleaned = []
    for h in hits:
        v = " ".join(str(h).split())
        if v and v not in cleaned:
            cleaned.append(v)
    return "; ".join(cleaned) if cleaned else "NA"


def _extract_doctor_guidance(text: str) -> str:
    # Pull likely advice-like sentences as fallback when model omits advice tags.
    advice_cues = (
        "should", "must", "take", "avoid", "consult", "rest", "follow", "drink",
        "exercise", "test", "check", "advice", "recommend", "prescribe",
        "करें", "ले", "खाएं", "बचें", "दिखाएं", "आराम", "जांच", "सलाह",
    )
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    picks = []
    for s in sentences:
        ss = " ".join(s.split())
        low = ss.lower()
        if len(ss) < 12:
            continue
        if any(cue in low for cue in advice_cues):
            picks.append(ss)
        if len(picks) >= 3:
            break
    if not picks:
        return "NA"
    return " ".join(picks)


def _generate_medical_fields(transcript_text: str, model_device: torch.device) -> dict[str, str]:
    prompt = (
        "You are extracting fields from a medical conversation transcript.\n"
        "Return only these tags and nothing else:\n"
        "<DOCTOR_ADVICE>Concise doctor guidance, if present else NA</DOCTOR_ADVICE>\n"
        "<CLINICAL_INDICATORS>Clinical measurements/findings (BP/sugar/SPO2/etc.) else NA</CLINICAL_INDICATORS>\n\n"
        f"Transcript:\n{transcript_text}"
    )
    messages = [
        {"role": "system", "content": "Return clean tagged output only."},
        {"role": "user", "content": prompt},
    ]
    formatted = sum_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    ids = sum_tokenizer(formatted, return_tensors="pt").input_ids.to(model_device)
    with torch.no_grad():
        out = sum_model.generate(
            ids,
            max_new_tokens=192,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            repetition_penalty=1.0,
            pad_token_id=sum_tokenizer.eos_token_id,
        )
    raw = sum_tokenizer.decode(out[0][ids.shape[1] :], skip_special_tokens=True).strip()
    return {
        "DOCTOR_ADVICE": _extract_tag(raw, "DOCTOR_ADVICE") or "NA",
        "CLINICAL_INDICATORS": _extract_tag(raw, "CLINICAL_INDICATORS") or "NA",
    }


def _clean_structured_summary(
    raw_text: str,
    transcript_text: str,
    model_device: torch.device,
) -> tuple[str, dict[str, str]]:
    sections = {tag: _extract_tag(raw_text, tag) for tag in _SUMMARY_TAGS}
    if not sections["SUMMARY"]:
        # Fallback: salvage model text into SUMMARY if tags are missing.
        cleaned = " ".join(raw_text.strip().split())
        sections["SUMMARY"] = cleaned
    sections["TOPIC"] = sections["TOPIC"] or "NA"
    sections["CONCLUSION"] = sections["CONCLUSION"] or "NA"
    domain = (sections["DOMAIN"] or "").strip().upper()
    is_medical = domain == "MEDICAL" or _is_medical_transcript(transcript_text)
    sections["DOMAIN"] = "MEDICAL" if is_medical else "GENERAL"

    if sections["DOMAIN"] == "MEDICAL":
        sections["DOCTOR_ADVICE"] = sections["DOCTOR_ADVICE"] or "NA"
        sections["CLINICAL_INDICATORS"] = sections["CLINICAL_INDICATORS"] or "NA"
        if sections["DOCTOR_ADVICE"] == "NA" or sections["CLINICAL_INDICATORS"] == "NA":
            gen = _generate_medical_fields(transcript_text, model_device)
            if sections["DOCTOR_ADVICE"] == "NA":
                sections["DOCTOR_ADVICE"] = gen["DOCTOR_ADVICE"]
            if sections["CLINICAL_INDICATORS"] == "NA":
                sections["CLINICAL_INDICATORS"] = gen["CLINICAL_INDICATORS"]
        if sections["DOCTOR_ADVICE"] == "NA":
            sections["DOCTOR_ADVICE"] = _extract_doctor_guidance(transcript_text)
        if sections["CLINICAL_INDICATORS"] == "NA":
            sections["CLINICAL_INDICATORS"] = _extract_clinical_indicators(transcript_text)
    else:
        sections["DOCTOR_ADVICE"] = "NA"
        sections["CLINICAL_INDICATORS"] = "NA"

    tagged = "\n".join(
        f"<{tag}>{sections[tag]}</{tag}>"
        for tag in _SUMMARY_TAGS
    )
    return tagged, sections


def summarize_transcript(transcript_text: str) -> tuple[str, dict[str, str]]:
    transcript_text = _trim_transcript(transcript_text, _MAX_TRANSCRIPT_CHARS)

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
    return _clean_structured_summary(raw, transcript_text, model_device)


if __name__ == "__main__":
    audio_file = "179729.wav"
    result = pipeline.transcribe(audio_file)
    print(result)
