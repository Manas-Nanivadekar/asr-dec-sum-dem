import os

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

print("Loading summarization model (meta-llama/Llama-3.2-3B-Instruct)...")

_sum_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

sum_tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    token=HF_TOKEN,
)
sum_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    token=HF_TOKEN,
    torch_dtype=_sum_dtype,
).to(device)
sum_model.eval()

print("Summarization model ready.")

_SUMMARIZE_SYSTEM = (
    "You are a research assistant specializing in speech and conversation analysis. "
    "Summarize the provided conversation transcript clearly and concisely. "
    "Highlight key topics discussed, the main points raised by each speaker, "
    "and any notable conclusions or action items. "
    "Write in formal academic prose. Do not include filler phrases."
)

_MAX_TRANSCRIPT_CHARS = 4000  # ~1 000 tokens of source text


def summarize_transcript(transcript_text: str) -> str:
    if len(transcript_text) > _MAX_TRANSCRIPT_CHARS:
        transcript_text = transcript_text[:_MAX_TRANSCRIPT_CHARS] + "…"

    messages = [
        {"role": "system", "content": _SUMMARIZE_SYSTEM},
        {"role": "user",   "content": f"Transcript:\n\n{transcript_text}\n\nSummary:"},
    ]

    input_ids = sum_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output_ids = sum_model.generate(
            input_ids,
            max_new_tokens=400,
            temperature=0.3,
            do_sample=True,
            pad_token_id=sum_tokenizer.eos_token_id,
        )

    generated = output_ids[0][input_ids.shape[1]:]
    return sum_tokenizer.decode(generated, skip_special_tokens=True).strip()


if __name__ == "__main__":
    audio_file = "179729.wav"
    result = pipeline.transcribe(audio_file)
    print(result)
