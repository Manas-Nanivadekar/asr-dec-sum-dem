import torch
import os
import tempfile
from transformers import AutoModel
from diarizen.pipelines.inference import DiariZenPipeline

from librosa import load as libr_load


# -------------------------------
# Load models at startup
# -------------------------------

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

print("Loading ASR model...")

asr_model = AutoModel.from_pretrained(
    "ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True
)
asr_model = asr_model.to(device)

print("Loading diarization model...")

diar_pipeline = DiariZenPipeline.from_pretrained(
    "BUT-FIT/diarizen-wavlm-large-s80-md"
).to(device)

diar_pipeline.embedding_batch_size = 16
diar_pipeline.segmentation_batch_size = 16

# -------------------------------
# Pipeline Class (simplified)
# -------------------------------


class DiCoIndicPipeline:

    def __init__(self, asr_model, diarization_pipeline, device):
        self.asr_model = asr_model
        self.diarization_pipeline = diarization_pipeline
        self.device = device

    def decode_audio(self, wav_file, rttm_segments, target_sr=16000):

        # ---- Load audio ----
        audio, sr = sf.read(wav_file, dtype="float32")
        # check if Mono
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        # Resample
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
                transcription_text = ""
                continue

            # ---- Convert to torch ----
            waveform = torch.from_numpy(segment).unsqueeze(0).to(self.device)

            # Pad if too short (minimum length for model compatibility)
            min_length = 512

            if waveform.shape[1] < min_length:
                padding = min_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))

            # Transcribe using model directly
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

        transcripts = self.decode_audio(
            audio_path,
            segments,
        )

        return transcripts


pipeline = DiCoIndicPipeline(
    asr_model, diarization_pipeline=diar_pipeline, device=device
)

audio_file = "179729.wav"  # path to uploaded audio
result = pipeline.transcribe(audio_file)
print(result)
