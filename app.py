import gradio as gr
from main import pipeline

# --------------------------------
# Speaker color palette
# --------------------------------

SPEAKER_COLORS = [
    "#2563EB",  # Blue
    "#DC2626",  # Red
    "#16A34A",  # Green
    "#D97706",  # Amber
    "#7C3AED",  # Violet
    "#0891B2",  # Cyan
    "#BE185D",  # Pink
    "#C2410C",  # Orange
]

SPEAKER_COLORS_LIGHT = [
    "#EFF6FF",
    "#FEF2F2",
    "#F0FDF4",
    "#FFFBEB",
    "#F5F3FF",
    "#ECFEFF",
    "#FDF2F8",
    "#FFF7ED",
]


# --------------------------------
# Helpers
# --------------------------------


def format_time(seconds):
    m = int(seconds // 60)
    s = seconds % 60
    if m > 0:
        return f"{m}:{s:05.2f}"
    return f"{s:.1f}s"


def format_transcript_html(transcripts):
    if not transcripts:
        return _empty_state("No speech detected in the audio.")

    speakers = sorted(set(t["speaker"] for t in transcripts))
    color_map = {spk: SPEAKER_COLORS[i % len(SPEAKER_COLORS)] for i, spk in enumerate(speakers)}
    light_map = {spk: SPEAKER_COLORS_LIGHT[i % len(SPEAKER_COLORS_LIGHT)] for i, spk in enumerate(speakers)}

    total_duration = max(t["end"] for t in transcripts)

    # ---- Stats bar ----
    stats = f"""
    <div style="
        display:flex; gap:32px; padding:14px 20px;
        background:#F8FAFC; border-radius:10px; margin-bottom:18px;
        border:1px solid #E2E8F0; font-family:system-ui,sans-serif;
        flex-wrap:wrap;
    ">
        <div>
            <div style="font-size:11px;color:#64748B;font-weight:700;text-transform:uppercase;letter-spacing:.06em;">Duration</div>
            <div style="font-size:20px;font-weight:800;color:#0F172A;">{format_time(total_duration)}</div>
        </div>
        <div>
            <div style="font-size:11px;color:#64748B;font-weight:700;text-transform:uppercase;letter-spacing:.06em;">Speakers</div>
            <div style="font-size:20px;font-weight:800;color:#0F172A;">{len(speakers)}</div>
        </div>
        <div>
            <div style="font-size:11px;color:#64748B;font-weight:700;text-transform:uppercase;letter-spacing:.06em;">Utterances</div>
            <div style="font-size:20px;font-weight:800;color:#0F172A;">{len(transcripts)}</div>
        </div>
    </div>
    """

    # ---- Speaker legend ----
    legend = '<div style="display:flex;gap:14px;flex-wrap:wrap;margin-bottom:16px;font-family:system-ui,sans-serif;">'
    for spk in speakers:
        c = color_map[spk]
        legend += f"""
        <div style="display:flex;align-items:center;gap:7px;
            padding:4px 12px;border-radius:20px;background:{light_map[spk]};
            border:1px solid {c}33;">
            <div style="width:10px;height:10px;border-radius:50%;background:{c};flex-shrink:0;"></div>
            <span style="font-size:12px;color:{c};font-weight:700;">Speaker {spk}</span>
        </div>"""
    legend += "</div>"

    # ---- Utterances ----
    rows = '<div style="display:flex;flex-direction:column;gap:6px;">'
    for t in transcripts:
        c = color_map[t["speaker"]]
        bg = light_map[t["speaker"]]
        rows += f"""
        <div style="
            display:flex;gap:14px;align-items:flex-start;
            padding:10px 16px;border-radius:10px;
            background:{bg};border-left:3px solid {c};
        ">
            <div style="min-width:72px;flex-shrink:0;">
                <div style="font-size:11px;font-weight:800;color:{c};text-transform:uppercase;letter-spacing:.04em;">
                    Spk&nbsp;{t['speaker']}
                </div>
                <div style="font-size:10px;color:#94A3B8;margin-top:3px;">{format_time(t['start'])}</div>
                <div style="font-size:10px;color:#CBD5E1;">↓ {format_time(t['end'])}</div>
            </div>
            <div style="
                font-size:17px;color:#1E293B;line-height:1.65;flex:1;
                font-family:'Noto Sans Devanagari','Mangal','Arial Unicode MS',sans-serif;
            ">{t['text']}</div>
        </div>"""
    rows += "</div>"

    return stats + legend + rows


def _empty_state(msg="Upload or record audio to see the transcript here."):
    return f"""
    <div style="
        padding:60px 20px;text-align:center;
        color:#94A3B8;font-family:system-ui,sans-serif;font-size:15px;
        border:2px dashed #E2E8F0;border-radius:12px;margin-top:8px;
    ">{msg}</div>"""


# --------------------------------
# Gradio callback
# --------------------------------


def transcribe_audio(audio_path):
    if audio_path is None:
        return _empty_state()
    try:
        result = pipeline.transcribe(audio_path)
        return format_transcript_html(result)
    except Exception as e:
        return f"""
        <div style="
            padding:16px 20px;color:#DC2626;font-family:system-ui,sans-serif;
            border:1px solid #FCA5A5;border-radius:10px;background:#FEF2F2;
        "><strong>Error:</strong> {str(e)}</div>"""


# --------------------------------
# Gradio UI
# --------------------------------

CSS = """
h1 { font-family: system-ui, sans-serif !important; }
.gradio-container { max-width: 1200px !important; }
"""

with gr.Blocks(title="Indic ASR + Diarization", theme=gr.themes.Soft(), css=CSS) as demo:

    gr.Markdown(
        """
        # Indic Speech Transcription
        **Speaker Diarization + ASR** &nbsp;·&nbsp; Supports Hindi and other Indic languages
        """
    )

    with gr.Row(equal_height=False):

        # Left panel — input
        with gr.Column(scale=1, min_width=280):
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Audio Input",
            )
            transcribe_btn = gr.Button("▶  Transcribe", variant="primary", size="lg")
            gr.Markdown(
                "<small style='color:#94A3B8;'>Tip: click the mic icon to record, or drag-and-drop a WAV/MP3 file.</small>"
            )

        # Right panel — output
        with gr.Column(scale=2):
            transcript_output = gr.HTML(value=_empty_state(), label="Transcript")

    transcribe_btn.click(
        fn=transcribe_audio,
        inputs=audio_input,
        outputs=transcript_output,
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
