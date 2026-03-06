import os
import subprocess
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from main import pipeline

app = FastAPI()


def ensure_wav(src: str) -> tuple[str, bool]:
    """Convert to 16kHz mono WAV via ffmpeg if needed. Returns (path, is_temp)."""
    if src.lower().endswith(".wav"):
        return src, False
    out = src + ".wav"
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", src,
                "-ar", "16000", "-ac", "1",
                "-c:a", "pcm_s16le", out,
                "-loglevel", "quiet",
            ],
            check=True,
            timeout=120,
        )
        return out, True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return src, False  # let soundfile try anyway


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        src_path = tmp.name

    wav_path, wav_is_temp = ensure_wav(src_path)

    try:
        result = pipeline.transcribe(wav_path)
        for seg in result:
            seg["speaker"] = str(seg["speaker"])
        return JSONResponse({"status": "ok", "transcript": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(src_path)
        except OSError:
            pass
        if wav_is_temp:
            try:
                os.unlink(wav_path)
            except OSError:
                pass


app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
