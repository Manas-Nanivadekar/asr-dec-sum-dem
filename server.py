import os
import logging
import subprocess
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel

# ── Logging setup ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("server")

# ── Load pipeline (models load here at startup) ───────────
log.info("Importing pipeline from main.py …")
from main import pipeline, summarize_transcript
log.info("All models ready.")

app = FastAPI()
app.mount("/assets", StaticFiles(directory="assets"), name="assets")


# ── Always-JSON exception handlers ───────────────────────
@app.exception_handler(StarletteHTTPException)
async def http_exc_handler(request: Request, exc: StarletteHTTPException):
    log.error(f"HTTP {exc.status_code} on {request.method} {request.url.path}: {exc.detail}")
    return JSONResponse({"status": "error", "detail": str(exc.detail)}, status_code=exc.status_code)


@app.exception_handler(Exception)
async def unhandled_exc_handler(request: Request, exc: Exception):
    log.exception(f"Unhandled exception on {request.method} {request.url.path}")
    return JSONResponse({"status": "error", "detail": str(exc)}, status_code=500)


# ── Routes ────────────────────────────────────────────────
@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Helpers ───────────────────────────────────────────────
def ensure_wav(src: str) -> tuple[str, bool]:
    """Convert to WAV via ffmpeg if needed. Returns (path, is_temp)."""
    if src.lower().endswith(".wav"):
        return src, False
    out = src + ".wav"
    log.info(f"Converting {Path(src).name} → WAV via ffmpeg …")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", src, "-ar", "16000", "-ac", "1",
             "-c:a", "pcm_s16le", out, "-loglevel", "quiet"],
            check=True, timeout=120,
        )
        log.info("Conversion done.")
        return out, True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        log.warning(f"ffmpeg conversion failed ({e}); will try reading directly.")
        return src, False


# ── Transcribe endpoint ───────────────────────────────────
@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    t0 = time.time()
    size_kb = 0

    # ── Save upload to temp file ──────────────────────────
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await audio.read()
        size_kb = len(content) / 1024
        tmp.write(content)
        src_path = tmp.name

    log.info(f"Received: '{audio.filename}'  ({size_kb:.1f} KB)  →  {src_path}")

    # ── Convert to WAV if needed ──────────────────────────
    wav_path, wav_is_temp = ensure_wav(src_path)

    # ── Run pipeline ──────────────────────────────────────
    try:
        log.info("Running diarization …")
        t1 = time.time()
        result = pipeline.transcribe(wav_path)
        elapsed = time.time() - t1

        log.info(f"Diarization + ASR done in {elapsed:.1f}s  →  {len(result)} segments")
        for i, seg in enumerate(result):
            seg["speaker"] = str(seg["speaker"])
            log.info(f"  [{i+1:3d}] Spk {seg['speaker']}  "
                     f"{seg['start']:6.2f}s – {seg['end']:6.2f}s  |  {seg['text']}")

        log.info(f"Total request time: {time.time()-t0:.1f}s")
        return JSONResponse({"status": "ok", "transcript": result})

    except Exception as e:
        log.exception(f"Pipeline failed after {time.time()-t0:.1f}s")
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)

    finally:
        for path in ([src_path] + ([wav_path] if wav_is_temp else [])):
            try:
                os.unlink(path)
            except OSError:
                pass


# ── Script endpoint ───────────────────────────────────────
@app.get("/script")
async def get_script():
    path = Path("sample_text.txt")
    if not path.exists():
        return JSONResponse({"status": "error", "detail": "Script file not found."}, status_code=404)
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split("\t", 1)
        if len(parts) == 2:
            lines.append({"speaker": parts[0].strip(), "text": parts[1].strip()})
    return JSONResponse({"status": "ok", "lines": lines})


# ── Summarize endpoint ────────────────────────────────────
class SummarizeRequest(BaseModel):
    text: str


@app.post("/summarize")
async def summarize(req: SummarizeRequest):
    t0 = time.time()
    if not req.text.strip():
        return JSONResponse({"status": "error", "detail": "Empty transcript text."}, status_code=400)

    log.info(f"Summarization request: {len(req.text)} chars")
    try:
        summary = summarize_transcript(req.text)
        log.info(f"Summarization done in {time.time()-t0:.1f}s  ({len(summary)} chars)")
        return JSONResponse({"status": "ok", "summary": summary})
    except Exception as e:
        log.exception("Summarization failed")
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)


# ── Entry point ───────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    log.info("Starting server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
