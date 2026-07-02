import os
import uuid
import subprocess
import json
import shutil
import glob
import time
import asyncio
from dotenv import load_dotenv
from typing import Dict, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

# ── Locate ffmpeg/ffprobe (common locations) ─────────────
_FFMPEG_DIRS = [
    os.path.join(os.path.dirname(__file__), "ffmpeg", "bin"),
    r"C:\ffmpeg\bin",
    r"C:\Program Files\ffmpeg\bin",
]
for _d in _FFMPEG_DIRS:
    if os.path.isfile(os.path.join(_d, "ffmpeg.exe")):
        os.environ["PATH"] = _d + os.pathsep + os.environ.get("PATH", "")
        break

# Constants
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_FILE_SIZE_MB = 2048
JOB_RETENTION_SECONDS = 3600
DISABLE_YOUTUBE_URL = os.environ.get("DISABLE_YOUTUBE_URL", "false").lower() in ("1", "true", "yes")

# Application State
jobs: Dict[str, Dict] = {}

# Common yt-dlp options for YouTube downloading
_COMMON_YDL_OPTS = {
    'quiet': True,
    'no_warnings': True,
    'socket_timeout': 30,
    'retries': 10,
    'fragment_retries': 10,
    'nocheckcertificate': True,
    'cachedir': False,
    'extractor_args': {
        'youtube': {
            'player_client': ['tv_embed', 'android', 'mweb', 'web'],
            'player_skip': ['webpage', 'configs'],
        }
    },
    'http_headers': {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/120.0.0.0 Safari/537.36'
        ),
    },
}


async def cleanup_jobs():
    while True:
        try:
            await asyncio.sleep(300)
            now = time.time()

            for job_id in os.listdir(OUTPUT_DIR):
                job_path = os.path.join(OUTPUT_DIR, job_id)
                if os.path.isdir(job_path):
                    if now - os.path.getmtime(job_path) > JOB_RETENTION_SECONDS:
                        print(f"Purging old job: {job_id}")
                        shutil.rmtree(job_path, ignore_errors=True)
                        if job_id in jobs:
                            del jobs[job_id]

            for filename in os.listdir(UPLOAD_DIR):
                file_path = os.path.join(UPLOAD_DIR, filename)
                try:
                    if now - os.path.getmtime(file_path) > JOB_RETENTION_SECONDS:
                        os.remove(file_path)
                except Exception:
                    pass

        except Exception as e:
            print(f"Cleanup error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    cleanup_task = asyncio.create_task(cleanup_jobs())
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/videos", StaticFiles(directory=OUTPUT_DIR), name="videos")


def _cut_and_convert_clip(input_path: str, output_path: str,
                          start: float, end: float,
                          output_width: int = 1080, output_height: int = 1920):
    """Cut video segment and convert to 9:16 using FFmpeg filters only (no frame-by-frame)."""
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
        "-of", "csv=p=0", input_path
    ], capture_output=True, text=True, timeout=30)
    has_audio = "audio" in probe.stdout

    filter_complex = (
        f"[0:v]scale={output_width}:{output_height}:"
        f"force_original_aspect_ratio=increase,"
        f"crop={output_width}:{output_height},boxblur=30:5[b];"
        f"[0:v]scale={output_width}:{output_height}:"
        f"force_original_aspect_ratio=decrease[v];"
        f"[b][v]overlay=(main_w-overlay_w)/2:(main_h-overlay_h)/2,format=yuv420p"
    )

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-to", str(end),
        "-i", input_path,
        "-filter_complex", filter_complex,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
    ]

    if has_audio:
        cmd.extend(["-c:a", "aac", "-b:a", "128k"])

    cmd.append(output_path)

    subprocess.run(cmd, check=True, capture_output=True, timeout=300)


def _transcribe_only(input_path: str, groq_key: str) -> dict:
    import subprocess
    from groq import Groq

    stem = os.path.splitext(os.path.basename(input_path))[0]
    audio_path = os.path.join(os.path.dirname(input_path), f"{stem}.opus")
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-map", "a",
        "-ar", "16000", "-ac", "1",
        "-c:a", "libopus", "-b:a", "12k",
        audio_path
    ], check=True, capture_output=True)

    client = Groq(api_key=groq_key)
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            file=(os.path.basename(audio_path), f.read()),
            model="whisper-large-v3",
            response_format="verbose_json",
            temperature=0.0,
        )

    segments = []
    for seg in getattr(response, "segments", []):
        segments.append({
            "start": seg.get("start", 0),
            "end": seg.get("end", 0),
            "text": seg.get("text", "").strip(),
        })

    full_text = response.text.strip() if hasattr(response, "text") else str(response).strip()
    return {"transcript": full_text, "segments": segments}

async def _run_engine_process_job(job_id: str, input_path: str, output_dir: str,
                                  gemini_key: str = "", groq_key: str = "",
                                  transcript_only: bool = False):
    """Run engine pipeline + cut clips. Used by both /api/process and /api/engine/process."""

    def _sync_work():
        local_jobs = jobs
        try:
            local_jobs[job_id]['status'] = 'processing'
            local_jobs[job_id]['logs'].append("Engine pipeline starting...")

            os.environ["GROQ_API_KEY"] = groq_key
            if gemini_key:
                os.environ["GEMINI_API_KEY"] = gemini_key

            from engine.config import load_config_from_yaml
            load_config_from_yaml("engine_config.yaml")

            if transcript_only:
                result_data = _transcribe_only(input_path, groq_key)
                local_jobs[job_id]['status'] = 'completed'
                local_jobs[job_id]['result'] = result_data
                local_jobs[job_id]['logs'].append("Transcription complete.")
                return

            from engine.pipeline import Pipeline
            pipeline = Pipeline(video_id=job_id)
            result = pipeline.run(input_path)

            if result.error:
                local_jobs[job_id]['status'] = 'failed'
                local_jobs[job_id]['logs'].append(f"Engine error: {result.error}")
                return

            clips = []
            for i, candidate in enumerate(result.candidates[:10]):
                start = candidate['start']
                end = candidate['end']
                clip_filename = f"{job_id}_clip_{i+1}.mp4"
                clip_path = os.path.join(output_dir, clip_filename)

                try:
                    _cut_and_convert_clip(input_path, clip_path, start, end)
                except Exception as e:
                    local_jobs[job_id]['logs'].append(f"Clip {i+1} FFmpeg error: {e}")
                    continue

                if os.path.exists(clip_path) and os.path.getsize(clip_path) > 0:
                    hook_text = candidate.get('hook_text', '')
                    clips.append({
                        "video_url": f"/videos/{job_id}/{clip_filename}",
                        "start": start,
                        "end": end,
                        "total_duration": candidate.get('total_duration', end - start),
                        "hook_text": hook_text,
                        "hook_score": candidate.get('hook_score', 0),
                        "total_score": candidate.get('total_score', 0),
                        "video_title_for_youtube_short": hook_text[:100] if hook_text else "Viral Short",
                        "video_description_for_tiktok": hook_text,
                        "video_description_for_instagram": hook_text,
                        "viral_hook_text": hook_text[:50] if hook_text else "",
                    })

            local_jobs[job_id]['status'] = 'completed'
            local_jobs[job_id]['result'] = {'clips': clips, 'stats': result.stats}
            local_jobs[job_id]['logs'].append(
                f"Generated {len(clips)} clips (from {len(result.candidates)} candidates)")

        except Exception as e:
            local_jobs[job_id]['status'] = 'failed'
            local_jobs[job_id]['logs'].append(f"Error: {str(e)}")

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _sync_work)


@app.get("/api/config")
async def get_config():
    return {"youtubeUrlEnabled": not DISABLE_YOUTUBE_URL}


@app.post("/api/process")
async def process_endpoint(
    request: Request,
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    acknowledged: Optional[str] = Form(None)
):
    groq_key = request.headers.get("X-Groq-Key") or os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        raise HTTPException(status_code=400, detail="Missing X-Groq-Key header (required for engine pipeline)")
    gemini_key = request.headers.get("X-Gemini-Key", "")

    ack_flag = str(acknowledged).lower() in ("1", "true", "yes")

    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
        url = body.get("url")
        ack_flag = bool(body.get("acknowledged"))

    if not url and not file:
        raise HTTPException(status_code=400, detail="Must provide URL or File")

    if not ack_flag:
        raise HTTPException(status_code=400, detail="You must confirm you own the content or have rights to process it.")

    if url and DISABLE_YOUTUBE_URL:
        raise HTTPException(status_code=403, detail="YouTube URL ingest is disabled on this deployment. Please upload a file you own.")

    client_ip = request.client.host if request.client else "unknown"
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        client_ip = fwd.split(",")[0].strip()
    user_agent = request.headers.get("user-agent", "")
    attestation = {
        "acknowledged": True,
        "ip": client_ip,
        "user_agent": user_agent,
        "timestamp": time.time(),
        "source": "url" if url else "file",
    }

    job_id = str(uuid.uuid4())
    job_output_dir = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(job_output_dir, exist_ok=True)

    input_path = None
    if url:
        from yt_dlp import YoutubeDL
        ydl_opts = {
            **_COMMON_YDL_OPTS,
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "outtmpl": os.path.join(job_output_dir, "%(id)s.%(ext)s"),
            "overwrites": True,
        }
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                input_path = os.path.join(job_output_dir, f"{info['id']}.mp4")
                if not os.path.exists(input_path):
                    for f in os.listdir(job_output_dir):
                        if f.startswith(info['id']) and f.endswith('.mp4'):
                            input_path = os.path.join(job_output_dir, f)
                            break
                video_title = info.get('title', 'youtube_video')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"YouTube download failed: {str(e)}")
    else:
        input_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
        size = 0
        limit_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
        with open(input_path, "wb") as buffer:
            while content := await file.read(1024 * 1024):
                size += len(content)
                if size > limit_bytes:
                    os.remove(input_path)
                    shutil.rmtree(job_output_dir)
                    raise HTTPException(status_code=413, detail=f"File too large. Max size {MAX_FILE_SIZE_MB}MB")
                buffer.write(content)

    print(f"[attestation] job={job_id} ip={attestation['ip']} source={attestation['source']} ack=true")

    jobs[job_id] = {
        'status': 'queued',
        'logs': [f"Job {job_id} queued."],
        'result': None,
        'output_dir': job_output_dir,
        'attestation': attestation,
    }

    asyncio.create_task(_run_engine_process_job(
        job_id, input_path, job_output_dir,
        gemini_key=gemini_key, groq_key=groq_key
    ))

    return {"job_id": job_id, "status": "queued"}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    return {
        "status": job['status'],
        "logs": job['logs'],
        "result": job.get('result')
    }


from editor import VideoEditor
from subtitles import generate_srt, burn_subtitles, generate_srt_from_video
from hooks import add_hook_to_video


@app.post("/api/engine/process")
async def engine_process_endpoint(
    request: Request,
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    acknowledged: Optional[str] = Form(None),
    category: Optional[str] = Form("general"),
    mode: Optional[str] = Form("full"),
):
    groq_key = request.headers.get("X-Groq-Key") or os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        raise HTTPException(status_code=400, detail="Groq API key required for engine pipeline")
    gemini_key = request.headers.get("X-Gemini-Key", "")

    ack_flag = str(acknowledged).lower() in ("1", "true", "yes") if acknowledged else False

    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
        url = body.get("url")
        ack_flag = bool(body.get("acknowledged"))
        category = body.get("category", "general")
        mode = body.get("mode", "full")

    if not url and not file:
        raise HTTPException(status_code=400, detail="Must provide URL or File")
    if not ack_flag:
        raise HTTPException(status_code=400, detail="You must confirm you own the content or have rights to process it.")

    job_id = str(uuid.uuid4())
    job_output_dir = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(job_output_dir, exist_ok=True)

    input_path = None
    if url:
        from yt_dlp import YoutubeDL
        ydl_opts = {
            **_COMMON_YDL_OPTS,
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "outtmpl": os.path.join(job_output_dir, "%(id)s.%(ext)s"),
            "overwrites": True,
        }
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                input_path = os.path.join(job_output_dir, f"{info['id']}.mp4")
                if not os.path.exists(input_path):
                    for f in os.listdir(job_output_dir):
                        if f.startswith(info['id']) and f.endswith('.mp4'):
                            input_path = os.path.join(job_output_dir, f)
                            break
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"YouTube download failed: {str(e)}")
    elif file:
        input_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
        with open(input_path, "wb") as buffer:
            while content := await file.read(1024 * 1024):
                buffer.write(content)

    jobs[job_id] = {
        'status': 'queued',
        'logs': [f"Engine job {job_id} queued."],
        'result': None,
        'output_dir': job_output_dir,
    }

    asyncio.create_task(_run_engine_process_job(
        job_id, input_path, job_output_dir,
        gemini_key=gemini_key, groq_key=groq_key,
        transcript_only=(mode == "transcript")
    ))

    return {"job_id": job_id, "status": "queued"}


class EditRequest(BaseModel):
    job_id: str
    clip_index: int
    api_key: Optional[str] = None
    input_filename: Optional[str] = None


@app.post("/api/edit")
async def edit_clip(
    req: EditRequest,
    x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key")
):
    final_api_key = req.api_key or x_gemini_key or os.environ.get("GEMINI_API_KEY")

    if not final_api_key:
        raise HTTPException(status_code=400, detail="Missing Gemini API Key (Header or Body)")

    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[req.job_id]
    if 'result' not in job or 'clips' not in job['result']:
        raise HTTPException(status_code=400, detail="Job result not available")

    try:
        if req.input_filename:
            safe_name = os.path.basename(req.input_filename)
            input_path = os.path.join(OUTPUT_DIR, req.job_id, safe_name)
            filename = safe_name
        else:
            clip = job['result']['clips'][req.clip_index]
            filename = clip['video_url'].split('/')[-1]
            input_path = os.path.join(OUTPUT_DIR, req.job_id, filename)

        if not os.path.exists(input_path):
            raise HTTPException(status_code=404, detail=f"Video file not found: {input_path}")

        edited_filename = f"edited_{filename}"
        output_path = os.path.join(OUTPUT_DIR, req.job_id, edited_filename)

        def run_edit():
            editor = VideoEditor(api_key=final_api_key)

            safe_filename = f"temp_input_{req.job_id}.mp4"
            safe_input_path = os.path.join(OUTPUT_DIR, req.job_id, safe_filename)

            shutil.copy(input_path, safe_input_path)

            try:
                vid_file = editor.upload_video(safe_input_path)

                import cv2
                cap = cv2.VideoCapture(safe_input_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = frame_count / fps if fps else 0
                cap.release()

                transcript = None
                try:
                    meta_files = glob.glob(os.path.join(OUTPUT_DIR, req.job_id, "*_metadata.json"))
                    if meta_files:
                        with open(meta_files[0], 'r') as f:
                            data = json.load(f)
                            transcript = data.get('transcript')
                except Exception as e:
                    print(f"⚠️ Could not load transcript for editing context: {e}")

                filter_data = editor.get_ffmpeg_filter(vid_file, duration, fps=fps, width=width, height=height, transcript=transcript)

                safe_output_path = os.path.join(OUTPUT_DIR, req.job_id, f"temp_output_{req.job_id}.mp4")
                editor.apply_edits(safe_input_path, safe_output_path, filter_data)

                if os.path.exists(safe_output_path):
                    shutil.move(safe_output_path, output_path)

                return filter_data
            finally:
                if os.path.exists(safe_input_path):
                    os.remove(safe_input_path)

        loop = asyncio.get_event_loop()
        plan = await loop.run_in_executor(None, run_edit)

        new_video_url = f"/videos/{req.job_id}/{edited_filename}"

        return {
            "success": True,
            "new_video_url": new_video_url,
            "edit_plan": plan
        }

    except Exception as e:
        print(f"❌ Edit Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class SubtitleRequest(BaseModel):
    job_id: str
    clip_index: int
    position: str = "bottom"
    font_size: int = 16
    font_name: str = "Verdana"
    font_color: str = "#FFFFFF"
    border_color: str = "#000000"
    border_width: int = 2
    bg_color: str = "#000000"
    bg_opacity: float = 0.0
    input_filename: Optional[str] = None


@app.get("/api/clip/{job_id}/{clip_index}/transcript")
async def get_clip_transcript(job_id: str, clip_index: int):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    # Try legacy metadata.json (old pipeline) first
    output_dir = os.path.join(OUTPUT_DIR, job_id)
    json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
    if json_files:
        with open(json_files[0], 'r') as f:
            data = json.load(f)
        transcript = data.get('transcript')
        clips = data.get('shorts', [])
        if transcript and clip_index < len(clips):
            clip_data = clips[clip_index]
            clip_start = clip_data.get('start', 0)
            clip_end = clip_data.get('end', 0)
            captions = []
            for segment in transcript.get('segments', []):
                for word_info in segment.get('words', []):
                    if word_info['end'] > clip_start and word_info['start'] < clip_end:
                        captions.append({
                            "text": word_info.get('word', '').strip(),
                            "startMs": int((max(0, word_info['start'] - clip_start)) * 1000),
                            "endMs": int((max(0, word_info['end'] - clip_start)) * 1000),
                        })
            return {
                "captions": captions,
                "durationSec": clip_end - clip_start,
                "language": transcript.get('language', 'en'),
            }

    # New engine pipeline — return clip timing without word-level captions
    result = jobs[job_id].get('result')
    if not result or 'clips' not in result:
        raise HTTPException(status_code=400, detail="Clip data not available")

    clips = result['clips']
    if clip_index >= len(clips):
        raise HTTPException(status_code=404, detail="Clip not found")

    clip = clips[clip_index]
    clip_start = clip.get('start', 0)
    clip_end = clip.get('end', 0)
    duration_sec = clip_end - clip_start

    return {
        "captions": [],
        "durationSec": duration_sec,
        "language": "en",
    }


@app.post("/api/subtitle")
async def add_subtitles(req: SubtitleRequest):
    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[req.job_id]
    output_dir = os.path.join(OUTPUT_DIR, req.job_id)

    # Try legacy metadata.json first, then new engine result
    clip_start = clip_end = 0
    transcript = None
    has_transcript_data = False

    json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
    if json_files:
        with open(json_files[0], 'r') as f:
            data = json.load(f)
        transcript = data.get('transcript')
        clips = data.get('shorts', [])
        if transcript and req.clip_index < len(clips):
            clip_data = clips[req.clip_index]
            clip_start = clip_data.get('start', 0)
            clip_end = clip_data.get('end', 0)
            has_transcript_data = True

    if not has_transcript_data:
        # New engine pipeline — use result clips
        result = job.get('result')
        if result and 'clips' in result and req.clip_index < len(result['clips']):
            clip = result['clips'][req.clip_index]
            clip_start = clip.get('start', 0)
            clip_end = clip.get('end', 0)

    if req.input_filename:
        filename = os.path.basename(req.input_filename)
    else:
        result = job.get('result')
        if result and 'clips' in result and req.clip_index < len(result['clips']):
            clip = result['clips'][req.clip_index]
            filename = clip.get('video_url', '').split('/')[-1]
            if not filename:
                filename = f"{job_id}_clip_{req.clip_index+1}.mp4"
        else:
            filename = f"{job_id}_clip_{req.clip_index+1}.mp4"

    input_path = os.path.join(output_dir, filename)
    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail=f"Video file not found: {input_path}")

    srt_filename = f"subs_{req.clip_index}_{int(time.time())}.srt"
    srt_path = os.path.join(output_dir, srt_filename)
    output_filename = f"subtitled_{filename}"
    output_path = os.path.join(output_dir, output_filename)

    try:
        if transcript and has_transcript_data:
            success = generate_srt(transcript, clip_start, clip_end, srt_path)
            if not success:
                raise HTTPException(status_code=400, detail="No words found for this clip range.")
        else:
            raise HTTPException(status_code=400, detail="Word-level transcript not available for subtitle generation.")

        def run_burn():
            burn_subtitles(input_path, srt_path, output_path,
                          alignment=req.position, fontsize=req.font_size,
                          font_name=req.font_name, font_color=req.font_color,
                          border_color=req.border_color, border_width=req.border_width,
                          bg_color=req.bg_color, bg_opacity=req.bg_opacity)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, run_burn)

    except Exception as e:
        print(f"Subtitle Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if job.get('result') and 'clips' in job['result'] and req.clip_index < len(job['result']['clips']):
        job['result']['clips'][req.clip_index]['video_url'] = f"/videos/{req.job_id}/{output_filename}"

    return {
        "success": True,
        "new_video_url": f"/videos/{req.job_id}/{output_filename}"
    }


class HookRequest(BaseModel):
    job_id: str
    clip_index: int
    text: str
    input_filename: Optional[str] = None
    position: Optional[str] = "top"
    size: Optional[str] = "M"


@app.post("/api/hook")
async def add_hook(req: HookRequest):
    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[req.job_id]
    output_dir = os.path.join(OUTPUT_DIR, req.job_id)

    if req.input_filename:
        filename = os.path.basename(req.input_filename)
    else:
        result = job.get('result')
        if result and 'clips' in result and req.clip_index < len(result['clips']):
            clip = result['clips'][req.clip_index]
            filename = clip.get('video_url', '').split('/')[-1]
        else:
            filename = f"{job_id}_clip_{req.clip_index+1}.mp4"

    input_path = os.path.join(output_dir, filename)
    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail=f"Video file not found: {input_path}")

    output_filename = f"hook_{filename}"
    output_path = os.path.join(output_dir, output_filename)

    size_map = {"S": 0.8, "M": 1.0, "L": 1.3}
    font_scale = size_map.get(req.size, 1.0)

    try:
        def run_hook():
            add_hook_to_video(input_path, req.text, output_path, position=req.position, font_scale=font_scale)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, run_hook)

    except Exception as e:
        print(f"Hook Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if job.get('result') and 'clips' in job['result'] and req.clip_index < len(job['result']['clips']):
        job['result']['clips'][req.clip_index]['video_url'] = f"/videos/{req.job_id}/{output_filename}"

    return {
        "success": True,
        "new_video_url": f"/videos/{req.job_id}/{output_filename}"
    }


@app.get("/api/health")
async def health_check():
    return {"status": "ok"}
