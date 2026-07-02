import os
import sys
import uuid
import subprocess
import threading
import json
import shutil
import glob
import time
import asyncio
from dotenv import load_dotenv
from typing import Dict, Optional, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

# Constants
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", "5"))
MAX_FILE_SIZE_MB = 2048
JOB_RETENTION_SECONDS = 3600
DISABLE_YOUTUBE_URL = os.environ.get("DISABLE_YOUTUBE_URL", "false").lower() in ("1", "true", "yes")

# Application State
job_queue = asyncio.Queue()
jobs: Dict[str, Dict] = {}
concurrency_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)


def _relocate_root_job_artifacts(job_id: str, job_output_dir: str) -> bool:
    try:
        os.makedirs(job_output_dir, exist_ok=True)
        pattern = os.path.join(OUTPUT_DIR, f"{job_id}_*_metadata.json")
        meta_candidates = sorted(glob.glob(pattern), key=lambda p: os.path.getmtime(p), reverse=True)
        if not meta_candidates:
            return False

        metadata_path = meta_candidates[0]
        base_name = os.path.basename(metadata_path).replace("_metadata.json", "")

        dest_metadata = os.path.join(job_output_dir, os.path.basename(metadata_path))
        if os.path.abspath(metadata_path) != os.path.abspath(dest_metadata):
            shutil.move(metadata_path, dest_metadata)

        clip_pattern = os.path.join(OUTPUT_DIR, f"{base_name}_clip_*.mp4")
        for clip_path in glob.glob(clip_pattern):
            dest_clip = os.path.join(job_output_dir, os.path.basename(clip_path))
            if os.path.abspath(clip_path) != os.path.abspath(dest_clip):
                shutil.move(clip_path, dest_clip)

        temp_clip_pattern = os.path.join(OUTPUT_DIR, f"temp_{base_name}_clip_*.mp4")
        for clip_path in glob.glob(temp_clip_pattern):
            dest_clip = os.path.join(job_output_dir, os.path.basename(clip_path))
            if os.path.abspath(clip_path) != os.path.abspath(dest_clip):
                shutil.move(clip_path, dest_clip)

        return True
    except Exception:
        return False


async def cleanup_jobs():
    while True:
        try:
            await asyncio.sleep(300)
            now = time.time()

            for job_id in os.listdir(OUTPUT_DIR):
                job_path = os.path.join(OUTPUT_DIR, job_id)
                if os.path.isdir(job_path):
                    if now - os.path.getmtime(job_path) > JOB_RETENTION_SECONDS:
                        print(f"🧹 Purging old job: {job_id}")
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
            print(f"⚠️ Cleanup error: {e}")


async def process_queue():
    print(f"🚀 Job Queue Worker started with {MAX_CONCURRENT_JOBS} concurrent slots.")
    while True:
        try:
            job_id = await job_queue.get()
            await concurrency_semaphore.acquire()
            print(f"🔄 Acquired slot for job: {job_id}")
            asyncio.create_task(run_job_wrapper(job_id))
        except Exception as e:
            print(f"❌ Queue dispatch error: {e}")
            await asyncio.sleep(1)


async def run_job_wrapper(job_id):
    try:
        job = jobs.get(job_id)
        if job:
            await run_job(job_id, job)
    except Exception as e:
        print(f"❌ Job wrapper error {job_id}: {e}")
    finally:
        concurrency_semaphore.release()
        job_queue.task_done()
        print(f"✅ Released slot for job: {job_id}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    worker_task = asyncio.create_task(process_queue())
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


class ProcessRequest(BaseModel):
    url: str


def enqueue_output(out, job_id):
    try:
        for line in iter(out.readline, b''):
            decoded_line = line.decode('utf-8').strip()
            if decoded_line:
                print(f"📝 [Job Output] {decoded_line}")
                if job_id in jobs:
                    jobs[job_id]['logs'].append(decoded_line)
    except Exception as e:
        print(f"Error reading output for job {job_id}: {e}")
    finally:
        out.close()


async def run_job(job_id, job_data):
    cmd = job_data['cmd']
    env = job_data['env']
    output_dir = job_data['output_dir']

    jobs[job_id]['status'] = 'processing'
    jobs[job_id]['logs'].append("Job started by worker.")
    print(f"🎬 [run_job] Executing command for {job_id}: {' '.join(cmd)}")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=os.getcwd()
        )

        t_log = threading.Thread(target=enqueue_output, args=(process.stdout, job_id))
        t_log.daemon = True
        t_log.start()

        start_wait = time.time()
        while process.poll() is None:
            await asyncio.sleep(2)

            try:
                json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
                if json_files:
                    target_json = json_files[0]
                    if os.path.getsize(target_json) > 0:
                        with open(target_json, 'r') as f:
                            data = json.load(f)

                        base_name = os.path.basename(target_json).replace('_metadata.json', '')
                        clips = data.get('shorts', [])
                        cost_analysis = data.get('cost_analysis')

                        ready_clips = []
                        for i, clip in enumerate(clips):
                            clip_filename = f"{base_name}_clip_{i+1}.mp4"
                            clip_path = os.path.join(output_dir, clip_filename)
                            if os.path.exists(clip_path) and os.path.getsize(clip_path) > 0:
                                clip['video_url'] = f"/videos/{job_id}/{clip_filename}"
                                ready_clips.append(clip)

                        if ready_clips:
                            jobs[job_id]['result'] = {'clips': ready_clips, 'cost_analysis': cost_analysis}
            except Exception:
                pass

        returncode = process.returncode

        if returncode == 0:
            jobs[job_id]['status'] = 'completed'
            jobs[job_id]['logs'].append("Process finished successfully.")

            json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
            if not json_files:
                if _relocate_root_job_artifacts(job_id, output_dir):
                    json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
            if json_files:
                target_json = json_files[0]
                with open(target_json, 'r') as f:
                    data = json.load(f)

                base_name = os.path.basename(target_json).replace('_metadata.json', '')
                clips = data.get('shorts', [])
                cost_analysis = data.get('cost_analysis')

                for i, clip in enumerate(clips):
                    clip_filename = f"{base_name}_clip_{i+1}.mp4"
                    clip['video_url'] = f"/videos/{job_id}/{clip_filename}"

                jobs[job_id]['result'] = {'clips': clips, 'cost_analysis': cost_analysis}
            else:
                jobs[job_id]['status'] = 'failed'
                jobs[job_id]['logs'].append("No metadata file generated.")
        else:
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['logs'].append(f"Process failed with exit code {returncode}")

    except Exception as e:
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['logs'].append(f"Execution error: {str(e)}")


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
    api_key = request.headers.get("X-Gemini-Key")
    if not api_key:
        raise HTTPException(status_code=400, detail="Missing X-Gemini-Key header")
    groq_key = request.headers.get("X-Groq-Key") or os.environ.get("GROQ_API_KEY", "")

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

    cmd = [sys.executable, "-u", "main.py"]
    env = os.environ.copy()
    env["GEMINI_API_KEY"] = api_key
    if groq_key:
        env["GROQ_API_KEY"] = groq_key

    if url:
        cmd.extend(["-u", url])
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

        cmd.extend(["-i", input_path])

    cmd.extend(["-o", job_output_dir])

    print(f"[attestation] job={job_id} ip={attestation['ip']} source={attestation['source']} ack=true")

    jobs[job_id] = {
        'status': 'queued',
        'logs': [f"Job {job_id} queued."],
        'cmd': cmd,
        'env': env,
        'output_dir': job_output_dir,
        'attestation': attestation
    }

    await job_queue.put(job_id)

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


class EngineProcessRequest(BaseModel):
    url: Optional[str] = None
    category: str = "general"
    acknowledged: bool = False


@app.post("/api/engine/process")
async def engine_process_endpoint(
    req: EngineProcessRequest,
    request: Request,
    file: Optional[UploadFile] = File(None),
):
    api_key = request.headers.get("X-Gemini-Key")
    groq_key = request.headers.get("X-Groq-Key") or os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        raise HTTPException(status_code=400, detail="Groq API key required for engine pipeline")

    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
        req = EngineProcessRequest(**body)

    if not req.url and not file:
        raise HTTPException(status_code=400, detail="Must provide URL or File")
    if not req.acknowledged:
        raise HTTPException(status_code=400, detail="You must confirm you own the content or have rights to process it.")

    job_id = str(uuid.uuid4())
    job_output_dir = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(job_output_dir, exist_ok=True)

    input_path = None
    if req.url:
        from yt_dlp import YoutubeDL
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": os.path.join(job_output_dir, "%(id)s.%(ext)s"),
            "quiet": True,
        }
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(req.url, download=True)
            input_path = os.path.join(job_output_dir, f"{info['id']}.{info['ext']}")
    elif file:
        input_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
        with open(input_path, "wb") as buffer:
            while content := await file.read(1024 * 1024):
                buffer.write(content)

    os.environ["GROQ_API_KEY"] = groq_key
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key

    jobs[job_id] = {
        'status': 'queued',
        'logs': [f"Engine job {job_id} queued."],
        'result': None,
        'output_dir': job_output_dir,
    }

    asyncio.create_task(_run_engine_job(job_id, input_path, req.category, job_output_dir))

    return {"job_id": job_id, "status": "queued"}


async def _run_engine_job(job_id: str, input_path: str, category: str, output_dir: str):
    try:
        jobs[job_id]['status'] = 'processing'
        jobs[job_id]['logs'].append("Engine pipeline starting...")

        from engine.pipeline import Pipeline
        pipeline = Pipeline(video_id=job_id)

        def run_sync():
            return pipeline.run(input_path, category=category)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, run_sync)

        if result.error:
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['logs'].append(f"Engine error: {result.error}")
        else:
            jobs[job_id]['status'] = 'completed'
            jobs[job_id]['result'] = {
                'video_id': result.video_id,
                'candidates': result.candidates,
                'best_clip': result.best_clip,
                'stats': result.stats,
            }
            jobs[job_id]['logs'].append(f"Engine complete: {len(result.candidates)} candidates, best score {result.best_clip.get('total_score', 0) if result.best_clip else 0}")
    except Exception as e:
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['logs'].append(f"Engine error: {str(e)}")


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

    output_dir = os.path.join(OUTPUT_DIR, job_id)
    json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))

    if not json_files:
        raise HTTPException(status_code=404, detail="Metadata not found")

    with open(json_files[0], 'r') as f:
        data = json.load(f)

    transcript = data.get('transcript')
    if not transcript:
        raise HTTPException(status_code=400, detail="Transcript not found in metadata")

    clips = data.get('shorts', [])
    if clip_index >= len(clips):
        raise HTTPException(status_code=404, detail="Clip not found")

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

    duration_sec = clip_end - clip_start

    return {
        "captions": captions,
        "durationSec": duration_sec,
        "language": transcript.get('language', 'en'),
    }


@app.post("/api/subtitle")
async def add_subtitles(req: SubtitleRequest):
    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[req.job_id]

    output_dir = os.path.join(OUTPUT_DIR, req.job_id)
    json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))

    if not json_files:
        raise HTTPException(status_code=404, detail="Metadata not found")

    with open(json_files[0], 'r') as f:
        data = json.load(f)

    transcript = data.get('transcript')
    if not transcript:
        raise HTTPException(status_code=400, detail="Transcript not found in metadata. Please process a new video.")

    clips = data.get('shorts', [])
    if req.clip_index >= len(clips):
        raise HTTPException(status_code=404, detail="Clip not found")

    clip_data = clips[req.clip_index]

    if req.input_filename:
        filename = os.path.basename(req.input_filename)
    else:
        filename = clip_data.get('video_url', '').split('/')[-1]
        if not filename:
            base_name = os.path.basename(json_files[0]).replace('_metadata.json', '')
            filename = f"{base_name}_clip_{req.clip_index+1}.mp4"

    input_path = os.path.join(output_dir, filename)
    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail=f"Video file not found: {input_path}")

    srt_filename = f"subs_{req.clip_index}_{int(time.time())}.srt"
    srt_path = os.path.join(output_dir, srt_filename)

    output_filename = f"subtitled_{filename}"
    output_path = os.path.join(output_dir, output_filename)

    try:
        is_dubbed = filename.startswith("translated_")

        if is_dubbed:
            print(f"🎙️ Dubbed video detected, transcribing audio for subtitles...")
            def run_transcribe_srt():
                return generate_srt_from_video(input_path, srt_path)

            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(None, run_transcribe_srt)
        else:
            success = generate_srt(transcript, clip_data['start'], clip_data['end'], srt_path)

        if not success:
            raise HTTPException(status_code=400, detail="No words found for this clip range.")

        def run_burn():
            burn_subtitles(input_path, srt_path, output_path,
                          alignment=req.position, fontsize=req.font_size,
                          font_name=req.font_name, font_color=req.font_color,
                          border_color=req.border_color, border_width=req.border_width,
                          bg_color=req.bg_color, bg_opacity=req.bg_opacity)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, run_burn)

    except Exception as e:
        print(f"❌ Subtitle Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if req.clip_index < len(job['result']['clips']):
        job['result']['clips'][req.clip_index]['video_url'] = f"/videos/{req.job_id}/{output_filename}"

    try:
        if req.clip_index < len(clips):
            clips[req.clip_index]['video_url'] = f"/videos/{req.job_id}/{output_filename}"
            data['shorts'] = clips

            with open(json_files[0], 'w') as f:
                json.dump(data, f, indent=4)
                print(f"✅ Metadata updated with subtitled video for clip {req.clip_index}")
    except Exception as e:
        print(f"⚠️ Failed to update metadata.json: {e}")

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
    json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))

    if not json_files:
        raise HTTPException(status_code=404, detail="Metadata not found")

    with open(json_files[0], 'r') as f:
        data = json.load(f)

    clips = data.get('shorts', [])
    if req.clip_index >= len(clips):
        raise HTTPException(status_code=404, detail="Clip not found")

    clip_data = clips[req.clip_index]

    if req.input_filename:
        filename = os.path.basename(req.input_filename)
    else:
        filename = clip_data.get('video_url', '').split('/')[-1]
        if not filename:
            base_name = os.path.basename(json_files[0]).replace('_metadata.json', '')
            filename = f"{base_name}_clip_{req.clip_index+1}.mp4"

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
        print(f"❌ Hook Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if req.clip_index < len(job['result']['clips']):
        job['result']['clips'][req.clip_index]['video_url'] = f"/videos/{req.job_id}/{output_filename}"

    try:
        if req.clip_index < len(clips):
            clips[req.clip_index]['video_url'] = f"/videos/{req.job_id}/{output_filename}"
            data['shorts'] = clips
            with open(json_files[0], 'w') as f:
                json.dump(data, f, indent=4)
                print(f"✅ Metadata updated with hook video for clip {req.clip_index}")
    except Exception as e:
        print(f"⚠️ Failed to update metadata.json: {e}")

    return {
        "success": True,
        "new_video_url": f"/videos/{req.job_id}/{output_filename}"
    }


@app.get("/api/health")
async def health_check():
    return {"status": "ok"}
