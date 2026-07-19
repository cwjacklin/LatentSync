import os
import sys
import uuid
import subprocess
import shutil
import logging
import time
import threading
import re
import json
import signal
import tempfile
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Header, Query, Request, Depends
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

def get_env_name(env_header: str) -> str:
    if not env_header: return "prod"
    env_header = env_header.lower()
    if env_header == "development": return "dev"
    if env_header == "production": return "prod"
    return env_header


SHARED_STORAGE_BASE = os.environ.get("SHARED_STORAGE_BASE", "/data/realfakeai")

# Default to GPU 1 for LatentSync (GPU 0 reserved for GPT-SoVITS/LivePortrait)
# Override at runtime with CUDA_VISIBLE_DEVICES if needed.
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "1")

DEFAULT_LATENTSYNC_VERSION = os.environ.get("LATENTSYNC_VERSION", "1.5")

LATENTSYNC_VERSION_CONFIGS = {
    "1.5": {
        "config": "configs/unet/stage2.yaml",
        "checkpoint": "checkpoints/latentsync_unet_256.pt",
    },
    "1.6": {
        "config": "configs/unet/stage2_512.yaml",
        "checkpoint": "checkpoints/latentsync_unet.pt",
    },
}


def get_latentsync_gpu() -> str:
    return os.environ.get("CUDA_VISIBLE_DEVICES", "1")


def normalize_latentsync_version(value: Optional[str]) -> str:
    raw = (value or DEFAULT_LATENTSYNC_VERSION).strip().lower()

    aliases = {
        "1.5": "1.5",
        "v1.5": "1.5",
        "latentsync-1.5": "1.5",
        "latentsync_1.5": "1.5",
        "1.6": "1.6",
        "v1.6": "1.6",
        "latentsync-1.6": "1.6",
        "latentsync_1.6": "1.6",
    }

    if raw not in aliases:
        allowed = ", ".join(sorted(LATENTSYNC_VERSION_CONFIGS.keys()))
        raise ValueError(f"Unsupported LatentSync version '{value}'. Allowed versions: {allowed}")

    return aliases[raw]


def get_latentsync_paths(version: str) -> tuple[str, str]:
    config = LATENTSYNC_VERSION_CONFIGS[version]
    return config["config"], config["checkpoint"]


def get_requested_latentsync_version(value: Optional[str]) -> str:
    try:
        return normalize_latentsync_version(value)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# Basic Observability / Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("LatentSyncAPI")


# --- Standardized Response Models ---

class StatusResponse(BaseModel):
    status: str  # "ok" for healthy
    service: str

class JobCreatedResponse(BaseModel):
    job_id: str
    status: str  # Always "queued"

class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # queued | processing | completed | failed | cancelled
    progress: float  # 0-100
    type: Optional[str] = None
    user_id: Optional[str] = None
    created_at: Optional[str] = None
    latentsync_version: Optional[str] = None
    times: Optional[Dict[str, float]] = None
    error: Optional[str] = None
    result_path: Optional[str] = None

class JobListResponse(BaseModel):
    jobs: List[Dict[str, Any]]

class CancelResponse(BaseModel):
    job_id: str
    status: str
    message: str

class ErrorResponse(BaseModel):
    detail: str

class VideoSyncRequest(BaseModel):
    video_s3_key: str
    audio_s3_key: str
    user_id: str
    s3_bucket: str
    output_key: str
    webhook_url: str
    mock_run: bool = False

class PortraitSyncRequest(BaseModel):
    image_s3_key: str
    driving_video_s3_key: str
    audio_s3_key: str
    user_id: str
    s3_bucket: str
    output_key: str
    webhook_url: str
    mock_run: bool = False

class LivePortraitAnimationRequest(BaseModel):
    image_s3_key: str
    user_id: str
    s3_bucket: str
    output_key: str
    webhook_url: str
    mock_run: bool = False


# Job tracking
jobs = {}
active_processes = {}
JOBS_FILE = "jobs.json"
gpu_lock = threading.Lock()

def _load_jobs():
    global jobs
    if os.path.exists(JOBS_FILE):
        try:
            with open(JOBS_FILE, "r") as f:
                jobs = json.load(f)
            # Mark stuck jobs as failed since their background task is dead
            for jid, jdata in jobs.items():
                if jdata["status"] in ("queued", "processing"):
                    jdata["status"] = "failed"
                    jdata["error"] = "API Server restarted"
        except Exception as e:
            logger.error(f"Failed to load jobs: {e}")

def _save_jobs():
    try:
        with open(JOBS_FILE, "w") as f:
            json.dump(jobs, f)
    except Exception as e:
        logger.error(f"Failed to save jobs: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_jobs()
    yield
    logger.info("API Server shutting down, terminating active processes...")
    for job_id, process in list(active_processes.items()):
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except Exception:
            pass
    _save_jobs()


app = FastAPI(
    title="LatentSync Enterprise API",
    lifespan=lifespan,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)

# Directories for temp files
ASSETS_DIR = os.path.abspath("uploaded_media")  # Directory for default assets like driving videos
os.makedirs(ASSETS_DIR, exist_ok=True)


def _job_response(job_id: str, job_data: dict) -> dict:
    """Build a standardized job status response."""
    return {
        "job_id": job_id,
        "status": job_data.get("status", "unknown"),
        "progress": job_data.get("progress", 0),
        "type": job_data.get("type"),
        "user_id": job_data.get("user_id"),
        "created_at": job_data.get("created_at"),
        "latentsync_version": job_data.get("latentsync_version"),
        "times": job_data.get("times"),
        "error": job_data.get("error"),
        "result_path": job_data.get("result_path"),
    }


def process_video_job(job_id: str, video_path: str, audio_path: str, job_dir: str, latentsync_version: str, s3_bucket: str, output_key: str, webhook_url: str, mock_run: bool = False):
    import os
    import s3_utils
    jobs[job_id]["status"] = "processing"
    jobs[job_id]["times"] = {}
    jobs[job_id]["progress"] = 0
    log_path = os.path.join(job_dir, f"{job_id}.log")
    jobs[job_id]["log_path"] = log_path
    open(log_path, 'a').close()  # create empty file immediately
    _save_jobs()
    start_time = time.time()
    try:
        if mock_run:
            logger.info(f"MOCK RUN for {job_id}: Skipping LatentSync model inference.")
            time.sleep(2)
            jobs[job_id]["times"]["total"] = 2.0
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["result_path"] = output_key
            _save_jobs()
            s3_utils.notify_webhook(webhook_url, {
                "job_id": job_id,
                "status": "completed",
                "result_path": output_key,
                "mock_run": True
            })
            return
            
        final_video_path = run_latentsync(video_path, audio_path, job_id, job_dir, latentsync_version)
        if not final_video_path: return
        
        s3_utils.upload_to_s3(final_video_path, output_key, s3_bucket)
        
        end_time = time.time()
        jobs[job_id]["times"]["latentsync_inference"] = round(end_time - start_time, 2)
        jobs[job_id]["times"]["total"] = round(end_time - start_time, 2)
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result_path"] = output_key
        _save_jobs()
        
        s3_utils.notify_webhook(webhook_url, {
            "job_id": job_id,
            "status": "completed",
            "result_path": output_key
        })
    except Exception as e:
        if jobs.get(job_id, {}).get("status") != "cancelled":
            logger.error(f"Job {job_id} failed: {e}")
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)
            _save_jobs()

def process_image_job(job_id: str, img_path: str, vid_path: str, aud_path: str, job_dir: str, latentsync_version: str, s3_bucket: str, output_key: str, webhook_url: str, mock_run: bool = False):
    import os
    import s3_utils
    jobs[job_id]["status"] = "processing"
    jobs[job_id]["times"] = {}
    jobs[job_id]["progress"] = 0
    log_path = os.path.join(job_dir, f"{job_id}.log")
    jobs[job_id]["log_path"] = log_path
    open(log_path, 'a').close()  # create empty file immediately
    _save_jobs()
    total_start = time.time()
    try:
        if mock_run:
            logger.info(f"MOCK RUN for {job_id}: Skipping LivePortrait and LatentSync model inference.")
            time.sleep(2)
            jobs[job_id]["times"]["total"] = 2.0
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["result_path"] = output_key
            _save_jobs()
            s3_utils.notify_webhook(webhook_url, {
                "job_id": job_id,
                "status": "completed",
                "result_path": output_key,
                "mock_run": True
            })
            return
            
        lp_start = time.time()
        lp_video_path = run_liveportrait(img_path, vid_path, job_id, job_dir)
        if not lp_video_path: return
        lp_end = time.time()
        jobs[job_id]["times"]["liveportrait_inference"] = round(lp_end - lp_start, 2)
        
        ls_start = time.time()
        final_video_path = run_latentsync(lp_video_path, aud_path, job_id, job_dir, latentsync_version)
        if not final_video_path: return
        ls_end = time.time()
        jobs[job_id]["times"]["latentsync_inference"] = round(ls_end - ls_start, 2)
        
        s3_utils.upload_to_s3(final_video_path, output_key, s3_bucket)

        jobs[job_id]["times"]["total"] = round(time.time() - total_start, 2)
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result_path"] = output_key
        _save_jobs()
        
        s3_utils.notify_webhook(webhook_url, {
            "job_id": job_id,
            "status": "completed",
            "result_path": output_key
        })
    except Exception as e:
        if jobs.get(job_id, {}).get("status") != "cancelled":
            logger.error(f"Job {job_id} failed: {e}")
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)
            _save_jobs()

def process_liveportrait_job(job_id: str, img_path: str, vid_path: str, job_dir: str, s3_bucket: str, output_key: str, webhook_url: str, mock_run: bool = False):
    import os
    import s3_utils
    jobs[job_id]["status"] = "processing"
    jobs[job_id]["times"] = {}
    jobs[job_id]["progress"] = 0
    log_path = os.path.join(job_dir, f"{job_id}.log")
    jobs[job_id]["log_path"] = log_path
    open(log_path, 'a').close()  # create empty file immediately
    _save_jobs()
    total_start = time.time()
    try:
        if mock_run:
            logger.info(f"MOCK RUN for {job_id}: Skipping LivePortrait model inference.")
            time.sleep(2)
            jobs[job_id]["times"]["total"] = 2.0
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["result_path"] = output_key
            _save_jobs()
            s3_utils.notify_webhook(webhook_url, {
                "job_id": job_id,
                "status": "completed",
                "result_path": output_key,
                "mock_run": True
            })
            return
            
        lp_start = time.time()
        lp_video_path = run_liveportrait(img_path, vid_path, job_id, job_dir)
        if not lp_video_path: return
        lp_end = time.time()
        jobs[job_id]["times"]["liveportrait_inference"] = round(lp_end - lp_start, 2)
        
        s3_utils.upload_to_s3(lp_video_path, output_key, s3_bucket)

        jobs[job_id]["times"]["total"] = round(time.time() - total_start, 2)
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result_path"] = output_key
        _save_jobs()
        
        s3_utils.notify_webhook(webhook_url, {
            "job_id": job_id,
            "status": "completed",
            "result_path": output_key
        })
    except Exception as e:
        if jobs.get(job_id, {}).get("status") != "cancelled":
            logger.error(f"Job {job_id} failed: {e}")
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)
            _save_jobs()


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/status", response_model=StatusResponse)
async def status():
    """Health check endpoint for uptime monitoring."""
    return {"status": "ok", "service": "LatentSyncAPI"}


# ============================================================
# JOB MANAGEMENT (standardized across all APIs)
# ============================================================

@app.get("/jobs", response_model=JobListResponse)
async def list_jobs(user_id: Optional[str] = Query(None)):
    """List all jobs, optionally filtered by user_id."""
    result = []
    for jid, jdata in jobs.items():
        if user_id and jdata.get("user_id") != user_id:
            continue
        result.append(_job_response(jid, jdata))
    return {"jobs": result}


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a specific job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_response(job_id, jobs[job_id])


@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a pending or processing job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job["status"] in ("completed", "failed", "cancelled"):
        raise HTTPException(status_code=400, detail=f"Cannot cancel job: already {job['status']}")
    
    jobs[job_id]["status"] = "cancelled"
    _save_jobs()
    
    if job_id in active_processes:
        process = active_processes[job_id]
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        except Exception as e:
            logger.warning(f"Error killing process group: {e}")
            try:
                process.terminate()
            except:
                pass
                
    return {"job_id": job_id, "status": "cancelled", "message": "Job cancellation requested"}


@app.get("/jobs/{job_id}/download")
async def download_result(job_id: str):
    """Download the result file for a completed job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = jobs[job_id]
    if job_data["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job is not completed yet")
        
    if "result_path" not in job_data:
        raise HTTPException(status_code=404, detail="Result path not available")
        
    result_path = job_data["result_path"]
    
    if result_path and result_path.startswith("user_assets/"):
        s3_bucket = job_data.get("s3_bucket", os.environ.get("S3_BUCKET"))
        if s3_bucket:
            import s3_utils
            tmp_path = tempfile.mktemp(suffix=".mp4")
            s3_utils.download_from_s3(result_path, tmp_path, s3_bucket)
            return FileResponse(
                path=tmp_path,
                media_type="video/mp4",
                filename=f"latentsync_result_{job_id}.mp4"
            )
        else:
            return JSONResponse(content={"s3_path": result_path})
    
    if not result_path or not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Result file missing")
        
    return FileResponse(
        path=result_path,
        media_type="video/mp4",
        filename=f"latentsync_result_{job_id}.mp4"
    )


@app.get("/jobs/{job_id}/log")
async def get_job_log(job_id: str, lines: int = 200):
    """Return the last `lines` lines from the job log file."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    job_dir = job.get("job_dir")
    log_path = job.get("log_path") or (os.path.join(job_dir, f"{job_id}.log") if job_dir else None)

    if not log_path or not os.path.exists(log_path):
        raise HTTPException(status_code=404, detail="Log not found")

    try:
        with open(log_path, "r") as f:
            all_lines = f.readlines()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    start = max(0, len(all_lines) - int(lines))
    tail = [l.rstrip('\n') for l in all_lines[start:]]
    return JSONResponse(content={"job_id": job_id, "log": tail})


# ============================================================
# INFERENCE (create jobs)
# ============================================================

def run_latentsync(base_video_path: str, audio_path: str, job_id: str, job_dir: str, latentsync_version: str) -> str:
    """
    Runs the LatentSync inference script via subprocess.
    """
    output_video_path = os.path.join(job_dir, f"{job_id}_synced.mp4")
    
    version = normalize_latentsync_version(latentsync_version)
    config_path, ckpt_path = get_latentsync_paths(version)

    command = [
        "python", "-m", "scripts.inference",
        "--unet_config_path", config_path,
        "--inference_ckpt_path", ckpt_path,
        "--inference_steps", "20",
        "--guidance_scale", "1.5",
        "--enable_deepcache",
        "--video_path", base_video_path,
        "--audio_path", audio_path,
        "--video_out_path", output_video_path
    ]
    
    logger.info(f"Running LatentSync {version} with config={config_path} checkpoint={ckpt_path}")
    logger.info(f"Waiting for LatentSync lock...")
    with gpu_lock:
        if jobs.get(job_id, {}).get("status") == "cancelled":
            logger.info(f"Job {job_id} was cancelled before LatentSync started. Aborting.")
            return None
            
        jobs[job_id]["status"] = "processing"
        _save_jobs()
        logger.info(f"Acquired lock. Running LatentSync command: {' '.join(command)}")
        start_time = time.time()
        process = None
        
        try:
            # ensure child python process is unbuffered so we can stream progress
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            # If using the system python, prefer -u to force unbuffered mode as well
            if len(command) > 0 and command[0] == "python":
                command = [command[0], "-u"] + command[1:]

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env,
                start_new_session=True
            )
            
            active_processes[job_id] = process

            log_file_path = os.path.join(job_dir, f"{job_id}.log")
            jobs[job_id]["log_path"] = log_file_path
            progress_pattern = re.compile(r'(\d+)%\|')
            
            import sys
            with open(log_file_path, "w") as log_file:
                for line in process.stdout:
                    log_file.write(line)
                    log_file.flush()
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    match = progress_pattern.search(line)
                    if match:
                        jobs[job_id]["progress"] = int(match.group(1))
            
            process.wait()
            
            end_time = time.time()
            duration = end_time - start_time
            
            if process.returncode != 0:
                logger.error(f"LatentSync failed after {duration:.2f} seconds.")
                raise Exception(f"LatentSync error (return code {process.returncode})")
                
            logger.info(f"LatentSync completed successfully in {duration:.2f} seconds.")
            return output_video_path
            
        except Exception as e:
            logger.error(f"Error during LatentSync execution: {e}")
            raise Exception(str(e))
        finally:
            active_processes.pop(job_id, None)

def run_liveportrait(image_path: str, driving_video_path: str, job_id: str, job_dir: str) -> str:
    """Runs LivePortrait inside its own conda environment."""
    lp_dir = "/home/jack/ai/LivePortrait"
    start_time = time.time()
    
    # We will output directly to LatentSync's uploaded_media directory so we can easily find it
    output_name = f"{job_id}_lp.mp4"
    
    logger.info("Waiting for GPU lock for LivePortrait...")
    with gpu_lock:
        jobs[job_id]["status"] = "processing"
        _save_jobs()
        if jobs.get(job_id, {}).get("status") == "cancelled":
            logger.info(f"Job {job_id} was cancelled before LivePortrait started. Aborting.")
            return None
    
        # Use conda run to isolate the environment
        command = [
            "conda", "run", "--no-capture-output", "-n", "liveportrait",
            "python", "-u", "inference.py",
            "-s", image_path,
            "-d", driving_video_path,
            #"--no-flag-stitching",
            "--flag-normalize-lip", 
            "--no-flag-do-rot" # Attempt to disable side-by-side concat if possible, or we will just use the output
        ]
        
        lp_env = os.environ.copy()
        lp_env["CUDA_VISIBLE_DEVICES"] = "0"
        lp_env["PYTHONUNBUFFERED"] = "1"
        
        logger.info(f"Running LivePortrait command: {' '.join(command)}")
        process = None
        try:
            process = subprocess.Popen(
                command,
                cwd=lp_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=lp_env,
                start_new_session=True
            )
            active_processes[job_id] = process
            
            log_file_path = os.path.join(job_dir, f"{job_id}.log")
            jobs[job_id]["log_path"] = log_file_path
            progress_pattern = re.compile(r'(\d+)%\|')
            
            import sys
            with open(log_file_path, "w") as log_file:
                line_buf = ""
                while True:
                    char = process.stdout.read(1)
                    if not char:
                        break
                    log_file.write(char)
                    log_file.flush()
                    sys.stdout.write(char)
                    sys.stdout.flush()
                    line_buf += char
                    if char in ('\r', '\n'):
                        match = progress_pattern.search(line_buf)
                        if match:
                            jobs[job_id]["progress"] = int(match.group(1))
                        line_buf = "" 
                    
            process.wait()
            duration = time.time() - start_time
            
            if process.returncode != 0:
                logger.error(f"LivePortrait failed after {duration:.2f}s")
                raise Exception(f"LivePortrait error (return code {process.returncode})")
                
            logger.info(f"LivePortrait completed in {duration:.2f}s")
            
            # LivePortrait defaults to saving in `animations/` folder, usually named {source}--{driving}.mp4
            # Since predicting the exact filename is hard if the flags change, we should find the newest mp4 in animations/
            animations_dir = os.path.join(lp_dir, "animations")
            files = [os.path.join(animations_dir, f) for f in os.listdir(animations_dir) if f.endswith('.mp4')]
            newest_file = max(files, key=os.path.getctime)
            
            # Copy to our output dir
            final_lp_path = os.path.join(job_dir, output_name)
            shutil.copy(newest_file, final_lp_path)
            return final_lp_path
            
        except Exception as e:
            logger.error(f"LivePortrait execution failed: {e}")
            raise Exception(str(e))
        finally:
            active_processes.pop(job_id, None)
def enforce_json(request: Request):
    if not request.headers.get("content-type", "").startswith("application/json"):
        raise HTTPException(status_code=415, detail="Unsupported Media Type. This endpoint requires application/json.")

@app.post("/generate/video-sync", response_model=JobCreatedResponse, dependencies=[Depends(enforce_json)])
async def generate_video(
    background_tasks: BackgroundTasks,
    req: VideoSyncRequest,
    x_environment: str = Header("prod"),
    x_latentsync_version: Optional[str] = Header(None, alias="X-LatentSync-Version"),
):
    """
    **ASYNC** — Returns `job_id` immediately. Poll `GET /jobs/{job_id}` for status.

    Lip-syncs the provided audio onto the provided video.
    """
    video_s3_key = req.video_s3_key
    audio_s3_key = req.audio_s3_key
    user_id = req.user_id
    s3_bucket = req.s3_bucket
    output_key = req.output_key
    webhook_url = req.webhook_url
    mock_run = req.mock_run

    job_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    logger.info(f"Starting video lip-sync job {job_id} for user {user_id} in {x_environment}")
    
    x_env = get_env_name(x_environment)
    latentsync_version = get_requested_latentsync_version(x_latentsync_version)
    logger.info(f"Using LatentSync version {latentsync_version}")
    job_dir = os.path.join(SHARED_STORAGE_BASE, f"data-{x_env}", user_id, job_id)
    os.makedirs(job_dir, exist_ok=True)
    
    # Save inputs
    video_ext = os.path.splitext(video_s3_key)[1] or ".mp4"
    audio_ext = os.path.splitext(audio_s3_key)[1] or ".wav"
    
    video_path = os.path.join(job_dir, f"{job_id}_src{video_ext}")
    audio_path = os.path.join(job_dir, f"{job_id}_aud{audio_ext}")
    
    import s3_utils
    s3_utils.download_from_s3(video_s3_key, video_path, s3_bucket)
    s3_utils.download_from_s3(audio_s3_key, audio_path, s3_bucket)
        
    jobs[job_id] = {
        "status": "queued",
        "type": "video",
        "job_dir": job_dir,
        "user_id": user_id,
        "created_at": now,
        "latentsync_version": latentsync_version,
        "s3_bucket": s3_bucket,
    }
    _save_jobs()
    background_tasks.add_task(process_video_job, job_id, video_path, audio_path, job_dir, latentsync_version, s3_bucket, output_key, webhook_url, mock_run)
    
    return {"job_id": job_id, "status": "queued"}


@app.post("/generate/portrait-sync", response_model=JobCreatedResponse, dependencies=[Depends(enforce_json)])
async def generate_image(
    background_tasks: BackgroundTasks,
    req: PortraitSyncRequest,
    x_environment: str = Header("prod"),
    x_latentsync_version: Optional[str] = Header(None, alias="X-LatentSync-Version"),
):
    """
    **ASYNC** — Returns `job_id` immediately. Poll `GET /jobs/{job_id}` for status.

    Generates a talking-head video from a single image.
    """
    image_s3_key = req.image_s3_key
    driving_video_s3_key = req.driving_video_s3_key
    audio_s3_key = req.audio_s3_key
    user_id = req.user_id
    s3_bucket = req.s3_bucket
    output_key = req.output_key
    webhook_url = req.webhook_url
    mock_run = req.mock_run

    job_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    logger.info(f"Starting image lip-sync job {job_id} for user {user_id} in {x_environment}")
    
    x_env = get_env_name(x_environment)
    latentsync_version = get_requested_latentsync_version(x_latentsync_version)
    logger.info(f"Using LatentSync version {latentsync_version}")
    job_dir = os.path.join(SHARED_STORAGE_BASE, f"data-{x_env}", user_id, job_id)
    os.makedirs(job_dir, exist_ok=True)
    
    # Save inputs
    img_ext = os.path.splitext(image_s3_key)[1] or ".jpg"
    vid_ext = os.path.splitext(driving_video_s3_key)[1] or ".mp4"
    aud_ext = os.path.splitext(audio_s3_key)[1] or ".wav"
    
    img_path = os.path.join(job_dir, f"{job_id}_src{img_ext}")
    vid_path = os.path.join(job_dir, f"{job_id}_drv{vid_ext}")
    aud_path = os.path.join(job_dir, f"{job_id}_aud{aud_ext}")
    
    import s3_utils
    s3_utils.download_from_s3(image_s3_key, img_path, s3_bucket)
    s3_utils.download_from_s3(driving_video_s3_key, vid_path, s3_bucket)
    s3_utils.download_from_s3(audio_s3_key, aud_path, s3_bucket)
        
    jobs[job_id] = {
        "status": "queued",
        "type": "image",
        "job_dir": job_dir,
        "user_id": user_id,
        "created_at": now,
        "latentsync_version": latentsync_version,
        "s3_bucket": s3_bucket,
    }
    _save_jobs()
    background_tasks.add_task(process_image_job, job_id, img_path, vid_path, aud_path, job_dir, latentsync_version, s3_bucket, output_key, webhook_url, mock_run)
    
    return {"job_id": job_id, "status": "queued"}


@app.post("/generate/liveportrait-animation", response_model=JobCreatedResponse, dependencies=[Depends(enforce_json)])
async def generate_liveportrait(
    background_tasks: BackgroundTasks,
    req: LivePortraitAnimationRequest,
    x_environment: str = Header("prod"),
):
    """
    **ASYNC** — Returns `job_id` immediately. Poll `GET /jobs/{job_id}` for status.

    Animates a face image using a default driving video (no audio lip-sync).
    """
    image_s3_key = req.image_s3_key
    user_id = req.user_id
    s3_bucket = req.s3_bucket
    output_key = req.output_key
    webhook_url = req.webhook_url
    mock_run = req.mock_run

    job_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    logger.info(f"Starting LivePortrait job {job_id} for user {user_id} in {x_environment}")
    
    x_env = get_env_name(x_environment)
    job_dir = os.path.join(SHARED_STORAGE_BASE, f"data-{x_env}", user_id, job_id)
    os.makedirs(job_dir, exist_ok=True)
    
    # Save inputs
    img_ext = os.path.splitext(image_s3_key)[1] or ".jpg"
    img_path = os.path.join(job_dir, f"{job_id}_src{img_ext}")
    
    import s3_utils
    s3_utils.download_from_s3(image_s3_key, img_path, s3_bucket)
        
    vid_path = os.path.join(ASSETS_DIR, "driving_man2.mp4")  # driving_woman3.mp3 also ok
    if not os.path.exists(vid_path):
        raise HTTPException(status_code=500, detail="Default driving video not found")
    
    jobs[job_id] = {"status": "queued", "type": "liveportrait", "job_dir": job_dir, "user_id": user_id, "created_at": now, "s3_bucket": s3_bucket}
    _save_jobs()

    background_tasks.add_task(process_liveportrait_job, job_id, img_path, vid_path, job_dir, s3_bucket, output_key, webhook_url, mock_run)
    
    return {"job_id": job_id, "status": "queued"}


if __name__ == "__main__":
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=9882)
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)