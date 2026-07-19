"""
End-to-end smoke test for LatentSync at 256 resolution.

Runs the full inference pipeline (the same command the API uses) with
the 256 config/checkpoint and validates the output.
"""

import os
import subprocess
import sys
import time
import tempfile
import shutil

# ---------- configuration (mirrors api.py after the 256 switch) ----------
UNET_CONFIG = "configs/unet/stage2.yaml"
CKPT_PATH   = "checkpoints/latentsync_unet_256.pt"
INFERENCE_STEPS = 20
GUIDANCE_SCALE  = 1.5

# Use the demo assets shipped with the repo
TEST_VIDEO = "assets/demo1_video.mp4"
TEST_AUDIO = "assets/demo1_audio.wav"

MIN_OUTPUT_SIZE_KB = 50  # a valid output should be at least this large


def run_e2e():
    # Pre-flight checks
    for path in (UNET_CONFIG, CKPT_PATH, TEST_VIDEO, TEST_AUDIO):
        if not os.path.exists(path):
            print(f"FAIL – required file not found: {path}")
            return False

    output_dir = tempfile.mkdtemp(prefix="e2e_256_")
    output_path = os.path.join(output_dir, "test_out.mp4")

    command = [
        sys.executable, "-u", "-m", "scripts.inference",
        "--unet_config_path", UNET_CONFIG,
        "--inference_ckpt_path", CKPT_PATH,
        "--inference_steps", str(INFERENCE_STEPS),
        "--guidance_scale", str(GUIDANCE_SCALE),
        "--enable_deepcache",
        "--video_path", TEST_VIDEO,
        "--audio_path", TEST_AUDIO,
        "--video_out_path", output_path,
    ]

    print(f"Running: {' '.join(command)}")
    t0 = time.time()

    # Stream output in real-time so it's visible in the terminal
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    output_lines = []
    for line in process.stdout:
        print(line, end="")
        output_lines.append(line)

    process.wait()
    elapsed = time.time() - t0
    print(f"\n--- Inference finished in {elapsed:.1f}s (return code {process.returncode}) ---")

    if process.returncode != 0:
        print(f"\nFAIL – inference process exited with code {process.returncode}")
        return False

    # Validate output
    if not os.path.exists(output_path):
        print(f"\nFAIL – output file not created: {output_path}")
        return False

    size_kb = os.path.getsize(output_path) / 1024
    print(f"Output file: {output_path}  ({size_kb:.1f} KB)")

    if size_kb < MIN_OUTPUT_SIZE_KB:
        print(f"\nFAIL – output too small ({size_kb:.1f} KB < {MIN_OUTPUT_SIZE_KB} KB)")
        return False

    # Optional: try to read with cv2 to confirm it's a valid video
    try:
        import cv2
        cap = cv2.VideoCapture(output_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        print(f"Video metadata: {width}x{height}, {frame_count} frames, {fps:.1f} fps")
        if frame_count < 1:
            print("\nFAIL – video has 0 frames")
            return False
    except ImportError:
        print("  (cv2 not available, skipping frame-level validation)")

    print(f"\nOutput kept at: {output_path}")
    print("\nPASS – End-to-end 256 inference completed successfully.")
    return True


if __name__ == "__main__":
    success = run_e2e()
    sys.exit(0 if success else 1)