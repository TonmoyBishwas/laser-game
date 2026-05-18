"""
Replay a recorded clip (or every clip in a folder) through YOLO with the
same settings the friend's laptop would use, draw detection boxes, and
print + save a performance summary.

Defaults match the uni profile: CPU device, imgsz=320, conf=0.20. Pass
flags to test other configs (e.g. --device cuda:0 --imgsz 640 to see how
your home PC would handle the same clip).

Usage
-----
  venv\\Scripts\\python.exe eval_video.py --source clips\\01_black.mp4
  venv\\Scripts\\python.exe eval_video.py --source clips\\         (folder)
  venv\\Scripts\\python.exe eval_video.py --source clip.mp4 --device cuda:0 --imgsz 640
  venv\\Scripts\\python.exe eval_video.py --source clip.mp4 --conf 0.10
  venv\\Scripts\\python.exe eval_video.py --source clip.mp4 --save-video out.mp4
  venv\\Scripts\\python.exe eval_video.py --source clip.mp4 --no-window      (headless)

Keys during playback:
  q   stop this clip (advance to next if folder)
  p   pause / resume
  ESC quit everything
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from game import config as cfg


WINDOW = "eval_video — q=next  p=pause  ESC=quit"


def list_clips(source: str) -> list[Path]:
    p = Path(source)
    if p.is_file():
        return [p]
    if p.is_dir():
        clips = sorted(
            p.glob("*.mp4")) + sorted(p.glob("*.avi")) + sorted(p.glob("*.mov"))
        return [c for c in clips if c.is_file()]
    return []


def evaluate_clip(clip: Path, model: YOLO, conf: float, imgsz: int,
                  device: str, show_window: bool,
                  save_video: Path | None) -> dict:
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        print(f"[eval] FAILED to open {clip}")
        return {"clip": clip.name, "error": "cannot open"}

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if save_video is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(save_video), fourcc, fps_src,
                                 (width, height))

    frames = 0
    dets = 0
    sum_conf = 0.0
    max_conf = 0.0
    per_frame_dets = []
    inference_times_ms = []

    paused = False
    aborted = False
    t_eval_start = time.time()

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            t0 = time.time()
            result = model.predict(frame, conf=conf, imgsz=imgsz,
                                   device=device, verbose=False)[0]
            infer_ms = (time.time() - t0) * 1000.0
            inference_times_ms.append(infer_ms)

            n = len(result.boxes)
            frames += 1
            per_frame_dets.append(n)
            if n > 0:
                dets += n
                conf_sum_frame = float(result.boxes.conf.sum().item())
                sum_conf += conf_sum_frame
                max_conf = max(max_conf, float(result.boxes.conf.max().item()))

            # draw boxes
            annotated = frame.copy()
            if n > 0:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    c = float(box.conf[0].item())
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated, f"{c:.2f}", (x1, max(15, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 0), 2)

            # overlay live stats
            running_mean_conf = (sum_conf / dets) if dets > 0 else 0.0
            running_hit_pct = 100.0 * sum(1 for d in per_frame_dets if d > 0) / frames
            avg_infer_ms = float(np.mean(inference_times_ms))
            fps_eff = 1000.0 / max(avg_infer_ms, 1e-3)

            txt = [
                f"clip: {clip.name}",
                f"frame {frames}/{total}  src_fps={fps_src:.1f}",
                f"device={device} imgsz={imgsz} conf={conf}",
                f"avg infer={avg_infer_ms:.1f}ms ({fps_eff:.1f} fps)",
                f"dets={dets}  hit_pct={running_hit_pct:.1f}%  mean_conf={running_mean_conf:.2f}",
            ]
            for i, line in enumerate(txt):
                y = 24 + i * 26
                cv2.rectangle(annotated, (8, y - 20), (8 + 11 * len(line), y + 6),
                              (0, 0, 0), -1)
                cv2.putText(annotated, line, (12, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (40, 255, 255), 2)

            if writer is not None:
                writer.write(annotated)

            if show_window:
                cv2.imshow(WINDOW, annotated)

        if show_window:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                paused = not paused
            if key == 27:  # ESC
                aborted = True
                break

    cap.release()
    if writer is not None:
        writer.release()

    eval_secs = time.time() - t_eval_start
    hit_frames = sum(1 for d in per_frame_dets if d > 0)
    summary = {
        "clip": clip.name,
        "frames_processed": frames,
        "frames_total": total,
        "src_fps": round(fps_src, 1),
        "device": device,
        "imgsz": imgsz,
        "conf": conf,
        "detections_total": dets,
        "frames_with_detection": hit_frames,
        "hit_pct": round(100.0 * hit_frames / max(frames, 1), 1),
        "mean_conf": round((sum_conf / dets) if dets > 0 else 0.0, 3),
        "max_conf": round(max_conf, 3),
        "avg_infer_ms": round(float(np.mean(inference_times_ms))
                              if inference_times_ms else 0.0, 1),
        "effective_fps": round(1000.0 /
                               max(float(np.mean(inference_times_ms))
                                   if inference_times_ms else 1.0, 1e-3), 1),
        "eval_wall_secs": round(eval_secs, 1),
        "aborted": aborted,
    }
    return summary


def print_summary_row(s: dict) -> None:
    if "error" in s:
        print(f"  {s['clip']:<28} ERROR: {s['error']}")
        return
    print(f"  {s['clip']:<28}"
          f"  hit={s['hit_pct']:5.1f}%"
          f"  conf={s['mean_conf']:.2f}"
          f"  fps={s['effective_fps']:5.1f}"
          f"  ({s['avg_infer_ms']:.0f}ms/frame)"
          f"  dets={s['detections_total']}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True,
                   help="path to a video file OR a folder of clips")
    p.add_argument("--conf", type=float, default=0.20,
                   help="YOLO confidence threshold (uni default 0.20)")
    p.add_argument("--imgsz", type=int, default=320,
                   help="YOLO input size (uni default 320, home default 640)")
    p.add_argument("--device", type=str, default="cpu",
                   help="cpu, cuda:0, etc. (uni default cpu)")
    p.add_argument("--save-video", type=str, default="",
                   help="write annotated output video to this path (only "
                        "useful with single --source file)")
    p.add_argument("--no-window", action="store_true",
                   help="run headlessly without showing the preview window")
    p.add_argument("--out", type=str, default="eval_results",
                   help="folder to write JSON / TXT summaries to")
    args = p.parse_args()

    clips = list_clips(args.source)
    if not clips:
        print(f"[eval] no clips found at {args.source}")
        return

    print(f"[eval] loading YOLO from {cfg.MODEL_PATH}")
    model = YOLO(cfg.MODEL_PATH)
    model.to(args.device)
    print(f"[eval] device={args.device}  imgsz={args.imgsz}  conf={args.conf}")
    print(f"[eval] {len(clips)} clip(s) to evaluate")

    if not args.no_window:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    summaries = []
    save_video_path = Path(args.save_video) if args.save_video else None
    for idx, clip in enumerate(clips):
        print(f"\n[eval] {idx + 1}/{len(clips)}  {clip.name}")
        if save_video_path is not None and len(clips) > 1:
            # only attach save_video to the first clip when batch
            target = save_video_path if idx == 0 else None
        else:
            target = save_video_path
        s = evaluate_clip(clip, model, args.conf, args.imgsz, args.device,
                          show_window=not args.no_window,
                          save_video=target)
        summaries.append(s)
        if s.get("aborted"):
            print("[eval] aborted by user (ESC).")
            break

    if not args.no_window:
        cv2.destroyAllWindows()

    # ── final summary ────────────────────────────────────────────────────────
    print()
    print("=" * 78)
    print(f"  Summary  ({len(summaries)} clip(s), device={args.device}, "
          f"imgsz={args.imgsz}, conf={args.conf})")
    print("=" * 78)
    for s in summaries:
        print_summary_row(s)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    payload = {
        "timestamp": stamp,
        "device": args.device,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "model_path": str(cfg.MODEL_PATH),
        "summaries": summaries,
    }
    (out_dir / f"eval_{stamp}.json").write_text(json.dumps(payload, indent=2))
    (out_dir / "latest.json").write_text(json.dumps(payload, indent=2))

    txt_lines = [
        f"eval_video  {stamp}",
        f"model: {cfg.MODEL_PATH}",
        f"device={args.device}  imgsz={args.imgsz}  conf={args.conf}",
        "",
        f"  {'clip':<28}  {'hit%':>6}  {'conf':>5}  {'fps':>5}  {'ms':>4}  dets",
        "-" * 78,
    ]
    for s in summaries:
        if "error" in s:
            txt_lines.append(f"  {s['clip']:<28}  ERROR: {s['error']}")
            continue
        txt_lines.append(
            f"  {s['clip']:<28}"
            f"  {s['hit_pct']:>6.1f}"
            f"  {s['mean_conf']:>5.2f}"
            f"  {s['effective_fps']:>5.1f}"
            f"  {s['avg_infer_ms']:>4.0f}"
            f"  {s['detections_total']}"
        )
    txt = "\n".join(txt_lines)
    (out_dir / f"eval_{stamp}.txt").write_text(txt, encoding="utf-8")
    (out_dir / "latest.txt").write_text(txt, encoding="utf-8")

    print()
    print(f"[eval] saved {out_dir / f'eval_{stamp}.json'}")
    print(f"[eval] saved {out_dir / f'eval_{stamp}.txt'}")


if __name__ == "__main__":
    main()
