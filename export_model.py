r"""
Export the trained YOLO model to a faster CPU inference format.

PyTorch on CPU is the slowest backend. On an Intel CPU (the uni i7),
ONNX Runtime is ~1.5-2x faster and OpenVINO is ~2-3x faster than PyTorch
CPU for the same model + imgsz. Exporting does NOT change accuracy.

  venv\Scripts\python.exe export_model.py --format onnx --imgsz 320
  venv\Scripts\python.exe export_model.py --format openvino --imgsz 320

- ONNX:     produces best.onnx next to best.pt. Runs via onnxruntime,
            which is already installed on the uni laptop. No extra install
            needed to RUN it.
- OpenVINO: produces best_openvino_model/ next to best.pt. Needs
            `pip install openvino` to export AND to run. Biggest speedup
            on Intel CPUs.

Then point the game / sandbox at the exported model:
  venv\Scripts\python.exe flashlight_sandbox.py --model runs\detect\laser_detector_run\weights\best.onnx --imgsz 320
"""

import argparse

from ultralytics import YOLO

from game import config as cfg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--format", choices=["onnx", "openvino"], default="onnx")
    p.add_argument("--imgsz", type=int, default=320,
                   help="input size to bake into the export (use what you run)")
    p.add_argument("--model", default=cfg.MODEL_PATH,
                   help="path to the .pt model to export")
    args = p.parse_args()

    print(f"[export] loading {args.model}")
    model = YOLO(args.model)
    print(f"[export] exporting to {args.format} at imgsz={args.imgsz} ...")
    out = model.export(format=args.format, imgsz=args.imgsz,
                       half=False, dynamic=False)
    print(f"[export] DONE -> {out}")
    print(f"[export] use it with:  --model \"{out}\" --imgsz {args.imgsz}")


if __name__ == "__main__":
    main()
