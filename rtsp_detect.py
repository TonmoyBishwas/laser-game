import argparse
import cv2
from ultralytics import YOLO

WEIGHTS = r"runs\detect\laser_detector_run\weights\best.pt"


def main():
    parser = argparse.ArgumentParser(description="Run YOLO detection on an RTSP stream.")
    parser.add_argument("rtsp_url", help="RTSP URL of the camera/stream")
    parser.add_argument("--weights", default=WEIGHTS, help="Path to YOLO .pt weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--device", default=None, help="Device, e.g. 'cpu', '0', '0,1'")
    parser.add_argument("--show", action="store_true", default=True, help="Show window")
    parser.add_argument("--save", default=None, help="Optional path to save annotated MP4")
    args = parser.parse_args()

    model = YOLO(args.weights)

    cap = cv2.VideoCapture(args.rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open RTSP stream: {args.rtsp_url}")

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))

    win = "YOLO RTSP Detection (q to quit)"
    if args.show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Stream read failed, attempting to reconnect...")
                cap.release()
                cap = cv2.VideoCapture(args.rtsp_url, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    break
                continue

            results = model.predict(
                frame,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )
            annotated = results[0].plot()

            if writer is not None:
                writer.write(annotated)

            if args.show:
                cv2.imshow(win, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
