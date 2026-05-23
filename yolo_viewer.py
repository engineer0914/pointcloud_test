# 리얼센스 상에서 보이는 욜로 바운딩 박스 검증용 코드


import os
import time
import argparse

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO


def find_best_pt(base_dir="./yolo_models"):
    """
    yolo_models 하위에서 best.pt를 자동 탐색.
    가장 최근 수정된 best.pt를 사용.
    """
    candidates = []

    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f == "best.pt":
                path = os.path.join(root, f)
                candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"[ERROR] {base_dir} 하위에서 best.pt를 찾지 못했습니다."
        )

    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def init_realsense(width=1280, height=720, fps=30):
    """
    RealSense D435i / D435f 컬러 + 깊이 스트림 초기화.
    """
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    profile = pipeline.start(config)

    # depth를 color 기준으로 align
    align = rs.align(rs.stream.color)

    # depth scale 확인
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    print(f"[INFO] RealSense started")
    print(f"[INFO] Resolution: {width}x{height} @ {fps} FPS")
    print(f"[INFO] Depth scale: {depth_scale}")

    return pipeline, align, depth_scale


def draw_detections(image, results, conf_thres=0.5):
    """
    YOLO 검출 결과를 이미지에 표시.
    """
    annotated = image.copy()

    if results is None or len(results) == 0:
        return annotated

    result = results[0]
    names = result.names

    if result.boxes is None:
        return annotated

    for box in result.boxes:
        conf = float(box.conf[0])
        if conf < conf_thres:
            continue

        cls_id = int(box.cls[0])
        label = names.get(cls_id, str(cls_id))

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        text = f"{label} {conf:.2f}"
        cv2.putText(
            annotated,
            text,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # 중심점 표시
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)

    return annotated


def get_center_depth(depth_frame, cx, cy, window=5):
    """
    bbox 중심 주변 depth median 계산.
    단일 픽셀 depth는 튀는 경우가 있어서 작은 영역 median 사용.
    """
    w = depth_frame.get_width()
    h = depth_frame.get_height()

    x1 = max(cx - window, 0)
    x2 = min(cx + window + 1, w)
    y1 = max(cy - window, 0)
    y2 = min(cy + window + 1, h)

    depths = []

    for y in range(y1, y2):
        for x in range(x1, x2):
            d = depth_frame.get_distance(x, y)
            if d > 0:
                depths.append(d)

    if len(depths) == 0:
        return None

    return float(np.median(depths))


def draw_detections_with_depth(image, depth_frame, results, conf_thres=0.5):
    """
    YOLO 검출 결과 + 중심 depth 표시.
    """
    annotated = image.copy()

    if results is None or len(results) == 0:
        return annotated

    result = results[0]
    names = result.names

    if result.boxes is None:
        return annotated

    for box in result.boxes:
        conf = float(box.conf[0])
        if conf < conf_thres:
            continue

        cls_id = int(box.cls[0])
        label = names.get(cls_id, str(cls_id))

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        depth_m = get_center_depth(depth_frame, cx, cy, window=5)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)

        if depth_m is not None:
            text = f"{label} {conf:.2f} z={depth_m:.3f}m"
        else:
            text = f"{label} {conf:.2f} z=None"

        cv2.putText(
            annotated,
            text,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return annotated


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="YOLO best.pt path. None이면 yolo_models 하위에서 자동 탐색.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="confidence threshold",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="YOLO device. GPU면 0, CPU면 cpu",
    )

    args = parser.parse_args()

    if args.model is None:
        model_path = find_best_pt("./yolo_models")
    else:
        model_path = args.model

    print(f"[INFO] Using model: {model_path}")

    model = YOLO(model_path)

    pipeline, align, depth_scale = init_realsense(
        width=args.width,
        height=args.height,
        fps=args.fps,
    )

    prev_time = time.time()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                print("[WARN] Failed to get frames")
                continue

            color_image = np.asanyarray(color_frame.get_data())

            results = model.predict(
                source=color_image,
                conf=args.conf,
                device=args.device,
                verbose=False,
            )

            annotated = draw_detections_with_depth(
                color_image,
                depth_frame,
                results,
                conf_thres=args.conf,
            )

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("RealSense YOLO Detection", annotated)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[INFO] RealSense stopped")


if __name__ == "__main__":
    main()
