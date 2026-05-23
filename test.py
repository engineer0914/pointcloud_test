import time

import cv2
import numpy as np
import pyrealsense2 as rs


def capture_rgbd(width, height, fps=30, warmup=20):
    """
    지정한 해상도로 RealSense RGB + Depth 프레임을 1회 캡처한다.
    Depth는 Color 프레임 기준으로 align한다.
    """

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    print(f"[INFO] Starting RealSense: {width}x{height} @ {fps} FPS")

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    try:
        # 노출 / 깊이 안정화를 위해 초기 프레임 버림
        for _ in range(warmup):
            frames = pipeline.wait_for_frames()
            _ = align.process(frames)

        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            raise RuntimeError("[ERROR] Failed to get color/depth frame")

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

    finally:
        pipeline.stop()
        print(f"[INFO] Stopped RealSense: {width}x{height}")

    return color_image, depth_image, depth_scale


def make_depth_colormap(depth_image, depth_scale, max_depth_m=1.5):
    """
    z16 depth 이미지를 시각화용 컬러맵으로 변환한다.
    depth=0 영역은 검정색으로 표시.
    """

    depth_m = depth_image.astype(np.float32) * depth_scale

    depth_vis = np.clip(depth_m, 0, max_depth_m)
    depth_vis = (depth_vis / max_depth_m * 255.0).astype(np.uint8)

    depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    depth_colormap[depth_image == 0] = (0, 0, 0)

    return depth_colormap


def put_label(image, text):
    """
    이미지 상단에 검은 배경 라벨 추가.
    """

    out = image.copy()

    cv2.rectangle(out, (0, 0), (out.shape[1], 40), (0, 0, 0), -1)

    cv2.putText(
        out,
        text,
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return out


def make_pair_canvas_original_size(
    color_image,
    depth_image,
    depth_scale,
    title,
    max_depth_m=1.5,
    layout="horizontal",
):
    """
    원본 해상도를 유지한 상태로 Color와 Depth를 묶어서 출력할 canvas 생성.

    layout="horizontal":
        Color | Depth

    layout="vertical":
        Color
        Depth
    """

    depth_color = make_depth_colormap(
        depth_image=depth_image,
        depth_scale=depth_scale,
        max_depth_m=max_depth_m,
    )

    color_h, color_w = color_image.shape[:2]
    depth_h, depth_w = depth_color.shape[:2]

    # align을 했으면 보통 color/depth 해상도가 같지만,
    # 혹시 다르면 depth를 color 크기에 맞춤.
    if (color_w, color_h) != (depth_w, depth_h):
        depth_color = cv2.resize(
            depth_color,
            (color_w, color_h),
            interpolation=cv2.INTER_NEAREST,
        )

    color_vis = put_label(color_image, f"{title} Color")
    depth_vis = put_label(depth_color, f"{title} Depth")

    if layout == "horizontal":
        canvas = np.hstack([color_vis, depth_vis])
    elif layout == "vertical":
        canvas = np.vstack([color_vis, depth_vis])
    else:
        raise ValueError("layout must be 'horizontal' or 'vertical'")

    return canvas


def print_depth_info(name, depth_image, depth_scale):
    """
    depth 유효 픽셀 비율과 중심부 median depth 출력.
    """

    depth_m = depth_image.astype(np.float32) * depth_scale

    valid = depth_m > 0
    valid_ratio = np.sum(valid) / depth_m.size * 100.0

    h, w = depth_m.shape
    cx = w // 2
    cy = h // 2
    win = 20

    roi = depth_m[
        max(0, cy - win):min(h, cy + win + 1),
        max(0, cx - win):min(w, cx + win + 1),
    ]

    roi_valid = roi[roi > 0]

    if len(roi_valid) > 0:
        center_median = float(np.median(roi_valid))
    else:
        center_median = None

    print(f"\n[{name}]")
    print(f"resolution     : {w}x{h}")
    print(f"valid ratio    : {valid_ratio:.2f}%")

    if center_median is not None:
        print(f"center median  : {center_median:.4f} m")
    else:
        print("center median  : None")


def save_images(
    color_640,
    depth_640,
    scale_640,
    color_1280,
    depth_1280,
    scale_1280,
    max_depth_m,
):
    cv2.imwrite("color_640x480.png", color_640)
    cv2.imwrite("depth_640x480_raw.png", depth_640)
    cv2.imwrite(
        "depth_640x480_colormap.png",
        make_depth_colormap(depth_640, scale_640, max_depth_m),
    )

    cv2.imwrite("color_1280x720.png", color_1280)
    cv2.imwrite("depth_1280x720_raw.png", depth_1280)
    cv2.imwrite(
        "depth_1280x720_colormap.png",
        make_depth_colormap(depth_1280, scale_1280, max_depth_m),
    )

    print("[INFO] Images saved.")


def main():
    max_depth_m = 1.5

    # 1280x720을 좌우로 붙이면 2560x720이라 너무 넓을 수 있음.
    # 그래도 원본 비율 그대로 보려면 horizontal 유지.
    layout_640 = "horizontal"
    layout_1280 = "horizontal"

    while True:
        print("\n==============================")
        print("[INFO] Capturing 640x480 RGB-D")
        print("==============================")

        color_640, depth_640, scale_640 = capture_rgbd(
            width=640,
            height=480,
            fps=30,
            warmup=20,
        )

        time.sleep(0.5)

        print("\n==============================")
        print("[INFO] Capturing 1280x720 RGB-D")
        print("==============================")

        color_1280, depth_1280, scale_1280 = capture_rgbd(
            width=1280,
            height=720,
            fps=15,
            warmup=20,
        )

        print_depth_info("640x480 Depth", depth_640, scale_640)
        print_depth_info("1280x720 Depth", depth_1280, scale_1280)

        canvas_640 = make_pair_canvas_original_size(
            color_image=color_640,
            depth_image=depth_640,
            depth_scale=scale_640,
            title="640x480",
            max_depth_m=max_depth_m,
            layout=layout_640,
        )

        canvas_1280 = make_pair_canvas_original_size(
            color_image=color_1280,
            depth_image=depth_1280,
            depth_scale=scale_1280,
            title="1280x720",
            max_depth_m=max_depth_m,
            layout=layout_1280,
        )

        cv2.imshow("640x480 Color | Depth", canvas_640)
        cv2.moveWindow("640x480 Color | Depth", 50, 50)

        cv2.imshow("1280x720 Color | Depth", canvas_1280)
        cv2.moveWindow("1280x720 Color | Depth", 50, 600)

        print("\n[KEY]")
        print("SPACE : recapture")
        print("s     : save images")
        print("q/ESC : quit")

        key = cv2.waitKey(0) & 0xFF

        if key == ord("q") or key == 27:
            break

        elif key == ord("s"):
            save_images(
                color_640=color_640,
                depth_640=depth_640,
                scale_640=scale_640,
                color_1280=color_1280,
                depth_1280=depth_1280,
                scale_1280=scale_1280,
                max_depth_m=max_depth_m,
            )

        elif key == ord(" "):
            cv2.destroyWindow("640x480 Color | Depth")
            cv2.destroyWindow("1280x720 Color | Depth")
            continue

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
