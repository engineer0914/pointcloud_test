import cv2
import pyrealsense2 as rs
import numpy as np
import os
from datetime import datetime

# ============================================================
# RealSense RGB 캘리브레이션 이미지 캡처 코드
# Space : 이미지 저장
# q     : 종료
# ============================================================

def main():
    # =========================
    # 사용자 설정
    # =========================
    SERIAL = "327122072783"

    WIDTH = 640
    HEIGHT = 480
    FPS = 30

    SAVE_DIR = r"C:\Users\ULTIMATE NIGHTMARE\Desktop\test\pointcloud_test\opencv_calibration\d435i_images_640x480"

    os.makedirs(SAVE_DIR, exist_ok=True)

    # =========================
    # RealSense 시작
    # =========================
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_device(SERIAL)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

    profile = pipeline.start(config)

    # =========================
    # Intrinsics 출력
    # =========================
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()

    print("\n[RealSense COLOR Intrinsics]")
    print("width :", intr.width)
    print("height:", intr.height)
    print("fx    :", intr.fx)
    print("fy    :", intr.fy)
    print("cx    :", intr.ppx)
    print("cy    :", intr.ppy)
    print("model :", intr.model)
    print("coeffs:", intr.coeffs)

    print("\n[INFO]")
    print("Space : save image")
    print("q     : quit")
    print("SAVE_DIR:", SAVE_DIR)

    i = 0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            color_img = np.asanyarray(color_frame.get_data())

            # 화면 표시용 복사
            vis = color_img.copy()

            cv2.putText(
                vis,
                f"Saved: {i} | Space: save | q: quit",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            cv2.imshow("D435I Color Calibration Capture", vis)

            k = cv2.waitKey(1) & 0xFF

            # Space
            if k == 32:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                image_name = f"d435i_{WIDTH}x{HEIGHT}_{i:04d}_{timestamp}.jpg"
                image_path = os.path.join(SAVE_DIR, image_name)

                cv2.imwrite(image_path, color_img)
                print(f"[SAVE] {image_path}")

                i += 1

            # q
            elif k == ord("q"):
                print("[INFO] Quit.")
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[INFO] RealSense stopped safely.")


if __name__ == "__main__":
    main()