import cv2
import numpy as np
import glob
import os

# ============================================================
# 사용자 설정
# ============================================================
IMAGE_DIR = r"C:\Users\ULTIMATE NIGHTMARE\Desktop\test\pointcloud_test\opencv_calibration\d435i_images_640x480"


# # 잘못 캘리브레이션
# K = np.array([
#     [612.47085705,   0.0,        333.99167683],
#     [0.0,          612.84470568, 243.71783218],
#     [0.0,            0.0,          1.0]
# ], dtype=np.float64)

# D = np.array([
#     7.28805931e-02,
#     3.33991276e-01,
#    -1.49905110e-03,
#     2.44385652e-03,
#    -1.51889258e+00
# ], dtype=np.float64)

# 잘된 캘리브레이션
K = np.array([
    [612.18511698,   0.0,        334.06049318],
    [0.0,          612.57575986, 244.38795107],
    [0.0,            0.0,          1.0]
], dtype=np.float64)

D = np.array([
    0.15183487,
   -0.40626659,
   -0.00066745,
    0.00268326,
    0.33651456
], dtype=np.float64)

ALPHA = 0

# ============================================================
# 이미지 목록
# ============================================================
image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))

if len(image_paths) == 0:
    raise FileNotFoundError(f"이미지가 없습니다: {IMAGE_DIR}")

# 첫 이미지 기준 map 생성
tmp = cv2.imread(image_paths[0])
h, w = tmp.shape[:2]

new_K, roi = cv2.getOptimalNewCameraMatrix(
    K, D,
    (w, h),
    alpha=ALPHA,
    newImgSize=(w, h)
)

map1, map2 = cv2.initUndistortRectifyMap(
    K, D,
    None,
    new_K,
    (w, h),
    cv2.CV_32FC1
)

print("[INFO] image count:", len(image_paths))
print("[INFO] key: a=prev, d=next, q=quit")
print("[New K]")
print(new_K)
print("[ROI]", roi)

idx = 0

while True:
    img = cv2.imread(image_paths[idx])

    if img is None:
        idx = (idx + 1) % len(image_paths)
        continue

    undistorted = cv2.remap(
        img,
        map1, map2,
        interpolation=cv2.INTER_LINEAR
    )

    original_vis = img.copy()
    undist_vis = undistorted.copy()

    name = os.path.basename(image_paths[idx])

    cv2.putText(
        original_vis,
        f"Original {idx+1}/{len(image_paths)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        undist_vis,
        f"Undistorted alpha={ALPHA}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        undist_vis,
        name,
        (20, 470),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1,
        cv2.LINE_AA
    )

    compare = np.hstack([original_vis, undist_vis])

    cv2.imshow("Original vs Undistorted", compare)

    key = cv2.waitKey(0) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("d"):
        idx = (idx + 1) % len(image_paths)
    elif key == ord("a"):
        idx = (idx - 1) % len(image_paths)

cv2.destroyAllWindows()