import cv2
import numpy as np
import glob
import os
import yaml

# ============================================================
# 사용자 설정
# ============================================================
IMAGE_DIR = r"C:\Users\ULTIMATE NIGHTMARE\Desktop\test\pointcloud_test\opencv_calibration\d435i_images_640x480"
SAVE_YAML = r"C:\Users\ULTIMATE NIGHTMARE\Desktop\test\pointcloud_test\opencv_calibration\d435i_rgb_calib_640x480.yaml"

# 중요:
# 실제 체커보드가 8칸 x 6칸이면 내부 코너는 7 x 5
CHECKERBOARD = (7, 5)     # (가로 내부 코너 수, 세로 내부 코너 수)
SQUARE_SIZE = 30.0        # mm

# ============================================================
# 3D object points 준비
# 예: (0,0,0), (30,0,0), (60,0,0) ...
# ============================================================
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[
    0:CHECKERBOARD[0],
    0:CHECKERBOARD[1]
].T.reshape(-1, 2)

objp *= SQUARE_SIZE

objpoints = []  # 3D 점
imgpoints = []  # 2D 이미지 코너 점

# ============================================================
# 이미지 읽기
# ============================================================
image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))

if len(image_paths) == 0:
    raise FileNotFoundError(f"이미지가 없습니다: {IMAGE_DIR}")

print(f"[INFO] found images: {len(image_paths)}")

image_size = None
success_count = 0

# 코너 정밀화 조건
criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

for path in image_paths:
    img = cv2.imread(path)

    if img is None:
        print("[WARN] cannot read:", path)
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if image_size is None:
        image_size = gray.shape[::-1]  # (width, height)

    # 체커보드 코너 탐색
    ret, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_NORMALIZE_IMAGE
            + cv2.CALIB_CB_FAST_CHECK
    )

    if ret:
        corners_subpix = cv2.cornerSubPix(
            gray,
            corners,
            winSize=(11, 11),
            zeroZone=(-1, -1),
            criteria=criteria
        )

        objpoints.append(objp)
        imgpoints.append(corners_subpix)
        success_count += 1

        vis = img.copy()
        cv2.drawChessboardCorners(vis, CHECKERBOARD, corners_subpix, ret)
        cv2.imshow("checkerboard detection", vis)
        cv2.waitKey(100)

        print("[OK]", os.path.basename(path))
    else:
        print("[FAIL]", os.path.basename(path))

cv2.destroyAllWindows()

print("\n[INFO] successful images:", success_count)

if success_count < 10:
    raise RuntimeError("성공 이미지가 너무 적습니다. 최소 10장 이상, 가능하면 20~40장 추천.")

# ============================================================
# 카메라 캘리브레이션
# ============================================================
rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    image_size,
    None,
    None
)

print("\n==============================")
print("[CALIBRATION RESULT]")
print("==============================")
print("RMS reprojection error:", rms)
print("\nCamera Matrix K:")
print(camera_matrix)
print("\nDistortion Coeffs:")
print(dist_coeffs.ravel())

# ============================================================
# 결과 저장
# ============================================================
calib_data = {
    "image_width": int(image_size[0]),
    "image_height": int(image_size[1]),
    "checkerboard_inner_corners": {
        "cols": int(CHECKERBOARD[0]),
        "rows": int(CHECKERBOARD[1]),
    },
    "square_size_mm": float(SQUARE_SIZE),
    "rms_reprojection_error": float(rms),
    "camera_matrix": camera_matrix.tolist(),
    "dist_coeffs": dist_coeffs.ravel().tolist(),
}

os.makedirs(os.path.dirname(SAVE_YAML), exist_ok=True)

with open(SAVE_YAML, "w", encoding="utf-8") as f:
    yaml.dump(calib_data, f, sort_keys=False, allow_unicode=True)

print("\n[SAVED]", SAVE_YAML)