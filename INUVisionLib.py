import os
import yaml
import numpy as np
import cv2
import pyrealsense2 as rs
import matplotlib.pyplot as plt
import pprint
import open3d as o3d
from sklearn.cluster import DBSCAN
from ultralytics import YOLO
import torch


# ==========================================
# 🎯 카메라 상황별 맞춤 설정 (Profiles)
# ==========================================
CAMERA_PROFILES = {
    # 1. 바닥(Floor) 모드: RANSAC 평면 검출용 (넓고 강하게)
    "floor": {
        "preset_id": 4,              # High Density (바닥 구멍 채우기)
        "smooth_alpha": 0.5,
        "smooth_delta": 20,          # 평면을 더 평평하게 다듬기 위해 낮춤
        "min_dist": 0.30,            # 카메라 바로 앞 먼지/노이즈 무시
        "max_dist": 3.00,            # 바닥까지만 보고 불필요한 원거리 컷
        "target_laser_power": 360,   # 바닥 반사율 확보를 위해 최대 파워
        "target_shift": 0,           # 정상 시력 구간
        "roi_percent": 80,           # 화면 전체를 기준으로 노출 계산
        "auto_awb_value": 1,
        "depth_Units": 0.0001
    },
    
    # 2. 근접 30cm 모드: 듀플로 픽킹용 (정밀하고 어둡게)
    "macro_30": {
    "preset_id": 4,              # High Accuracy 계열이면 3 유지, 안 맞으면 4도 테스트
    "smooth_alpha": 0.6,         # 엣지 보존 위해 너무 낮게 하지 않음
    "smooth_delta": 20,          # 50~100은 직각/얇은 부분을 뭉갤 수 있음
    "min_dist": 0.08,
    "max_dist": 0.32,            # 20cm 근처만 보기
    "target_laser_power": 150,    # 너무 가까우면 150도 과할 수 있음
    "target_shift": 20,          # 20cm 근거리용 시작값
    "roi_percent": 15,           # 중앙 객체 기준 AE
    "auto_awb_value": 1,
    "depth_Units": 0.00001       # 근거리 전용. 불안정하면 0.0001로 복귀
    },
    
    # 3. 원거리 50cm 모드: 접근 및 탐색용 (밸런스형)
    "mid_50": {
        "preset_id": 4,              # High Density
        "smooth_alpha": 0.5,
        "smooth_delta": 50,
        "min_dist": 0.20,
        "max_dist": 0.80,            # 작업대(테이블) 영역 정도까지만 컷
        "target_laser_power": 250,   # 너무 세지도 약하지도 않은 중간 파워
        "target_shift": 0,           # 50cm는 기본 시력 구간에 포함됨 (Shift 불필요)
        "roi_percent": 40,           # 작업 영역인 중앙 40% 기준 노출
        "auto_awb_value": 1,
        "depth_Units": 0.0001        
    }
}

# 카메라 설정 함수들

def get_realsense_ids():
    """
    연결된 모든 리얼센스 카메라의 이름과 시리얼 번호(ID)를 딕셔너리 형태로 반환합니다.
    """
    connected_devices = {}
    
    # 리얼센스 컨텍스트 생성 (연결된 기기들을 관리)
    ctx = rs.context()
    
    # 연결된 기기가 없는 경우 예외 처리
    if len(ctx.devices) == 0:
        print("연결된 리얼센스 카메라를 찾을 수 없습니다.")
        return connected_devices

    # 연결된 모든 기기를 순회하며 정보 추출
    for dev in ctx.devices:
        name = dev.get_info(rs.camera_info.name)
        serial_number = dev.get_info(rs.camera_info.serial_number)
        
        # 시리얼 번호를 키(Key)로, 모델명을 값(Value)으로 저장
        connected_devices[serial_number] = name
        
    return connected_devices

def configure_realsense(
        serial_number=None,
        preset_id=4, 
        smooth_alpha=0.5, 
        smooth_delta=50, 
        min_dist=0.15, 
        max_dist=2.0, 
        target_laser_power = 150, 
        target_shift = 0, 
        roi_percent=80, 
        auto_awb_value=1,
        depth_Units=0.0001
        ):
    
    pipeline = rs.pipeline()
    config = rs.config()

    # 1. 스트림 해상도 설정
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

    # 파이프라인 시작
    profile = pipeline.start(config)
    dev = profile.get_device()
    advnc_mode = rs.rs400_advanced_mode(dev)
    
    # 2. 센서 객체 가져오기
    depth_sensor = dev.first_depth_sensor()
    
    # 💡 추가해야 할 부분!
    color_sensor = dev.first_color_sensor()
    
    # 3. 비주얼 프리셋 설정
    if depth_sensor.supports(rs.option.visual_preset):
        depth_sensor.set_option(rs.option.visual_preset, preset_id)
        print(f"✅ Preset 설정 완료: {depth_sensor.get_option_value_description(rs.option.visual_preset, preset_id)}")

    # 4. 뎁스 유닛(Depth Unit) 설정
    if depth_sensor.supports(rs.option.depth_units):
        try:
            depth_sensor.set_option(rs.option.depth_units, depth_Units)
            print(f"✅ Depth Unit 설정 완료: {depth_sensor.get_option(rs.option.depth_units)}")
        except Exception as e:
            print(f"⚠️ Depth Unit 설정 실패: {e}")

    # 5. 자동 노출(Auto Exposure) 활성화 및 ROI 설정
    if depth_sensor.supports(rs.option.enable_auto_exposure):
        depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
        print("✅ 뎁스 센서 자동 노출(AE) 스위치 ON")

    roi_sensor = rs.roi_sensor(depth_sensor)
    if roi_sensor:
        # 비율을 기반으로 여백(margin) 계산 (예: 80% -> 남은 20%를 반으로 나눠 0.1의 여백 생성)
        # 20: 초근접 중앙 객체
        # 40: 일반적인 중앙 영역
        # 80: 넓은 중앙 영역
        margin = (100 - roi_percent) / 2.0 / 100.0        
        roi = rs.region_of_interest()
        # 해상도 848x480 기준 중앙 영역 동적 계산
        roi.min_x = int(848 * margin)
        roi.max_x = int(848 * (1.0 - margin))
        roi.min_y = int(480 * margin)
        roi.max_y = int(480 * (1.0 - margin))
        
        # 기기에 ROI 세팅 적용
        roi_sensor.set_region_of_interest(roi)
        print(f"✅ Depth ROI 영역 설정 완료 (중앙 {roi_percent}% / X: {roi.min_x}~{roi.max_x}, Y: {roi.min_y}~{roi.max_y})")

    # 레이저 파워 설정
    # 추천값 -> 60cm 이상 바닥: 360 (최대치) / 30cm 코앞: 약 150
    if depth_sensor.supports(rs.option.laser_power):
        depth_sensor.set_option(rs.option.laser_power, target_laser_power)
        print(f"✅ 레이저 파워 설정 완료: {target_laser_power}")

    # 🎯 디스패리티 시프트 설정 (고급 모드 사용)
    # 추천값 -> 60cm 이상 바닥: 0 (기본값) / 30cm 코앞: 50 ~ 100 사이 조절
    if advnc_mode.is_enabled():
        depth_table = advnc_mode.get_depth_table()
        depth_table.disparityShift = target_shift
        advnc_mode.set_depth_table(depth_table)
        print(f"✅ Disparity Shift 설정 완료: {target_shift}")

    # 6. Temporal Filter 설정
    temp_filter = rs.temporal_filter()
    # alpha 값을 낮출수록(예: 0.1~0.2) 이전 프레임의 데이터를 더 오래, 무겁게 기억합니다.
    temp_filter.set_option(rs.option.filter_smooth_alpha, smooth_alpha)
    # delta 값을 높일수록 노이즈를 깎지 않고 찰흙처럼 뭉개버립니다.
    temp_filter.set_option(rs.option.filter_smooth_delta, smooth_delta)
    # # hole filling 옵션: 누적된 데이터로 구멍을 강제로 메웁니다 (0~8)
    # temp_filter.set_option(rs.option.holes_fill, 3)
    print(f"✅ Temporal Filter 설정 완료")

    # ==========================================
    # 💡 7. Threshold Filter (거리 제한) 설정
    # ==========================================
    thres_filter = rs.threshold_filter()
    thres_filter.set_option(rs.option.min_distance, min_dist)
    thres_filter.set_option(rs.option.max_distance, max_dist)
    print(f"✅ Threshold Filter 설정 완료 (최소: {min_dist}m, 최대: {max_dist}m)")


    # 🎯 컬러 센서 자동 화이트 밸런스(AWB) 설정
    if color_sensor.supports(rs.option.enable_auto_white_balance):
        color_sensor.set_option(rs.option.enable_auto_white_balance, auto_awb_value)
        print(f"✅ 컬러 센서 자동 화이트 밸런스(AWB) {'ON' if auto_awb_value == 1 else 'OFF'}")

    # # (선택) 수동으로 색온도를 고정하고 싶을 때의 예시
    # if color_sensor.supports(rs.option.enable_auto_white_balance):
    #     color_sensor.set_option(rs.option.enable_auto_white_balance, 0) # 자동 끄기        
    #     color_sensor.set_option(rs.option.white_balance, 4600)
    #     print("✅ 컬러 센서 화이트 밸런스 수동 고정 (4600K)")

    # 8. 컬러 화면 기준 정렬(Align) 객체 생성
    align_to = rs.stream.color
    align = rs.align(align_to)

    # 리턴 값에 thres_filter 추가!
    return pipeline, align, temp_filter, thres_filter

def get_aligned_intrinsics(pipeline):
    """
    현재 활성화된 파이프라인에서 정렬된(Aligned) 영상의 인트린직 정보를 반환합니다.
    (뎁스가 컬러에 정렬되므로, 컬러 센서의 인트린직을 반환합니다.)
    """
    # 1. 파이프라인에서 현재 실행 중인 프로필(Profile) 가져오기
    active_profile = pipeline.get_active_profile()
    
    # 2. 컬러 스트림(rs.stream.color) 정보 가져오기
    color_stream = active_profile.get_stream(rs.stream.color)
    
    # 3. 비디오 스트림 프로필로 캐스팅한 뒤 인트린직(Intrinsics) 추출
    intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
    
    return intrinsics

def intrinsics_checker(pipeline):

    intrinsics = get_aligned_intrinsics(pipeline)
    # 인트린직 내부 값 확인하기

    print("\n[카메라 인트린직 정보]")
    print(f"해상도 (Width x Height): {intrinsics.width} x {intrinsics.height}")
    print(f"초점 거리 (fx, fy): {intrinsics.fx:.2f}, {intrinsics.fy:.2f}")
    print(f"주점/중심점 (ppx, ppy): {intrinsics.ppx:.2f}, {intrinsics.ppy:.2f}")
    print(f"왜곡 모델 (Distortion Model): {intrinsics.model}")
    print(f"왜곡 계수 (Coeffs): {intrinsics.coeffs}\n")

    return 0

def get_aligned_frames_with_units(
    pipeline,
    align,
    temp_filter,
    thres_filter,
    profile_depth_units=None,
    apply_filter=True
):
    """
    aligned depth/color 프레임을 받고,
    실제 3D 변환에 사용할 depth_scale까지 같이 반환.

    중요:
    - sensor.get_depth_scale()이 0.0001로 남아 있어도,
      macro_30처럼 profile에서 depth_Units=0.00001을 쓴 경우
      profile_depth_units를 우선 사용한다.
    """
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not aligned_depth_frame or not color_frame:
        return None, None, None, None

    if apply_filter:
        aligned_depth_frame = thres_filter.process(aligned_depth_frame)
        aligned_depth_frame = temp_filter.process(aligned_depth_frame)

    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # frame 자체의 unit 확인
    frame_units = None
    center_distance_m = None
    center_raw = None

    try:
        depth_frame_obj = aligned_depth_frame.as_depth_frame()
        frame_units = depth_frame_obj.get_units()

        H, W = depth_image.shape[:2]
        cx = W // 2
        cy = H // 2
        center_raw = int(depth_image[cy, cx])
        center_distance_m = float(depth_frame_obj.get_distance(cx, cy))
    except Exception as e:
        frame_units = None
        center_distance_m = None
        center_raw = None

    # sensor scale 확인
    try:
        depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
        sensor_scale = float(depth_sensor.get_depth_scale())
        sensor_depth_units_option = float(depth_sensor.get_option(rs.option.depth_units))
    except Exception:
        sensor_scale = None
        sensor_depth_units_option = None

    # 핵심:
    # profile_depth_units가 주어지면 그걸 우선 사용.
    # macro_30에서 depth_Units=0.00001을 쓰는 경우 이게 제일 안전.
    if profile_depth_units is not None:
        depth_scale_used = float(profile_depth_units)
    elif frame_units is not None:
        depth_scale_used = float(frame_units)
    elif sensor_scale is not None:
        depth_scale_used = float(sensor_scale)
    else:
        depth_scale_used = 0.001

    debug_info = {
        "frame_units": frame_units,
        "sensor_scale": sensor_scale,
        "sensor_depth_units_option": sensor_depth_units_option,
        "depth_scale_used": depth_scale_used,
        "center_raw": center_raw,
        "center_distance_m": center_distance_m,
        "center_raw_times_used_scale": None if center_raw is None else center_raw * depth_scale_used,
    }

    return depth_image, color_image, depth_scale_used, debug_info

def make_depth_colormap_meters(
    depth_img,
    depth_scale,
    min_m=0.08,
    max_m=0.35,
    colormap=cv2.COLORMAP_JET
):
    """
    raw depth가 아니라 meter 값 기준으로 컬러맵 생성.
    depth_units가 0.00001이든 0.0001이든 시각화가 일관됨.
    """
    depth_m = depth_img.astype(np.float32) * float(depth_scale)

    valid = (depth_m > min_m) & (depth_m < max_m)

    depth_norm = np.zeros_like(depth_img, dtype=np.uint8)

    if np.count_nonzero(valid) > 0:
        clipped = np.clip(depth_m, min_m, max_m)
        depth_norm[valid] = (
            (clipped[valid] - min_m) / (max_m - min_m) * 255.0
        ).astype(np.uint8)

    depth_colormap = cv2.applyColorMap(depth_norm, colormap)
    depth_colormap_rgb = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)

    # invalid는 검정
    depth_colormap_rgb[~valid] = 0

    return depth_colormap_rgb, depth_m


# 조립체 분석용 함수들

def depth_to_xyz_map(depth_img, depth_scale, intrinsics):
    """
    depth_img: uint16 raw depth image, H x W
    depth_scale: raw depth unit -> meter
    intrinsics: pyrealsense2 intrinsics
    return:
        xyz_map: H x W x 3, meter
        valid_mask: H x W, bool
    """
    H, W = depth_img.shape[:2]

    z = depth_img.astype(np.float32) * float(depth_scale)
    valid_mask = z > 0

    u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))

    x = (u_grid.astype(np.float32) - intrinsics.ppx) / intrinsics.fx * z
    y = (v_grid.astype(np.float32) - intrinsics.ppy) / intrinsics.fy * z

    xyz_map = np.stack([x, y, z], axis=-1)
    xyz_map[~valid_mask] = 0

    return xyz_map, valid_mask

def fit_plane_ransac_numpy(
    points,
    num_iter=500,
    distance_threshold=0.006,
    sample_size=3,
    max_points=15000,
    random_seed=0
):
    """
    points: N x 3, meter
    plane: ax + by + cz + d = 0
    return:
        best_plane: np.array([a, b, c, d])
        best_inlier_mask: N bool
    """
    rng = np.random.default_rng(random_seed)

    if points.shape[0] > max_points:
        idx = rng.choice(points.shape[0], size=max_points, replace=False)
        sample_points = points[idx]
    else:
        sample_points = points

    N = sample_points.shape[0]

    if N < 3:
        raise ValueError("RANSAC에 사용할 포인트가 너무 적습니다.")

    best_plane = None
    best_inlier_mask = None
    best_inlier_count = -1

    for _ in range(num_iter):
        ids = rng.choice(N, size=sample_size, replace=False)
        p1, p2, p3 = sample_points[ids]

        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)

        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            continue

        normal = normal / norm
        d = -np.dot(normal, p1)

        distances = np.abs(sample_points @ normal + d)
        inlier_mask = distances < distance_threshold
        inlier_count = np.count_nonzero(inlier_mask)

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_plane = np.array([normal[0], normal[1], normal[2], d], dtype=np.float32)
            best_inlier_mask = inlier_mask

    if best_plane is None:
        raise RuntimeError("RANSAC plane fitting 실패")

    # inlier들로 plane 한 번 더 정밀 보정
    inlier_points = sample_points[best_inlier_mask]
    centroid = np.mean(inlier_points, axis=0)
    centered = inlier_points - centroid

    _, _, vh = np.linalg.svd(centered)
    normal = vh[-1]
    normal = normal / np.linalg.norm(normal)
    d = -np.dot(normal, centroid)

    refined_plane = np.array([normal[0], normal[1], normal[2], d], dtype=np.float32)

    # 원본 points 기준 inlier mask 재계산
    distances_full = np.abs(points @ refined_plane[:3] + refined_plane[3])
    inlier_mask_full = distances_full < distance_threshold

    return refined_plane, inlier_mask_full

def compute_plane_distance_map(xyz_map, valid_mask, plane):
    """
    xyz_map: H x W x 3
    plane: [a, b, c, d]
    return:
        distance_map: H x W, meter
    """
    normal = plane[:3]
    d = plane[3]

    dist = np.abs(
        xyz_map[..., 0] * normal[0] +
        xyz_map[..., 1] * normal[1] +
        xyz_map[..., 2] * normal[2] +
        d
    )

    dist[~valid_mask] = 0
    return dist

def pca_2d_from_mask(mask):
    """
    mask 안의 픽셀 좌표 기준 PCA.
    return:
        center_uv: [u, v]
        major_axis_uv: [du, dv]
        minor_axis_uv: [du, dv]
        eigenvalues
        angle_deg
        major_length_px: 장축 방향 실제 투영 길이 [px]
        minor_length_px: 단축 방향 실제 투영 길이 [px]
    """
    ys, xs = np.where(mask > 0)

    if len(xs) < 5:
        return None

    pts = np.stack([xs, ys], axis=1).astype(np.float32)

    center = np.mean(pts, axis=0)
    centered = pts - center

    cov = centered.T @ centered / max(len(pts) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    major = eigvecs[:, 0]
    minor = eigvecs[:, 1]

    # 방향 안정화
    if major[0] < 0:
        major = -major

    angle_rad = np.arctan2(major[1], major[0])
    angle_deg = np.degrees(angle_rad)

    # ------------------------------------------------
    # 실제 마스크 픽셀을 PCA 축에 투영해서 길이 계산
    # ------------------------------------------------
    proj_major = centered @ major
    proj_minor = centered @ minor

    major_min = np.min(proj_major)
    major_max = np.max(proj_major)
    minor_min = np.min(proj_minor)
    minor_max = np.max(proj_minor)

    major_length_px = major_max - major_min
    minor_length_px = minor_max - minor_min

    return {
        "center_uv": center,
        "major_axis_uv": major,
        "minor_axis_uv": minor,
        "eigenvalues": eigvals,
        "angle_deg": angle_deg,

        "major_length_px": float(major_length_px),
        "minor_length_px": float(minor_length_px),

        "major_range_px": (float(major_min), float(major_max)),
        "minor_range_px": (float(minor_min), float(minor_max)),
    }

def pca_3d_from_mask(mask, xyz_map, valid_mask):
    """
    mask 안의 3D point 기준 PCA.
    return:
        center_xyz
        major_axis_xyz
        middle_axis_xyz
        minor_axis_xyz
        eigenvalues
        major_length_m
        middle_length_m
        minor_length_m
    """
    use_mask = (mask > 0) & valid_mask
    pts = xyz_map[use_mask]

    if pts.shape[0] < 10:
        return None

    center = np.mean(pts, axis=0)
    centered = pts - center

    cov = centered.T @ centered / max(pts.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    major = eigvecs[:, 0]
    middle = eigvecs[:, 1]
    minor = eigvecs[:, 2]

    proj_major = centered @ major
    proj_middle = centered @ middle
    proj_minor = centered @ minor

    major_length_m = np.max(proj_major) - np.min(proj_major)
    middle_length_m = np.max(proj_middle) - np.min(proj_middle)
    minor_length_m = np.max(proj_minor) - np.min(proj_minor)

    return {
        "center_xyz": center,
        "major_axis_xyz": major,
        "middle_axis_xyz": middle,
        "minor_axis_xyz": minor,
        "eigenvalues": eigvals,

        "major_length_m": float(major_length_m),
        "middle_length_m": float(middle_length_m),
        "minor_length_m": float(minor_length_m),
    }

def extract_object_components_with_pca(
    depth_img,
    depth_scale,
    intrinsics,
    color_img_rgb=None,
    and_mask=None,
    median_ksize=3,
    ransac_distance_threshold=0.006,
    object_min_plane_dist=0.010,
    min_area_px=80,
    morph_open_ksize=3,
    morph_close_ksize=5,
    show=True
):
    """
    depth_img: raw depth image
    depth_scale: meter scale
    intrinsics: aligned color intrinsics
    color_img_rgb: 시각화용 RGB 이미지
    and_mask: YOLO mask 같은 추가 마스크. None이면 depth 기반만 사용.
              shape은 depth_img와 같아야 함.
    """

    # -----------------------------
    # A. 약한 median 처리
    # -----------------------------
    if median_ksize is not None and median_ksize >= 3:
        depth_med = cv2.medianBlur(depth_img, median_ksize)
        depth_med[depth_img == 0] = 0
    else:
        depth_med = depth_img.copy()

    # -----------------------------
    # B. depth -> xyz
    # -----------------------------
    xyz_map, valid_mask = depth_to_xyz_map(
        depth_img=depth_med,
        depth_scale=depth_scale,
        intrinsics=intrinsics
    )

    points = xyz_map[valid_mask]

    if points.shape[0] < 100:
        raise ValueError("유효 depth point가 너무 적습니다.")

    # -----------------------------
    # C. RANSAC plane fitting
    # -----------------------------
    plane, plane_inlier_mask_1d = fit_plane_ransac_numpy(
        points=points,
        num_iter=700,
        distance_threshold=ransac_distance_threshold,
        max_points=20000
    )

    plane_dist_map = compute_plane_distance_map(
        xyz_map=xyz_map,
        valid_mask=valid_mask,
        plane=plane
    )

    floor_mask = valid_mask & (plane_dist_map < ransac_distance_threshold)

    # 바닥에서 일정 거리 이상 튀어나온 부분만 객체 후보
    object_depth_mask = valid_mask & (plane_dist_map > object_min_plane_dist)

    # -----------------------------
    # D. YOLO mask 등과 AND
    # -----------------------------
    if and_mask is not None:
        and_mask_bool = and_mask.astype(bool)

        if and_mask_bool.shape != object_depth_mask.shape:
            raise ValueError(
                f"and_mask shape이 depth_img와 다릅니다. "
                f"and_mask={and_mask_bool.shape}, depth={object_depth_mask.shape}"
            )

        object_mask = object_depth_mask & and_mask_bool
    else:
        object_mask = object_depth_mask

    object_mask_u8 = object_mask.astype(np.uint8) * 255

    # -----------------------------
    # E. Morphology 정리
    # -----------------------------
    if morph_open_ksize is not None and morph_open_ksize > 1:
        k_open = np.ones((morph_open_ksize, morph_open_ksize), np.uint8)
        object_mask_u8 = cv2.morphologyEx(object_mask_u8, cv2.MORPH_OPEN, k_open)

    if morph_close_ksize is not None and morph_close_ksize > 1:
        k_close = np.ones((morph_close_ksize, morph_close_ksize), np.uint8)
        object_mask_u8 = cv2.morphologyEx(object_mask_u8, cv2.MORPH_CLOSE, k_close)

    # -----------------------------
    # F. Connected Components
    # -----------------------------
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        object_mask_u8,
        connectivity=8
    )

    components = []

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]

        if area < min_area_px:
            continue

        comp_mask = (labels == label_id).astype(np.uint8)

        pca2d = pca_2d_from_mask(comp_mask)
        pca3d = pca_3d_from_mask(comp_mask, xyz_map, valid_mask)

        if pca2d is None or pca3d is None:
            continue

        x = stats[label_id, cv2.CC_STAT_LEFT]
        y = stats[label_id, cv2.CC_STAT_TOP]
        w = stats[label_id, cv2.CC_STAT_WIDTH]
        h = stats[label_id, cv2.CC_STAT_HEIGHT]

        components.append({
            "label_id": label_id,
            "area_px": int(area),
            "bbox_xywh": (int(x), int(y), int(w), int(h)),
            "mask": comp_mask,
            "pca2d": pca2d,
            "pca3d": pca3d,
        })

    # 큰 덩어리 순서로 정렬
    components = sorted(components, key=lambda c: c["area_px"], reverse=True)

    result = {
        "depth_med": depth_med,
        "xyz_map": xyz_map,
        "valid_mask": valid_mask,
        "plane": plane,
        "plane_dist_map": plane_dist_map,
        "floor_mask": floor_mask,
        "object_depth_mask": object_depth_mask,
        "object_mask": object_mask_u8,
        "labels": labels,
        "components": components
    }

    if show:
        visualize_components_pca(
            color_img_rgb=color_img_rgb,
            object_mask=object_mask_u8,
            floor_mask=floor_mask,
            components=components
        )

    return result

def visualize_components_pca(
    color_img_rgb,
    object_mask,
    floor_mask,
    components,
    axis_len=70
):
    if color_img_rgb is None:
        H, W = object_mask.shape[:2]
        vis = np.zeros((H, W, 3), dtype=np.uint8)
    else:
        vis = color_img_rgb.copy()

    # 혹시 BGR이 들어오면 색이 이상해질 수 있으니, 여기서는 RGB 기준으로 처리
    overlay = vis.copy()

    # 객체 마스크 영역을 초록색으로 표시
    overlay[object_mask > 0] = (
        0.5 * overlay[object_mask > 0] + 0.5 * np.array([0, 255, 0])
    ).astype(np.uint8)

    # 바닥 plane 영역을 어둡게 표시
    overlay[floor_mask] = (
        0.7 * overlay[floor_mask] + 0.3 * np.array([80, 80, 80])
    ).astype(np.uint8)

    draw = overlay.copy()

    for idx, comp in enumerate(components):
        pca2d = comp["pca2d"]
        center = pca2d["center_uv"]
        major = pca2d["major_axis_uv"]
        minor = pca2d["minor_axis_uv"]

        cx, cy = center.astype(int)

        # major axis
        p1 = (
            int(cx - major[0] * axis_len),
            int(cy - major[1] * axis_len)
        )
        p2 = (
            int(cx + major[0] * axis_len),
            int(cy + major[1] * axis_len)
        )

        # minor axis
        q1 = (
            int(cx - minor[0] * axis_len * 0.5),
            int(cy - minor[1] * axis_len * 0.5)
        )
        q2 = (
            int(cx + minor[0] * axis_len * 0.5),
            int(cy + minor[1] * axis_len * 0.5)
        )

        # RGB 이미지지만 cv2 line은 그냥 배열에 색값만 넣는 거라 RGB 색으로 지정
        cv2.line(draw, p1, p2, (255, 0, 0), 3)      # major axis: red
        cv2.line(draw, q1, q2, (0, 0, 255), 2)      # minor axis: blue
        cv2.circle(draw, (cx, cy), 5, (255, 255, 0), -1)

        x, y, w, h = comp["bbox_xywh"]
        cv2.rectangle(draw, (x, y), (x + w, y + h), (255, 255, 0), 2)

        angle = pca2d["angle_deg"]
        text = f"id={idx}, area={comp['area_px']}, yaw2d={angle:.1f}"
        cv2.putText(
            draw,
            text,
            (x, max(15, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            1,
            cv2.LINE_AA
        )

    plt.figure(figsize=(14, 7))
    plt.imshow(draw)
    plt.axis("off")
    plt.title("Object Components + PCA Axes")
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.imshow(object_mask, cmap="gray")
    plt.axis("off")
    plt.title("Final Object Mask")
    plt.show()

def extract_contour_pca_from_mask(
    object_mask,
    xyz_map=None,
    valid_mask=None,
    min_contour_area=80,
    axis_scale=1.0
):
    """
    object_mask:
        RANSAC 바닥 제거 후 남은 객체 마스크.
        0/255 또는 0/1 모두 가능.

    xyz_map:
        H x W x 3, meter.
        3D PCA도 같이 계산하고 싶으면 입력.

    valid_mask:
        H x W bool.
        xyz_map 사용 시 필요.

    return:
        contour_objects: list of dict
    """

    mask_u8 = (object_mask > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        mask_u8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contour_objects = []

    for contour_id, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        if area < min_contour_area:
            continue

        # -----------------------------------------
        # 1. contour 중심점 계산
        # -----------------------------------------
        M = cv2.moments(contour)

        if abs(M["m00"]) < 1e-6:
            continue

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        center_uv = np.array([cx, cy], dtype=np.float32)

        # -----------------------------------------
        # 2. contour 내부 mask 생성
        # -----------------------------------------
        contour_mask = np.zeros_like(mask_u8)
        cv2.drawContours(
            contour_mask,
            [contour],
            contourIdx=-1,
            color=255,
            thickness=-1
        )

        ys, xs = np.where(contour_mask > 0)

        if len(xs) < 5:
            continue

        pts_uv = np.stack([xs, ys], axis=1).astype(np.float32)

        # 중심은 contour moment 중심을 기준으로 사용
        centered_uv = pts_uv - center_uv

        # -----------------------------------------
        # 3. 2D PCA
        # -----------------------------------------
        cov_2d = centered_uv.T @ centered_uv / max(len(pts_uv) - 1, 1)

        eigvals_2d, eigvecs_2d = np.linalg.eigh(cov_2d)

        order = np.argsort(eigvals_2d)[::-1]
        eigvals_2d = eigvals_2d[order]
        eigvecs_2d = eigvecs_2d[:, order]

        major_axis_uv = eigvecs_2d[:, 0]
        minor_axis_uv = eigvecs_2d[:, 1]

        # 방향 안정화
        if major_axis_uv[0] < 0:
            major_axis_uv = -major_axis_uv

        # minor도 major에 맞춰 오른손 방향 느낌으로 정리
        minor_axis_uv = np.array(
            [-major_axis_uv[1], major_axis_uv[0]],
            dtype=np.float32
        )

        # -----------------------------------------
        # 4. contour 픽셀들을 PCA 축에 투영해서 실제 길이 계산
        # -----------------------------------------
        proj_major = centered_uv @ major_axis_uv
        proj_minor = centered_uv @ minor_axis_uv

        major_min = np.min(proj_major)
        major_max = np.max(proj_major)
        minor_min = np.min(proj_minor)
        minor_max = np.max(proj_minor)

        major_length_px = major_max - major_min
        minor_length_px = minor_max - minor_min

        angle_rad = np.arctan2(major_axis_uv[1], major_axis_uv[0])
        angle_deg = np.degrees(angle_rad)

        # -----------------------------------------
        # 5. minAreaRect도 같이 저장
        # -----------------------------------------
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = box.astype(np.int32)

        x, y, w, h = cv2.boundingRect(contour)

        obj = {
            "contour_id": contour_id,
            "contour": contour,
            "contour_mask": contour_mask,
            "area_px": float(area),
            "bbox_xywh": (int(x), int(y), int(w), int(h)),

            "center_uv": center_uv,

            "major_axis_uv": major_axis_uv.astype(np.float32),
            "minor_axis_uv": minor_axis_uv.astype(np.float32),

            "major_length_px": float(major_length_px),
            "minor_length_px": float(minor_length_px),
            "major_range_px": (float(major_min), float(major_max)),
            "minor_range_px": (float(minor_min), float(minor_max)),

            "angle_deg": float(angle_deg),

            "eigvals_2d": eigvals_2d,

            "min_area_rect": rect,
            "min_area_box": box,
        }

        # -----------------------------------------
        # 6. 선택: 3D PCA도 계산
        # -----------------------------------------
        if xyz_map is not None and valid_mask is not None:
            use_3d_mask = (contour_mask > 0) & valid_mask
            pts_xyz = xyz_map[use_3d_mask]

            if pts_xyz.shape[0] >= 10:
                center_xyz = np.mean(pts_xyz, axis=0)
                centered_xyz = pts_xyz - center_xyz

                cov_3d = centered_xyz.T @ centered_xyz / max(pts_xyz.shape[0] - 1, 1)

                eigvals_3d, eigvecs_3d = np.linalg.eigh(cov_3d)

                order3 = np.argsort(eigvals_3d)[::-1]
                eigvals_3d = eigvals_3d[order3]
                eigvecs_3d = eigvecs_3d[:, order3]

                major_axis_xyz = eigvecs_3d[:, 0]
                middle_axis_xyz = eigvecs_3d[:, 1]
                minor_axis_xyz = eigvecs_3d[:, 2]

                proj_x = centered_xyz @ major_axis_xyz
                proj_y = centered_xyz @ middle_axis_xyz
                proj_z = centered_xyz @ minor_axis_xyz

                obj["center_xyz"] = center_xyz
                obj["major_axis_xyz"] = major_axis_xyz
                obj["middle_axis_xyz"] = middle_axis_xyz
                obj["minor_axis_xyz"] = minor_axis_xyz

                obj["major_length_m"] = float(np.max(proj_x) - np.min(proj_x))
                obj["middle_length_m"] = float(np.max(proj_y) - np.min(proj_y))
                obj["minor_length_m"] = float(np.max(proj_z) - np.min(proj_z))

                obj["eigvals_3d"] = eigvals_3d

        contour_objects.append(obj)

    contour_objects = sorted(
        contour_objects,
        key=lambda o: o["area_px"],
        reverse=True
    )

    return contour_objects

def visualize_contour_pca_axes(
    color_img_rgb,
    object_mask,
    contour_objects,
    draw_mask=True,
    draw_contour=True,
    draw_min_rect=True,
    axis_len_mode="pca_length",
    fixed_axis_len=80
):
    """
    axis_len_mode:
        "pca_length" -> contour 투영 길이 기반으로 축 길이 그림
        "fixed"      -> fixed_axis_len으로 고정 길이 그림
    """

    if color_img_rgb is None:
        H, W = object_mask.shape[:2]
        vis = np.zeros((H, W, 3), dtype=np.uint8)
    else:
        vis = color_img_rgb.copy()

    mask_u8 = (object_mask > 0).astype(np.uint8) * 255

    overlay = vis.copy()

    if draw_mask:
        overlay[mask_u8 > 0] = (
            0.55 * overlay[mask_u8 > 0] +
            0.45 * np.array([0, 255, 0])
        ).astype(np.uint8)

    draw = overlay.copy()

    for idx, obj in enumerate(contour_objects):
        contour = obj["contour"]
        center = obj["center_uv"]
        major = obj["major_axis_uv"]
        minor = obj["minor_axis_uv"]

        cx, cy = center
        cxi, cyi = int(round(cx)), int(round(cy))

        if axis_len_mode == "pca_length":
            major_half_len = obj["major_length_px"] * 0.5
            minor_half_len = obj["minor_length_px"] * 0.5
        else:
            major_half_len = fixed_axis_len
            minor_half_len = fixed_axis_len * 0.5

        # 장축 endpoints
        p1 = (
            int(round(cx - major[0] * major_half_len)),
            int(round(cy - major[1] * major_half_len))
        )
        p2 = (
            int(round(cx + major[0] * major_half_len)),
            int(round(cy + major[1] * major_half_len))
        )

        # 단축 endpoints
        q1 = (
            int(round(cx - minor[0] * minor_half_len)),
            int(round(cy - minor[1] * minor_half_len))
        )
        q2 = (
            int(round(cx + minor[0] * minor_half_len)),
            int(round(cy + minor[1] * minor_half_len))
        )

        if draw_contour:
            cv2.drawContours(draw, [contour], -1, (255, 255, 0), 2)

        if draw_min_rect:
            box = obj["min_area_box"]
            cv2.drawContours(draw, [box], 0, (255, 0, 255), 2)

        # PCA 장축: 빨강
        cv2.line(draw, p1, p2, (255, 0, 0), 3)

        # PCA 단축: 파랑
        cv2.line(draw, q1, q2, (0, 0, 255), 2)

        # 중심점: 노랑
        cv2.circle(draw, (cxi, cyi), 5, (255, 255, 0), -1)

        x, y, w, h = obj["bbox_xywh"]

        text1 = f"id={idx}, area={obj['area_px']:.0f}"
        text2 = f"yaw={obj['angle_deg']:.1f}, L={obj['major_length_px']:.1f}, W={obj['minor_length_px']:.1f}"

        cv2.putText(
            draw,
            text1,
            (x, max(15, y - 22)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            draw,
            text2,
            (x, max(15, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
            cv2.LINE_AA
        )

    plt.figure(figsize=(14, 7))
    plt.imshow(draw)
    plt.axis("off")
    plt.title("RANSAC Object Mask Contours + PCA Axes")
    plt.show()

    return draw



# 포인트 클라우드 함수
def refine_2d_mask_with_hull(projected_mask_01, color_bgr):
    """
    파먹히고 조각난 2D 투영 마스크(0 or 1)를 Convex Hull과 모폴로지 연산을 
    이용해 꽉 찬(Solid) 객체 마스크로 복원하고 컬러 이미지를 커팅합니다.
    """
    # 1. OpenCV 연산을 위해 마스크를 0~255 스케일(uint8)로 변환
    mask_255 = (projected_mask_01 * 255).astype(np.uint8)
    
    # 2. 미세하게 끊어진 조각들을 하나로 뭉치기 위해 팽창(Dilation) 및 닫기(Close) 적용
    # 커널 사이즈(3x3)는 객체가 서로 너무 가까워 붙지 않는 선에서 조절 (필요시 5x5 로 변경)
    kernel = np.ones((7, 7), np.uint8)
    closed_mask = cv2.morphologyEx(mask_255, cv2.MORPH_CLOSE, kernel)
    
    # 3. 마스크 내의 모든 윤곽선(Contours) 찾기
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 4. 복원된 마스크를 그릴 빈 도화지 생성
    refined_mask = np.zeros_like(mask_255)
    
    # 5. 각 윤곽선에 대해 Convex Hull(볼록 껍질)을 구하고 꽉 채워서 그리기
    for cnt in contours:
        # 노이즈(너무 작은 덩어리)는 무시 (면적 임계값: 100픽셀)
        if cv2.contourArea(cnt) > 200:
            # 윤곽선을 고무줄로 묶듯이 볼록한 형태로 감쌈
            hull = cv2.convexHull(cnt)
            # 도화지에 하얀색(255)으로 꽉 채워서(thickness=-1) 그리기
            cv2.drawContours(refined_mask, [hull], 0, 255, -1)
            
    # 6. 최종 복원된 마스크를 0과 1로 다시 변환
    refined_mask_01 = (refined_mask / 255).astype(np.uint8)
    
    # 7. 완벽해진 마스크로 원본 컬러 이미지 다시 커팅
    final_cut_color = cv2.bitwise_and(color_bgr, color_bgr, mask=refined_mask_01)
    
    return refined_mask_01, final_cut_color

def refine_2d_mask_with_hull(projected_mask_01, color_bgr):
    # [함수 1] 조각난 2D 마스크를 Convex Hull로 복원
    mask_255 = (projected_mask_01 * 255).astype(np.uint8)
    kernel = np.ones((7, 7), np.uint8)
    closed_mask = cv2.morphologyEx(mask_255, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    refined_mask = np.zeros_like(mask_255)
    
    for cnt in contours:
        if cv2.contourArea(cnt) > 200:
            hull = cv2.convexHull(cnt)
            cv2.drawContours(refined_mask, [hull], 0, 255, -1)
            
    refined_mask_01 = (refined_mask / 255).astype(np.uint8)
    final_cut_color = cv2.bitwise_and(color_bgr, color_bgr, mask=refined_mask_01)
    return refined_mask_01, final_cut_color

def create_floor_anchored_3d_box_with_axes(box_2d, intrinsics, plane_normal, d, max_h, color, axis_size=0.03):
# [함수] 2D OBB를 3D 바닥 평면으로 역투영하여 3D 박스 및 좌표계 생성
    base_corners = []
    # 1. 2D 픽셀 4개의 꼭짓점을 3D 바닥 평면에 대입하여 교점 획득
    for (u, v) in box_2d:
        dir_vec = np.array([(u - intrinsics.ppx) / intrinsics.fx,
                            (v - intrinsics.ppy) / intrinsics.fy,
                            1.0])
        t = -d / np.dot(plane_normal, dir_vec)
        base_pt = t * dir_vec
        base_corners.append(base_pt)
        
    base_corners = np.array(base_corners)
    top_corners = base_corners + (plane_normal * max_h)
    
    # 2. 3D 박스 외곽선(LineSet) 생성
    vertices = np.vstack((base_corners, top_corners))
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0], # 바닥면
        [4, 5], [5, 6], [6, 7], [7, 4], # 천장면
        [0, 4], [1, 5], [2, 6], [3, 7]  # 수직 기둥
    ]
    
    colors = [color for _ in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # 🎯 3. 로컬 좌표계(Coordinate Frame) 생성 로직
    # 박스 중심점 계산
    center = np.mean(vertices, axis=0)

    # Z축: 평면의 법선 벡터 (바닥에서 위를 향함)
    Z = plane_normal / np.linalg.norm(plane_normal)

    # X축: 바닥의 변 중에서 '더 긴 변(Major Axis)'을 X축으로 설정 (그리퍼 파지 방향)
    vec1 = base_corners[1] - base_corners[0]
    vec2 = base_corners[2] - base_corners[1]
    
    if np.linalg.norm(vec1) > np.linalg.norm(vec2):
        X_dir = vec1
    else:
        X_dir = vec2
        
    X = X_dir / np.linalg.norm(X_dir)

    # Y축: Z와 X의 외적 (직교 보장)
    Y = np.cross(Z, X)
    Y = Y / np.linalg.norm(Y)

    # 회전 행렬 조립 (3x3 Matrix)
    R = np.column_stack((X, Y, Z))

    # Open3D 좌표계 메쉬 생성 및 이동
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size)
    axes.rotate(R, center=(0, 0, 0)) # 회전 적용
    axes.translate(center)           # 박스 중심으로 이동

    # 리턴값에 좌표계(axes) 추가
    return line_set, axes

def create_floor_anchored_3d_box(box_2d, intrinsics, plane_normal, d, max_h, color):
# [함수 2] 2D OBB를 3D 바닥 평면으로 역투영하여 바닥 밀착형 박스 생성
    base_corners = []
    # 2D 픽셀 4개의 꼭짓점을 3D 바닥 평면에 대입하여 교점 획득
    for (u, v) in box_2d:
        dir_vec = np.array([(u - intrinsics.ppx) / intrinsics.fx,
                            (v - intrinsics.ppy) / intrinsics.fy,
                            1.0])
        t = -d / np.dot(plane_normal, dir_vec)
        base_pt = t * dir_vec
        base_corners.append(base_pt)
        
    base_corners = np.array(base_corners)
    top_corners = base_corners + (plane_normal * max_h)
    
    vertices = np.vstack((base_corners, top_corners))
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0], # 바닥면
        [4, 5], [5, 6], [6, 7], [7, 4], # 천장면
        [0, 4], [1, 5], [2, 6], [3, 7]  # 수직 기둥
    ]
    
    colors = [color for _ in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


# 컨트롤 실행부 함수들




def capture_realsense_data(serial_number, mode="mid_50", warmup_frames=30, visualize=False):
    """
    특정 리얼센스 카메라를 지정한 모드로 켜서 예열한 뒤, 핵심 비전 데이터를 추출하는 함수.
    
    Args:
        serial_number (str): ivl.get_realsense_ids()로 찾은 기기 시리얼 번호
        mode (str): "floor", "macro_30", "mid_50" 중 택 1
        warmup_frames (int): 센서 안정화를 위해 버릴 초기 프레임 수
        visualize (bool): 캡처된 결과(Color + Depth)를 Matplotlib으로 출력할지 여부
        
    Returns:
        color_img_rgb (ndarray): RGB 포맷의 컬러 이미지 (YOLO, Open3D용)
        depth_img (ndarray): Raw 뎁스 이미지
        intrinsics (rs.intrinsics): 카메라 내부 파라미터 (3D 투영용)
        depth_scale (float): 뎁스 단위를 미터(m)로 변환하기 위한 스케일 값 (매우 중요)
    """
    print(f"[{mode}] 모드로 카메라(ID: {serial_number}) 구동을 시작합니다...")
    
    # 1. 프로필 파라미터 로드
    profile_params = CAMERA_PROFILES.get(mode)
    if profile_params is None:
        raise ValueError(f"지원하지 않는 모드입니다: {mode}")
        
    profile_depth_units = profile_params.get("depth_Units", None)
    
    # 2. 카메라 파이프라인 설정 및 구동
    pipeline, align, temp_filter, thres_filter = configure_realsense(
        serial_number=serial_number, # 💡 주의: ivl 라이브러리 수정 필요 (아래 참고)
        **profile_params
    )
    
    intrinsics = None
    color_img = None
    depth_img = None
    depth_scale = None
    
    try:
        # 3. 센서 예열 (안정화)
        if warmup_frames > 0:
            print(f"🔥 센서 안정화 중... ({warmup_frames} 프레임 대기)")
            for _ in range(warmup_frames):
                pipeline.wait_for_frames()
                
        # 4. 프레임 획득 (안정성을 위해 2번 돌고 마지막 프레임 사용)
        for _ in range(2):
            depth_img, color_img, depth_scale, debug_info = get_aligned_frames_with_units(
                pipeline=pipeline,
                align=align,
                temp_filter=temp_filter,
                thres_filter=thres_filter,
                profile_depth_units=profile_depth_units,
                apply_filter=True
            )
            
        intrinsics = get_aligned_intrinsics(pipeline)
        
        # 5. 시각화 (옵션)
        if visualize and depth_img is not None and color_img is not None:
            # 모드별 가시화 거리 설정
            vis_ranges = {
                "macro_30": (0.08, 0.35),
                "mid_50": (0.15, 0.80),
                "floor": (0.20, 3.00)
            }
            vis_min_m, vis_max_m = vis_ranges.get(mode, (0.20, 1.00))
            
            depth_colormap_rgb, _ = make_depth_colormap_meters(
                depth_img=depth_img, depth_scale=depth_scale, 
                min_m=vis_min_m, max_m=vis_max_m
            )
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            images = np.hstack((color_img, depth_colormap_rgb))
            ax.imshow(images)
            ax.axis("off")
            ax.set_title(f"Capture Result | Mode: {mode} | Scale: {depth_scale:.6f}")
            plt.show()

    except Exception as e:
        print(f"❌ 프레임 캡처 중 에러 발생: {e}")
        
    finally:
        pipeline.stop()
        print("✅ 카메라 스트리밍 안전 종료 완료.")
        
    # 6. BGR -> RGB 변환 (대부분의 딥러닝/비전 라이브러리 규격에 맞춤)
    color_img_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB) if color_img is not None else None
        
    return color_img_rgb, depth_img, intrinsics, depth_scale




