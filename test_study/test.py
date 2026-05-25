#!/usr/bin/env python
# coding: utf-8

# # 관련 단축키 설명

# ## 1. 셀 코드 실행
# 
# Shift + Enter: 현재 셀을 실행하고 다음 셀로 이동합니다. (가장 많이 사용)
# 
# Ctrl + Enter: 현재 셀을 실행하고 현재 셀에 머무릅니다. 결과를 확인하고 코드를 계속 수정할 때 유용합니다. (Mac: Cmd + Enter)
# 
# Alt + Enter: 현재 셀을 실행하고 바로 아래에 새로운 코드 셀을 추가합니다. (Mac: Option + Enter)

# ## 2. 셀 삭제
# 
# 명령 모드(Esc) 상태에서:
# 
# D , D: 키보드 알파벳 D를 연속으로 두 번 타닥 누르면 셀이 삭제됩니다.

# ## 3. 코드 셀 만들기
# 
# 명령 모드(Esc) 상태에서:
# 
# * A: 현재 셀 위(Above)에 새로운 코드 셀을 추가합니다.
# 
# * B: 현재 셀 아래(Below)에 새로운 코드 셀을 추가합니다.
# 
# * Y: 마크다운 등 다른 타입의 셀을 다시 코드(Code) 셀로 변경합니다.

# ## 4. 마크다운 셀 만들기
# 
# 명령 모드(Esc) 상태에서:
# 
# M: 현재 셀을 마크다운(Markdown) 셀로 변경합니다.
# 
# 실전 팁: 보통 마크다운 셀을 새로 만들 때는 **B**를 눌러 아래에 빈 코드 셀을 만든 직후, 바로 **M**을 눌러 마크다운 셀로 변환해서 글을 작성하는 방식을 가장 많이 사용합니다. 실수로 셀을 지웠다면 당황하지 말고 Z 키를 누르면 방금 삭제한 셀이 복구됩니다.

# ---

# # **비전 노드 개발 관련 실험코드**
# 
# ### 리얼센스 D435IF 기반 듀플로 단일 블럭 및 조립 객체 인식 상태 개발

# ---

# In[ ]:


pip install nbconvert
# jupyter nbconvert --to script INUVision.ipynb


# ## **리얼센스 카메라 인식 하는 함수**

# In[992]:


import os
import yaml
import numpy as np
import cv2
import pyrealsense2 as rs
import matplotlib.pyplot as plt
import pprint
from sklearn.cluster import DBSCAN
# !pip install scikit-learn


# In[993]:


def list_realsense_devices():
    """
    현재 연결된 Intel RealSense 카메라 목록을 출력하는 함수
    """
    ctx = rs.context()
    devices = ctx.query_devices()

    device_count = len(devices)

    print(f"[INFO] 연결된 RealSense 장치 수: {device_count}")

    if device_count == 0:
        print("[WARN] 연결된 RealSense 카메라가 없습니다.")
        return []

    device_info_list = []

    for i, dev in enumerate(devices):
        print("\n" + "=" * 50)
        print(f"[DEVICE {i}]")

        info = {}

        for info_type in [
            rs.camera_info.name,
            rs.camera_info.serial_number,
            rs.camera_info.firmware_version,
            rs.camera_info.physical_port,
            rs.camera_info.product_id,
            rs.camera_info.product_line,
        ]:
            if dev.supports(info_type):
                key = str(info_type).split(".")[-1]
                value = dev.get_info(info_type)
                info[key] = value
                print(f"{key}: {value}")

        sensors = dev.query_sensors()
        print(f"sensor count: {len(sensors)}")

        for j, sensor in enumerate(sensors):
            sensor_name = sensor.get_info(rs.camera_info.name)
            print(f"  - Sensor {j}: {sensor_name}")

        device_info_list.append(info)

    print("\n[INFO] RealSense 장치 확인 완료")
    return device_info_list


# In[994]:


devices = list_realsense_devices()


# ## **카메라에서 640 480 기준의 칼라, 뎁스 이미지 인트린직 요청**

# In[995]:


def capture_color_depth_and_save_intrinsics(
    devices,
    image_save_path="output/realsense_color.png",
    depth_save_path="output/realsense_depth_aligned.png", 
    yaml_save_path="output/realsense_intrinsics.yaml",
    width=640,
    height=480,
    fps=30,
    warmup_frames=10,
    show=True,
    
    # --- 필터 파라미터 ---
    use_realsense_filter=True,
    use_decimation=False,  # 듀플로 경계 보존을 위해 기본 OFF
    use_spatial=True,
    spatial_magnitude=2,
    spatial_smooth_alpha=0.5,
    spatial_smooth_delta=20,
    spatial_holes_fill=0,
    use_temporal=True,
    temporal_smooth_alpha=0.4,
    temporal_smooth_delta=20,
    temporal_persistency_index=3,
    use_hole_filling=False,
    hole_filling_mode=1
):
    """
    devices의 첫 번째 RealSense 카메라에서
    color image, aligned & filtered depth image를 1장 캡처하고,
    color/depth intrinsics를 출력 및 yaml로 저장한다.
    """

    import os
    import yaml
    import cv2
    import numpy as np
    import pyrealsense2 as rs
    import matplotlib.pyplot as plt

    # ---------------------------------------------------------
    # 1. 기기 유효성 검사
    # ---------------------------------------------------------
    if devices is None or len(devices) == 0:
        raise ValueError("devices가 비어 있습니다. 먼저 list_realsense_devices()를 실행하세요.")

    first_device = devices[0]
    if "serial_number" not in first_device:
        raise KeyError("devices[0] 안에 'serial_number'가 없습니다.")

    serial_number = first_device["serial_number"]
    print(f"[INFO] 첫 번째 카메라 serial_number: {serial_number}")

    # ---------------------------------------------------------
    # 2. 파이프라인 및 스트림 설정
    # ---------------------------------------------------------
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_device(serial_number)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    try:
        # 🔥 스트림 시작 (이후에 센서 설정 가능)
        profile = pipeline.start(config)
        print("[INFO] RealSense color/depth stream 시작")

        # 🔥 컬러 화이트 밸런스 수동 고정 (푸른 스머프 현상 방지)
        sensor = profile.get_device().first_color_sensor()
        sensor.set_option(rs.option.enable_auto_white_balance, 0)
        sensor.set_option(rs.option.white_balance, 4500) # 필요시 4000~5000 사이 조절
        print("[INFO] 화이트 밸런스 수동(4500K) 고정 완료")

        # 얼라인 객체 생성
        align = rs.align(rs.stream.color)

        # depth scale 확인
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print(f"[INFO] depth_scale: {depth_scale} meter/unit")

        # ---------------------------------------------------------
        # 3. 필터 객체 초기화
        # ---------------------------------------------------------
        if use_realsense_filter:
            depth_to_disparity = rs.disparity_transform(True)
            disparity_to_depth = rs.disparity_transform(False)
            
            if use_decimation:
                decimation = rs.decimation_filter()
            if use_spatial:
                spatial = rs.spatial_filter()
                spatial.set_option(rs.option.filter_magnitude, spatial_magnitude)
                spatial.set_option(rs.option.filter_smooth_alpha, spatial_smooth_alpha)
                spatial.set_option(rs.option.filter_smooth_delta, spatial_smooth_delta)
                spatial.set_option(rs.option.holes_fill, spatial_holes_fill)
            if use_temporal:
                temporal = rs.temporal_filter()
                temporal.set_option(rs.option.filter_smooth_alpha, temporal_smooth_alpha)
                temporal.set_option(rs.option.filter_smooth_delta, temporal_smooth_delta)
                temporal.set_option(rs.option.holes_fill, temporal_persistency_index)
            if use_hole_filling:
                hole_filling = rs.hole_filling_filter()
                hole_filling.set_option(rs.option.holes_fill, hole_filling_mode)

        # ---------------------------------------------------------
        # 4. 웜업 및 Temporal 필터 히스토리 누적
        # ---------------------------------------------------------
        for _ in range(warmup_frames):
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            
            if use_realsense_filter:
                if use_decimation: depth_frame = decimation.process(depth_frame)
                depth_frame = depth_to_disparity.process(depth_frame)
                if use_spatial: depth_frame = spatial.process(depth_frame)
                if use_temporal: depth_frame = temporal.process(depth_frame) # 과거 프레임 누적
                depth_frame = disparity_to_depth.process(depth_frame)
                if use_hole_filling: depth_frame = hole_filling.process(depth_frame)

        # ---------------------------------------------------------
        # 5. 최종 프레임 획득 및 데이터 추출
        # ---------------------------------------------------------
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame: raise RuntimeError("컬러 프레임을 가져오지 못했습니다.")
        if not depth_frame: raise RuntimeError("얼라인된 뎁스 프레임을 가져오지 못했습니다.")

        # 최종 뎁스 필터 적용
        if use_realsense_filter:
            if use_decimation: depth_frame = decimation.process(depth_frame)
            depth_frame = depth_to_disparity.process(depth_frame)
            if use_spatial: depth_frame = spatial.process(depth_frame)
            if use_temporal: depth_frame = temporal.process(depth_frame)
            depth_frame = disparity_to_depth.process(depth_frame)
            if use_hole_filling: depth_frame = hole_filling.process(depth_frame)

        # numpy 배열로 변환
        image_bgr = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # 🔥 최종 프레임에서 Intrinsics 추출 (필터 통과 후 해상도 변경 대비)
        color_intr = color_frame.profile.as_video_stream_profile().intrinsics
        depth_intr = depth_frame.profile.as_video_stream_profile().intrinsics

        print("\n[COLOR INTRINSICS]")
        print(f"width  : {color_intr.width}")
        print(f"height : {color_intr.height}")
        print(f"fx     : {color_intr.fx:.3f}, fy: {color_intr.fy:.3f}")
        print(f"cx     : {color_intr.ppx:.3f}, cy: {color_intr.ppy:.3f}")

    finally:
        pipeline.stop()
        print("\n[INFO] RealSense stream 종료")

    # ---------------------------------------------------------
    # 6. 결과 저장 (이미지 & YAML)
    # ---------------------------------------------------------
    for path in [image_save_path, depth_save_path, yaml_save_path]:
        save_dir = os.path.dirname(path)
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    cv2.imwrite(image_save_path, image_rgb)
    cv2.imwrite(depth_save_path, depth_image)
    print(f"[INFO] 컬러 이미지 저장 완료: {image_save_path}")
    print(f"[INFO] 뎁스 얼라인 이미지 저장 완료: {depth_save_path}")

    intrinsics_dict = {
        "device": {"serial_number": serial_number},
        "color": {
            "width": int(color_intr.width), "height": int(color_intr.height),
            "fx": float(color_intr.fx), "fy": float(color_intr.fy),
            "cx": float(color_intr.ppx), "cy": float(color_intr.ppy),
            "model": str(color_intr.model), "coeffs": [float(v) for v in color_intr.coeffs]
        },
        "depth": {
            "width": int(depth_intr.width), "height": int(depth_intr.height),
            "fx": float(depth_intr.fx), "fy": float(depth_intr.fy),
            "cx": float(depth_intr.ppx), "cy": float(depth_intr.ppy),
            "model": str(depth_intr.model), "coeffs": [float(v) for v in depth_intr.coeffs],
            "depth_scale": float(depth_scale)
        }
    }

    with open(yaml_save_path, "w", encoding="utf-8") as f:
        yaml.dump(intrinsics_dict, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    # ---------------------------------------------------------
    # 7. 시각화 (matplotlib)
    # ---------------------------------------------------------
    if show:
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )
        depth_colormap_rgb = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)

        intr_text = (
            "[INTRINSICS]\n"
            f"Size   : {color_intr.width}x{color_intr.height}\n"
            f"fx     : {color_intr.fx:.3f}\n"
            f"fy     : {color_intr.fy:.3f}\n"
            f"cx     : {color_intr.ppx:.3f}\n"
            f"cy     : {color_intr.ppy:.3f}\n\n"
            "[DEPTH SCALE]\n"
            f"{depth_scale:.8f} m/unit"
        )

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw={"width_ratios": [1.1, 2, 2]})

        axes[0].axis("off")
        axes[0].set_title("Camera Settings")
        axes[0].text(0.0, 0.95, intr_text, fontsize=10, family="monospace", verticalalignment="top")

        axes[1].imshow(image_rgb)
        axes[1].set_title("Color Image (WB Locked)")
        axes[1].axis("off")

        axes[2].imshow(depth_colormap_rgb)
        axes[2].set_title("Filtered & Aligned Depth Image")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()

    return image_bgr, depth_image, intrinsics_dict


# In[996]:


color_img, depth_img, intrinsics = capture_color_depth_and_save_intrinsics(
    devices=devices,
    image_save_path="output/realsense_color.png",
    depth_save_path="output/realsense_depth_raw.png",
    yaml_save_path="output/realsense_intrinsics.yaml",
    width=640,
    height=480,
    fps=30,
    warmup_frames=10,
    show=True
)


# # **뎁스 정규화**

# In[997]:


def normalize_depth_for_display(depth_img, valid_mask=None):
    """
    depth 이미지를 matplotlib/cv2 시각화용 uint8로 변환.
    실제 depth 계산용 아님.
    """

    if valid_mask is None:
        valid_mask = (depth_img > 0).astype(np.uint8)

    depth_float = depth_img.astype(np.float32)
    valid_values = depth_float[valid_mask > 0]

    if valid_values.size == 0:
        return np.zeros_like(depth_img, dtype=np.uint8)

    min_d = valid_values.min()
    max_d = valid_values.max()

    depth_vis = np.zeros_like(depth_float, dtype=np.uint8)

    if max_d > min_d:
        depth_vis[valid_mask > 0] = (
            (depth_float[valid_mask > 0] - min_d) / (max_d - min_d) * 255.0
        ).astype(np.uint8)

    return depth_vis


# In[998]:


depth_filtered = normalize_depth_for_display(depth_img)
show_depth_compare(depth_img, depth_filtered, title1="Original Depth", title2="Original Depth (for comparison)")


# # **MEDIAN**

# In[999]:


def apply_depth_median_filter(
    depth_img,
    ksize=5,
    valid_mask=None,
    fill_holes=True, # 🔥 0값(Invalid)에 의한 엣지 파먹힘 방지 옵션
    show=False,
    title="Depth Median Filter"
):
    """
    depth 이미지에 median filter를 적용한다.
    (선택적으로 Hole Filling을 선행하여 0값에 의한 엣지 파먹힘을 방지한다.)
    """

    if depth_img is None:
        raise ValueError("depth_img가 None입니다.")
    if not isinstance(depth_img, np.ndarray):
        raise TypeError("depth_img는 np.ndarray 형식이어야 합니다.")
    if depth_img.ndim != 2:
        raise ValueError(f"depth_img는 2D depth 이미지여야 합니다. 현재 shape: {depth_img.shape}")
    if ksize % 2 == 0 or ksize < 3:
        raise ValueError("ksize는 최소 3 이상의 홀수여야 합니다. 예: 3, 5, 7")

    # ---------------------------------------------------------
    # [1] 0값(Invalid)에 의한 엣지 침식 방지 (Hole Filling)
    # ---------------------------------------------------------
    if fill_holes:
        kernel = np.ones((3, 3), np.uint16)
        depth_to_filter = cv2.morphologyEx(depth_img, cv2.MORPH_CLOSE, kernel)
    else:
        depth_to_filter = depth_img.copy()

    # ---------------------------------------------------------
    # [2] Median Filtering
    # ---------------------------------------------------------
    depth_filtered = cv2.medianBlur(depth_to_filter, ksize)

    # ---------------------------------------------------------
    # [3] 통계 정보 계산
    # ---------------------------------------------------------
    if valid_mask is None:
        valid_bool = depth_img > 0
    else:
        if valid_mask.shape != depth_img.shape:
            raise ValueError("valid_mask와 depth_img의 shape가 다릅니다.")
        valid_bool = valid_mask > 0

    valid_before = depth_img[valid_bool]
    valid_after = depth_filtered[valid_bool]

    before_median, before_mean, before_std = None, None, None
    if valid_before.size > 0:
        before_median = float(np.median(valid_before))
        before_mean = float(np.mean(valid_before))
        before_std = float(np.std(valid_before))

    after_median, after_mean, after_std = None, None, None
    if valid_after.size > 0:
        after_median = float(np.median(valid_after))
        after_mean = float(np.mean(valid_after))
        after_std = float(np.std(valid_after))

    filter_info = {
        "ksize": int(ksize),
        "hole_filled": fill_holes,
        "before_median_mm": before_median,
        "before_mean_mm": before_mean,
        "before_std_mm": before_std,
        "after_median_mm": after_median,
        "after_mean_mm": after_mean,
        "after_std_mm": after_std,
    }

    # ---------------------------------------------------------
    # [4] 시각화 (show=True 일 때만 작동)
    # ---------------------------------------------------------
    if show:
        print("[DEPTH MEDIAN FILTER]")
        print(f"ksize: {ksize}, Hole Filling: {fill_holes}")
        
        raw_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)
        filtered_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_filtered, alpha=0.03), cv2.COLORMAP_JET)

        raw_rgb = cv2.cvtColor(raw_colormap, cv2.COLOR_BGR2RGB)
        filtered_rgb = cv2.cvtColor(filtered_colormap, cv2.COLOR_BGR2RGB)

        diff = cv2.absdiff(depth_img, depth_filtered)
        diff_vis = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].imshow(raw_rgb)
        axes[0].set_title("Raw Depth")
        axes[0].axis("off")

        axes[1].imshow(filtered_rgb)
        axes[1].set_title(title)
        axes[1].axis("off")

        axes[2].imshow(diff_vis, cmap="gray")
        axes[2].set_title("Absolute Difference")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()

    return depth_filtered, filter_info


# In[1000]:


valid_mask = (depth_img > 0).astype(np.uint8)

depth_filtered, depth_filter_info = apply_depth_median_filter(
    depth_img=depth_img,
    ksize=5,
    valid_mask=valid_mask,
    fill_holes=True, # 엣지 파먹힘 방지 ON
    show=True        # 시각화 ON
)

print("\n[필터 적용 결과 통계]")
pprint.pprint(depth_filter_info, sort_dicts=False)


# # **가우시안 필터**

# In[1001]:


def apply_depth_gaussian_filter(
    depth_img,
    ksize=5,
    sigma=0,
    valid_mask=None,
    show=False,
    title="Gaussian Depth Filter"
):
    """
    depth_img에 Gaussian Blur 적용.
    invalid depth=0 영역은 최대한 유지.

    Parameters
    ----------
    depth_img : np.ndarray
        uint16 또는 float depth image
    ksize : int
        Gaussian kernel size. 홀수여야 함.
    sigma : float
        Gaussian sigma. 0이면 OpenCV가 자동 계산.
    valid_mask : np.ndarray
        depth 유효 영역 mask. 1=valid, 0=invalid
    show : bool
        시각화 여부
    """

    if ksize % 2 == 0:
        raise ValueError("ksize는 홀수여야 합니다. 예: 3, 5, 7")

    if valid_mask is None:
        valid_mask = (depth_img > 0).astype(np.uint8)

    depth_float = depth_img.astype(np.float32)

    # Gaussian 적용
    gaussian_float = cv2.GaussianBlur(
        depth_float,
        (ksize, ksize),
        sigmaX=sigma,
        sigmaY=sigma
    )

    # invalid 영역은 원래처럼 0 유지
    gaussian_float[valid_mask == 0] = 0

    # 원본 dtype 유지
    if np.issubdtype(depth_img.dtype, np.integer):
        gaussian_img = np.clip(
            gaussian_float,
            0,
            np.iinfo(depth_img.dtype).max
        ).astype(depth_img.dtype)
    else:
        gaussian_img = gaussian_float.astype(depth_img.dtype)

    info = {
        "filter": "gaussian",
        "ksize": ksize,
        "sigma": sigma,
        "input_dtype": str(depth_img.dtype),
        "output_dtype": str(gaussian_img.dtype),
        "valid_pixels_before": int(np.count_nonzero(depth_img > 0)),
        "valid_pixels_after": int(np.count_nonzero(gaussian_img > 0)),
        "min_valid_depth": float(gaussian_img[gaussian_img > 0].min()) if np.any(gaussian_img > 0) else None,
        "max_valid_depth": float(gaussian_img.max()) if np.any(gaussian_img > 0) else None,
    }

    if show:
        show_depth_compare(
            depth_img,
            gaussian_img,
            title1="Input Depth",
            title2=title
        )

    return gaussian_img, info


# # **CLAHE**

# In[1002]:


def apply_depth_clahe_filter(
    depth_img,
    clip_limit=2.0,
    tile_grid_size=(8, 8),
    valid_mask=None,
    show=False,
    title="CLAHE Depth"
):
    """
    depth image에 CLAHE 적용.

    주의:
    CLAHE는 실제 depth 값을 보존하는 필터라기보다는
    시각화/대비 강화에 가까움.

    포즈 추정용 depth 값으로 직접 쓰기보다는,
    물체 경계 확인용 또는 마스크 보조용으로 보는 것을 추천.
    """

    if valid_mask is None:
        valid_mask = (depth_img > 0).astype(np.uint8)

    depth_float = depth_img.astype(np.float32)

    valid_values = depth_float[valid_mask > 0]

    if valid_values.size == 0:
        clahe_img = np.zeros_like(depth_img)
        info = {
            "filter": "clahe",
            "error": "No valid depth pixels"
        }
        return clahe_img, info

    min_d = valid_values.min()
    max_d = valid_values.max()

    # depth를 0~255로 정규화
    depth_norm = np.zeros_like(depth_float, dtype=np.uint8)

    if max_d > min_d:
        depth_norm[valid_mask > 0] = (
            (depth_float[valid_mask > 0] - min_d) / (max_d - min_d) * 255.0
        ).astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    clahe_norm = clahe.apply(depth_norm)

    # 다시 원래 depth range로 복원
    clahe_float = np.zeros_like(depth_float)
    clahe_float[valid_mask > 0] = (
        clahe_norm[valid_mask > 0].astype(np.float32) / 255.0
    ) * (max_d - min_d) + min_d

    clahe_float[valid_mask == 0] = 0

    if np.issubdtype(depth_img.dtype, np.integer):
        clahe_img = np.clip(
            clahe_float,
            0,
            np.iinfo(depth_img.dtype).max
        ).astype(depth_img.dtype)
    else:
        clahe_img = clahe_float.astype(depth_img.dtype)

    info = {
        "filter": "clahe",
        "clip_limit": clip_limit,
        "tile_grid_size": tile_grid_size,
        "input_dtype": str(depth_img.dtype),
        "output_dtype": str(clahe_img.dtype),
        "valid_pixels_before": int(np.count_nonzero(depth_img > 0)),
        "valid_pixels_after": int(np.count_nonzero(clahe_img > 0)),
        "min_valid_depth": float(clahe_img[clahe_img > 0].min()) if np.any(clahe_img > 0) else None,
        "max_valid_depth": float(clahe_img.max()) if np.any(clahe_img > 0) else None,
        "original_min_valid_depth": float(min_d),
        "original_max_valid_depth": float(max_d),
    }

    if show:
        show_depth_compare(
            depth_img,
            clahe_img,
            title1="Input Depth",
            title2=title
        )

    return clahe_img, info


# In[1003]:


# ------------------------------------------------------------
# 1. Gaussian filter 실행부
# ------------------------------------------------------------

valid_mask_filtered = (depth_filtered > 0).astype(np.uint8)

depth_gaussian, gaussian_info = apply_depth_gaussian_filter(
    depth_img=depth_filtered,
    ksize=5,
    sigma=0,
    valid_mask=valid_mask_filtered,
    show=True,
    title="Gaussian Filtered Depth"
)

print("\n[Gaussian 필터 적용 결과 통계]")
pprint.pprint(gaussian_info, sort_dicts=False)

# ------------------------------------------------------------
# 2. depth_filtered에 Gaussian 결과 덮어쓰기
# ------------------------------------------------------------

depth_filtered = depth_gaussian.copy()

valid_mask_filtered = (depth_filtered > 0).astype(np.uint8)

print("[INFO] depth_filtered가 Gaussian 결과로 갱신되었습니다.")


# In[1004]:


# ------------------------------------------------------------
# 4. CLAHE filter 실행부
# ------------------------------------------------------------

depth_clahe, clahe_info = apply_depth_clahe_filter(
    depth_img=depth_filtered,
    clip_limit=2.0,
    tile_grid_size=(8, 8),
    valid_mask=valid_mask_filtered,
    show=True,
    title="CLAHE Depth"
)

print("\n[CLAHE 필터 적용 결과 통계]")
pprint.pprint(clahe_info, sort_dicts=False)


# # **Local Protrusion Depth Mask**
# 
# 주변 local median depth와 현재 depth를 비교해서, 주변보다 카메라 쪽으로 튀어나온 영역을 검출한다.
# 절대적으로 가까운 물체만 잡는 방식이 아니라, 뒤쪽에 있어도 주변보다 돌출된 알맹이를 잡는 데 사용한다.

# In[1005]:


# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

def create_local_protrusion_depth_mask(
    depth_img,
    valid_mask=None,
    depth_scale=0.001,
    local_kernel=51,
    protrusion_threshold_mm=20.0,
    
    # 🔥 [추가] 허공 방어 로직 파라미터
    max_protrusion_mm=100.0,         # 너무 높게 튀어나온 곳(절벽 모서리 등) 무시
    max_background_depth_mm=1000.0,  # 너무 먼 바닥(허공)에서 발생한 돌출 무시
    
    min_area=50,
    max_area_ratio=0.35,
    close_kernel=7,
    open_kernel=3,
    ignore_border_touching=True,
    border_margin=3,
    show=False,
    title="Local Protrusion Depth Mask"
):
    """
    주변 local median depth보다 카메라 쪽으로 튀어나온 영역을 검출한다.
    (추가: 책상 모서리나 먼 바닥에서 발생하는 가짜 돌출 방지 로직 포함)
    """

    if depth_img is None:
        raise ValueError("depth_img가 None입니다.")
    if not isinstance(depth_img, np.ndarray):
        raise TypeError("depth_img는 np.ndarray 형식이어야 합니다.")
    if depth_img.ndim != 2:
        raise ValueError(f"depth_img는 2D여야 합니다. 현재 shape: {depth_img.shape}")
    if local_kernel % 2 == 0 or local_kernel < 3:
        raise ValueError("local_kernel은 최소 3 이상의 홀수여야 합니다. 예: 31, 51, 71")

    h, w = depth_img.shape
    image_area = h * w
    depth_mm = depth_img.astype(np.float32) * float(depth_scale) * 1000.0

    if valid_mask is None:
        valid_bool = depth_img > 0
    else:
        if valid_mask.shape != depth_img.shape:
            raise ValueError("valid_mask와 depth_img의 shape가 다릅니다.")
        valid_bool = (valid_mask > 0) & (depth_img > 0)

    valid_values = depth_mm[valid_bool]

    if valid_values.size == 0:
        print("[WARN] valid depth가 없어 local protrusion mask를 만들 수 없습니다.")
        return np.zeros_like(depth_img, dtype=np.uint8), {
            "component_count": 0,
            "components": [],
        }

    # invalid 영역은 local median 계산을 망치지 않도록 먼 거리값으로 임시 대체
    far_value = float(np.percentile(valid_values, 95))
    depth_for_median = depth_mm.copy()
    depth_for_median[~valid_bool] = far_value

    try:
        from scipy.ndimage import median_filter
        local_median = median_filter(depth_for_median, size=local_kernel)
        local_stat_method = "scipy.ndimage.median_filter"
    except ImportError:
        print("[WARN] scipy가 없어 cv2.blur로 local mean을 대신 사용합니다.")
        local_median = cv2.blur(depth_for_median, (local_kernel, local_kernel))
        local_stat_method = "cv2.blur_local_mean"

    # ------------------------------------------------------------
    # 🔥 [핵심 방어 로직] 
    # ------------------------------------------------------------
    protrusion_mm = local_median - depth_mm

    # 1. threshold 이상 튀어나올 것
    protrusion_bool = valid_bool & (protrusion_mm >= float(protrusion_threshold_mm))

    # 2. 책상 모서리 방어: 듀플로 블록 최대 높이보다 더 튀어나왔으면 탈락!
    if max_protrusion_mm is not None:
        protrusion_bool &= (protrusion_mm <= float(max_protrusion_mm))

    # 3. 심해 방어: 물체가 깔려있는 바닥(local_median)이 너무 멀면 탈락!
    if max_background_depth_mm is not None:
        protrusion_bool &= (local_median <= float(max_background_depth_mm))

    protrusion_mask = protrusion_bool.astype(np.uint8) * 255

    # morphology 정리
    if close_kernel is not None and close_kernel > 0:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        protrusion_mask = cv2.morphologyEx(protrusion_mask, cv2.MORPH_CLOSE, k_close, iterations=1)

    if open_kernel is not None and open_kernel > 0:
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        protrusion_mask = cv2.morphologyEx(protrusion_mask, cv2.MORPH_OPEN, k_open, iterations=1)

    # component filtering
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(protrusion_mask, connectivity=8)

    cleaned = np.zeros_like(protrusion_mask)
    components = []

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        area_ratio = area / max(image_area, 1)

        if area < min_area or area_ratio > max_area_ratio:
            continue

        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        bw = int(stats[label_id, cv2.CC_STAT_WIDTH])
        bh = int(stats[label_id, cv2.CC_STAT_HEIGHT])

        if ignore_border_touching:
            touches_border = (
                x <= border_margin or y <= border_margin or
                x + bw >= w - border_margin or y + bh >= h - border_margin
            )
            if touches_border:
                continue

        comp_mask = labels == label_id
        comp_depth = depth_mm[comp_mask & valid_bool]
        comp_protrusion = protrusion_mm[comp_mask & valid_bool]

        if comp_depth.size == 0:
            continue

        cx, cy = centroids[label_id]
        cleaned[comp_mask] = 255

        components.append({
            "label_id": int(label_id),
            "area": area,
            "area_ratio": float(area_ratio),
            "bbox": [x, y, bw, bh],
            "centroid": [float(cx), float(cy)],
            "depth_median_mm": float(np.median(comp_depth)),
            "depth_std_mm": float(np.std(comp_depth)),
            "protrusion_median_mm": float(np.median(comp_protrusion)),
            "protrusion_mean_mm": float(np.mean(comp_protrusion)),
        })

    protrusion_mask = cleaned

    print("\n[LOCAL PROTRUSION DEPTH MASK]")
    print(f"local method            : {local_stat_method}")
    print(f"local_kernel            : {local_kernel}")
    print(f"protrusion_threshold_mm : {protrusion_threshold_mm:.2f}")
    print(f"max_protrusion_mm       : {max_protrusion_mm}")
    print(f"max_background_depth_mm : {max_background_depth_mm}")
    print(f"detected components     : {len(components)}")

    for i, comp in enumerate(components):
        print(
            f"  - comp {i}: area={comp['area']}, bbox={comp['bbox']}, "
            f"depth_median={comp['depth_median_mm']:.2f} mm, "
            f"protrusion_median={comp['protrusion_median_mm']:.2f} mm"
        )

    protrusion_info = {
        "local_method": local_stat_method,
        "local_kernel": int(local_kernel),
        "protrusion_threshold_mm": float(protrusion_threshold_mm),
        "max_protrusion_mm": max_protrusion_mm,
        "max_background_depth_mm": max_background_depth_mm,
        "component_count": len(components),
        "components": components,
    }

    if show:
        valid_protrusion = protrusion_mm[valid_bool]
        upper = np.percentile(valid_protrusion, 99) if valid_protrusion.size > 0 else 1.0

        protrusion_vis = protrusion_mm.copy()
        protrusion_vis[~valid_bool] = 0
        protrusion_vis = np.clip(protrusion_vis, 0, upper)

        protrusion_norm = cv2.normalize(protrusion_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        protrusion_color = cv2.applyColorMap(protrusion_norm, cv2.COLORMAP_JET)
        protrusion_color = cv2.cvtColor(protrusion_color, cv2.COLOR_BGR2RGB)

        overlay = protrusion_color.copy()
        red = np.zeros_like(overlay)
        red[:, :, 0] = 255

        mask_bool = protrusion_mask > 0
        overlay[mask_bool] = (0.55 * overlay[mask_bool] + 0.45 * red[mask_bool]).astype(np.uint8)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].imshow(protrusion_color)
        axes[0].set_title("Local Protrusion Map")
        axes[0].axis("off")

        axes[1].imshow(protrusion_mask, cmap="gray")
        axes[1].set_title(title)
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title("Protrusion Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()

    return protrusion_mask, protrusion_info


# In[1006]:


local_protrusion_mask, local_protrusion_info = create_local_protrusion_depth_mask(
    depth_img=depth_filtered,
    valid_mask=valid_mask,
    depth_scale=intrinsics["depth"]["depth_scale"],

    local_kernel=121,
    protrusion_threshold_mm=15.0,

    max_protrusion_mm=70.0,
    max_background_depth_mm=1100.0,

    min_area=150,
    max_area_ratio=0.20,

    close_kernel=15,
    open_kernel=5,

    ignore_border_touching=True,
    border_margin=5,

    show=True
)


# # **Depth Gradient Barrier Mask**
# 
# depth_img에서 값이 급격히 변하는 지점을 찾아 공간상 경계선으로 사용하는 함수다.
# 객체 후보가 배경이나 훅 꺼진 공간으로 번지는 것을 막기 위한 barrier mask로 사용한다.

# In[1007]:


def create_depth_gradient_barrier_mask(
    depth_img,
    valid_mask=None,
    depth_scale=0.001,
    grad_threshold_mm=35.0,
    blur_ksize=3,
    dilate_kernel=3,
    show=False,
    title="Depth Gradient Barrier Mask"
):
    """
    depth gradient가 큰 위치를 barrier mask로 생성한다.

    객체와 배경 사이, 또는 공간상에서 훅 꺾이는 부분은 depth 변화량이 크므로
    이를 region growing이나 candidate mask 확장 제한선으로 사용할 수 있다.

    Parameters
    ----------
    depth_img : np.ndarray
        RealSense aligned depth image.

    valid_mask : np.ndarray or None
        유효 depth mask.
        None이면 depth_img > 0 기준으로 사용한다.

    depth_scale : float
        RealSense depth scale.
        기본 0.001이면 raw depth 1000 = 1000 mm.

    grad_threshold_mm : float
        gradient magnitude threshold.
        낮으면 barrier가 많이 생기고, 높으면 강한 경계만 남는다.

    blur_ksize : int
        gradient 계산 전 depth를 약하게 smoothing할 kernel size.
        0 또는 None이면 blur 생략.

    dilate_kernel : int
        barrier를 두껍게 만들기 위한 dilation kernel size.
        0 또는 None이면 생략.

    show : bool
        True이면 결과를 시각화한다.

    title : str
        시각화 제목.

    Returns
    -------
    barrier_mask : np.ndarray
        depth gradient barrier mask.
        dtype=np.uint8, 값은 0 또는 255.

    barrier_info : dict
        gradient 통계와 threshold 정보.
    """

    if depth_img is None:
        raise ValueError("depth_img가 None입니다.")

    if not isinstance(depth_img, np.ndarray):
        raise TypeError("depth_img는 np.ndarray 형식이어야 합니다.")

    if depth_img.ndim != 2:
        raise ValueError(f"depth_img는 2D여야 합니다. 현재 shape: {depth_img.shape}")

    depth_mm = depth_img.astype(np.float32) * float(depth_scale) * 1000.0

    if valid_mask is None:
        valid_bool = depth_img > 0
    else:
        if valid_mask.shape != depth_img.shape:
            raise ValueError("valid_mask와 depth_img의 shape가 다릅니다.")
        valid_bool = (valid_mask > 0) & (depth_img > 0)

    valid_count = int(np.count_nonzero(valid_bool))

    if valid_count == 0:
        print("[WARN] valid depth가 없어 gradient barrier를 만들 수 없습니다.")
        return np.zeros_like(depth_img, dtype=np.uint8), {
            "grad_threshold_mm": float(grad_threshold_mm),
            "barrier_pixels": 0,
            "valid_count": 0,
        }

    valid_values = depth_mm[valid_bool]
    median_depth = float(np.median(valid_values))

    # invalid 영역은 median으로 임시 대체해서 Sobel 계산 시 큰 튐을 줄임
    depth_for_grad = depth_mm.copy()
    depth_for_grad[~valid_bool] = median_depth

    if blur_ksize is not None and blur_ksize > 0:
        if blur_ksize % 2 == 0:
            raise ValueError("blur_ksize는 홀수여야 합니다. 예: 3, 5, 7")
        depth_for_grad = cv2.GaussianBlur(
            depth_for_grad,
            (blur_ksize, blur_ksize),
            0
        )

    gx = cv2.Sobel(depth_for_grad, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth_for_grad, cv2.CV_32F, 0, 1, ksize=3)

    grad_mag = np.sqrt(gx * gx + gy * gy)

    barrier_bool = valid_bool & (grad_mag >= float(grad_threshold_mm))
    barrier_mask = barrier_bool.astype(np.uint8) * 255

    if dilate_kernel is not None and dilate_kernel > 0:
        k_dilate = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dilate_kernel, dilate_kernel)
        )
        barrier_mask = cv2.dilate(barrier_mask, k_dilate, iterations=1)

    valid_grad = grad_mag[valid_bool]

    grad_p50 = float(np.percentile(valid_grad, 50))
    grad_p90 = float(np.percentile(valid_grad, 90))
    grad_p95 = float(np.percentile(valid_grad, 95))
    grad_p99 = float(np.percentile(valid_grad, 99))

    barrier_pixels = int(np.count_nonzero(barrier_mask))

    print("\n[DEPTH GRADIENT BARRIER MASK]")
    print(f"valid count          : {valid_count}")
    print(f"grad_threshold_mm    : {grad_threshold_mm:.2f}")
    print(f"barrier pixels       : {barrier_pixels}")
    print(f"grad p50/p90/p95/p99 : {grad_p50:.2f} / {grad_p90:.2f} / {grad_p95:.2f} / {grad_p99:.2f}")

    barrier_info = {
        "valid_count": valid_count,
        "grad_threshold_mm": float(grad_threshold_mm),
        "barrier_pixels": barrier_pixels,
        "grad_p50": grad_p50,
        "grad_p90": grad_p90,
        "grad_p95": grad_p95,
        "grad_p99": grad_p99,
        "blur_ksize": blur_ksize,
        "dilate_kernel": dilate_kernel,
    }

    if show:
        grad_clip = np.clip(grad_mag, 0, grad_p99)
        grad_norm = cv2.normalize(
            grad_clip,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        grad_color = cv2.applyColorMap(grad_norm, cv2.COLORMAP_JET)
        grad_color = cv2.cvtColor(grad_color, cv2.COLOR_BGR2RGB)

        overlay = grad_color.copy()
        red = np.zeros_like(overlay)
        red[:, :, 0] = 255

        mask_bool = barrier_mask > 0
        overlay[mask_bool] = (
            0.55 * overlay[mask_bool] + 0.45 * red[mask_bool]
        ).astype(np.uint8)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].imshow(grad_color)
        axes[0].set_title("Depth Gradient Magnitude")
        axes[0].axis("off")

        axes[1].imshow(barrier_mask, cmap="gray")
        axes[1].set_title(title)
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title("Barrier Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()

    return barrier_mask, barrier_info


# In[1008]:


depth_barrier_mask,depth_barrier_info = create_depth_gradient_barrier_mask(
    depth_img=depth_filtered,
    valid_mask=valid_mask,
    depth_scale=intrinsics["depth"]["depth_scale"],

    grad_threshold_mm=10.0,
    blur_ksize=3,
    dilate_kernel=3,

    show=True
)

print(depth_barrier_info)


# # **Depth Gradient Component Mask**
# 
# Depth gradient가 큰 부분을 경계선으로 보고, 그 경계를 제외한 valid depth 영역에서 connected component를 찾는 함수다.
# 고정된 60mm depth 차이에 의존하지 않고, 공간상에서 깊이 변화가 급격한 경계를 기준으로 덩어리 후보를 분리한다

# In[1009]:


def create_depth_gradient_component_mask(
    depth_img,
    valid_mask=None,
    depth_scale=0.001,
    grad_threshold_mode="percentile",
    grad_threshold_mm=35.0,
    grad_percentile=92.0,
    blur_ksize=3,
    sharpen_strength=1.5,  # 🔥 [추가] 샤프닝 강도
    barrier_dilate_kernel=3,
    min_area=40,
    max_area_ratio=0.35,
    max_component_depth_std_mm=120.0,
    ignore_border_touching=True,
    border_margin=3,
    close_kernel=5,
    open_kernel=3,
    show=False,
    title="Depth Gradient Component Mask"
):
    """
    Depth gradient를 barrier로 사용하여 valid depth 영역을 component로 분리한다.

    절대 depth 차이가 몇 mm 이상인지가 아니라,
    depth가 급격히 바뀌는 경계선을 기준으로 영역을 나눈다.

    Parameters
    ----------
    depth_img : np.ndarray
        RealSense aligned depth image.

    valid_mask : np.ndarray or None
        유효 depth mask. None이면 depth_img > 0 기준.

    depth_scale : float
        RealSense depth scale.
        기본 0.001이면 raw depth 1000 = 1000 mm.

    grad_threshold_mode : str
        "percentile" 또는 "absolute".
        percentile이면 gradient 상위 grad_percentile 기준으로 barrier 생성.
        absolute이면 grad_threshold_mm 기준으로 barrier 생성.

    grad_threshold_mm : float
        absolute mode에서 사용할 gradient threshold.

    grad_percentile : float
        percentile mode에서 사용할 gradient percentile.
        예: 92이면 gradient 상위 8%를 barrier로 사용.

    blur_ksize : int
        gradient 계산 전 depth smoothing kernel size.
        0 또는 None이면 blur 생략.

    barrier_dilate_kernel : int
        barrier를 두껍게 만들기 위한 dilation kernel size.
        0 또는 None이면 생략.

    min_area : int
        component 최소 면적.

    max_area_ratio : float
        화면 전체 대비 component 최대 면적 비율.
        너무 큰 영역은 배경/바닥일 가능성이 높으므로 제거.

    max_component_depth_std_mm : float
        component 내부 depth 표준편차 허용값.
        너무 크면 여러 표면/공간이 섞인 영역으로 보고 제거.

    ignore_border_touching : bool
        이미지 경계에 닿은 component를 제거할지 여부.

    border_margin : int
        경계 접촉 판정 여유 픽셀.

    close_kernel : int
        최종 component mask closing kernel size.

    open_kernel : int
        최종 component mask opening kernel size.

    show : bool
        True이면 결과 시각화.

    title : str
        시각화 제목.

    Returns
    -------
    component_mask : np.ndarray
        gradient barrier 기준으로 분리된 depth component mask.
        dtype=np.uint8, 값은 0 또는 255.

    component_info : dict
        gradient threshold, component 정보 등 디버깅 정보.
    """

    if depth_img is None:
        raise ValueError("depth_img가 None입니다.")

    if not isinstance(depth_img, np.ndarray):
        raise TypeError("depth_img는 np.ndarray 형식이어야 합니다.")

    if depth_img.ndim != 2:
        raise ValueError(f"depth_img는 2D여야 합니다. 현재 shape: {depth_img.shape}")

    if grad_threshold_mode not in ["percentile", "absolute"]:
        raise ValueError("grad_threshold_mode는 'percentile' 또는 'absolute'여야 합니다.")

    h, w = depth_img.shape
    image_area = h * w

    depth_mm = depth_img.astype(np.float32) * float(depth_scale) * 1000.0

    if valid_mask is None:
        valid_bool = depth_img > 0
    else:
        if valid_mask.shape != depth_img.shape:
            raise ValueError("valid_mask와 depth_img의 shape가 다릅니다.")
        valid_bool = (valid_mask > 0) & (depth_img > 0)

    valid_count = int(np.count_nonzero(valid_bool))

    if valid_count == 0:
        print("[WARN] valid depth가 없어 gradient component mask를 만들 수 없습니다.")
        return np.zeros_like(depth_img, dtype=np.uint8), {
            "component_count": 0,
            "components": [],
            "valid_count": 0,
        }

    valid_values = depth_mm[valid_bool]
    median_depth = float(np.median(valid_values))

    # invalid 영역은 median으로 대체해서 Sobel 계산 시 invalid 경계가 과도하게 튀는 것을 줄임
    depth_for_grad = depth_mm.copy()
    depth_for_grad[~valid_bool] = median_depth

    if blur_ksize is not None and blur_ksize > 0:
        if blur_ksize % 2 == 0:
            raise ValueError("blur_ksize는 홀수여야 합니다. 예: 3, 5, 7")
        depth_for_grad = cv2.GaussianBlur(
            depth_for_grad,
            (blur_ksize, blur_ksize),
            0
        )

    # depth gradient 계산
    gx = cv2.Sobel(depth_for_grad, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth_for_grad, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)

    valid_grad = grad_mag[valid_bool]

    grad_p50 = float(np.percentile(valid_grad, 50))
    grad_p90 = float(np.percentile(valid_grad, 90))
    grad_p92 = float(np.percentile(valid_grad, 92))
    grad_p95 = float(np.percentile(valid_grad, 95))
    grad_p99 = float(np.percentile(valid_grad, 99))

    if grad_threshold_mode == "percentile":
        grad_threshold_used = float(np.percentile(valid_grad, grad_percentile))
    else:
        grad_threshold_used = float(grad_threshold_mm)

    # depth 변화가 큰 부분을 barrier로 설정
    barrier_bool = valid_bool & (grad_mag >= grad_threshold_used)
    barrier_mask = barrier_bool.astype(np.uint8) * 255

    if barrier_dilate_kernel is not None and barrier_dilate_kernel > 0:
        k_barrier = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (barrier_dilate_kernel, barrier_dilate_kernel)
        )
        barrier_mask = cv2.dilate(barrier_mask, k_barrier, iterations=1)

    # valid 영역에서 barrier를 제외한 영역을 component 후보로 사용
    valid_uint8 = valid_bool.astype(np.uint8) * 255
    non_barrier_mask = cv2.bitwise_and(
        valid_uint8,
        cv2.bitwise_not(barrier_mask)
    )

    # 약간 정리
    if close_kernel is not None and close_kernel > 0:
        k_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (close_kernel, close_kernel)
        )
        non_barrier_mask = cv2.morphologyEx(
            non_barrier_mask,
            cv2.MORPH_CLOSE,
            k_close,
            iterations=1
        )

    if open_kernel is not None and open_kernel > 0:
        k_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (open_kernel, open_kernel)
        )
        non_barrier_mask = cv2.morphologyEx(
            non_barrier_mask,
            cv2.MORPH_OPEN,
            k_open,
            iterations=1
        )

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        non_barrier_mask,
        connectivity=8
    )

    component_mask = np.zeros_like(non_barrier_mask)
    components = []

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        area_ratio = area / max(image_area, 1)

        if area < min_area:
            continue

        if area_ratio > max_area_ratio:
            continue

        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        bw = int(stats[label_id, cv2.CC_STAT_WIDTH])
        bh = int(stats[label_id, cv2.CC_STAT_HEIGHT])

        if ignore_border_touching:
            touches_border = (
                x <= border_margin or
                y <= border_margin or
                x + bw >= w - border_margin or
                y + bh >= h - border_margin
            )

            if touches_border:
                continue

        comp_mask = labels == label_id
        comp_valid = comp_mask & valid_bool
        comp_depth = depth_mm[comp_valid]
        comp_grad = grad_mag[comp_valid]

        if comp_depth.size == 0:
            continue

        comp_depth_std = float(np.std(comp_depth))

        if comp_depth_std > max_component_depth_std_mm:
            continue

        cx, cy = centroids[label_id]

        component_mask[comp_mask] = 255

        components.append({
            "label_id": int(label_id),
            "area": area,
            "area_ratio": float(area_ratio),
            "bbox": [x, y, bw, bh],
            "centroid": [float(cx), float(cy)],
            "depth_median_mm": float(np.median(comp_depth)),
            "depth_mean_mm": float(np.mean(comp_depth)),
            "depth_std_mm": comp_depth_std,
            "depth_min_mm": float(np.min(comp_depth)),
            "depth_max_mm": float(np.max(comp_depth)),
            "grad_mean": float(np.mean(comp_grad)),
            "grad_median": float(np.median(comp_grad)),
        })

    print("\n[DEPTH GRADIENT COMPONENT MASK]")
    print(f"valid count              : {valid_count}")
    print(f"threshold mode           : {grad_threshold_mode}")
    print(f"grad threshold used      : {grad_threshold_used:.2f}")
    print(f"grad p50/p90/p92/p95/p99 : {grad_p50:.2f} / {grad_p90:.2f} / {grad_p92:.2f} / {grad_p95:.2f} / {grad_p99:.2f}")
    print(f"detected components      : {len(components)}")

    for i, comp in enumerate(components):
        print(
            f"  - comp {i}: "
            f"area={comp['area']}, "
            f"bbox={comp['bbox']}, "
            f"depth_median={comp['depth_median_mm']:.2f} mm, "
            f"depth_std={comp['depth_std_mm']:.2f} mm"
        )

    component_info = {
        "valid_count": valid_count,
        "grad_threshold_mode": grad_threshold_mode,
        "grad_threshold_used": grad_threshold_used,
        "grad_percentile": float(grad_percentile),
        "grad_threshold_mm": float(grad_threshold_mm),
        "grad_p50": grad_p50,
        "grad_p90": grad_p90,
        "grad_p92": grad_p92,
        "grad_p95": grad_p95,
        "grad_p99": grad_p99,
        "component_count": len(components),
        "components": components,
    }

    if show:
        # gradient 시각화
        grad_clip = np.clip(grad_mag, 0, grad_p99)
        grad_norm = cv2.normalize(
            grad_clip,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        grad_color = cv2.applyColorMap(grad_norm, cv2.COLORMAP_JET)
        grad_color = cv2.cvtColor(grad_color, cv2.COLOR_BGR2RGB)

        # depth view
        p05 = np.percentile(valid_values, 5)
        p95 = np.percentile(valid_values, 95)
        depth_clip = np.clip(depth_mm, p05, p95)
        depth_norm = ((depth_clip - p05) / max(p95 - p05, 1e-6) * 255).astype(np.uint8)
        depth_norm[~valid_bool] = 0
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)

        # component overlay
        overlay = depth_color.copy()
        red = np.zeros_like(overlay)
        red[:, :, 0] = 255

        comp_bool = component_mask > 0
        overlay[comp_bool] = (
            0.55 * overlay[comp_bool] + 0.45 * red[comp_bool]
        ).astype(np.uint8)

        fig, axes = plt.subplots(1, 4, figsize=(22, 5))

        axes[0].imshow(grad_color)
        axes[0].set_title("Depth Gradient Magnitude")
        axes[0].axis("off")

        axes[1].imshow(barrier_mask, cmap="gray")
        axes[1].set_title("Gradient Barrier")
        axes[1].axis("off")

        axes[2].imshow(component_mask, cmap="gray")
        axes[2].set_title(title)
        axes[2].axis("off")

        axes[3].imshow(overlay)
        axes[3].set_title("Component Overlay")
        axes[3].axis("off")

        plt.tight_layout()
        plt.show()

    return component_mask, component_info


# In[1010]:


depth_gradient_component_mask, depth_gradient_component_info = create_depth_gradient_component_mask(
    depth_img=depth_filtered,
    valid_mask=valid_mask,
    depth_scale=intrinsics["depth"]["depth_scale"],

    grad_threshold_mode="percentile",
    grad_percentile=70.0,

    blur_ksize=3,
    barrier_dilate_kernel=3,

    min_area=40,
    max_area_ratio=0.35,
    max_component_depth_std_mm=100.0,

    ignore_border_touching=True,
    border_margin=3,

    close_kernel=5,
    open_kernel=3,

    show=True
)

print(depth_gradient_component_info)


# In[1011]:


def downsample_depth_image(
    depth_img,
    scale=0.5,
    target_size=None,
    valid_mask=None,
    interpolation="nearest",
    show=False,
    title="Downsampled Depth Image"
):
    """
    depth image를 다운샘플링한다.

    Parameters
    ----------
    depth_img : np.ndarray
        입력 depth image. 보통 uint16 raw depth.
    scale : float
        축소 비율. target_size가 None일 때 사용.
        예: 0.5이면 가로/세로 절반.
    target_size : tuple or None
        (width, height) 형태의 목표 크기.
        지정하면 scale보다 우선.
    valid_mask : np.ndarray or None
        유효 depth mask. 있으면 같이 다운샘플링.
    interpolation : str
        "nearest", "area", "linear" 중 선택.
        depth에서는 보통 "nearest" 또는 "area" 추천.
    show : bool
        시각화 여부.

    Returns
    -------
    depth_down : np.ndarray
        다운샘플링된 depth image.
    valid_down : np.ndarray or None
        다운샘플링된 valid mask.
    info : dict
        다운샘플링 정보.
    """

    if depth_img is None:
        raise ValueError("depth_img가 None입니다.")
    if not isinstance(depth_img, np.ndarray):
        raise TypeError("depth_img는 np.ndarray여야 합니다.")
    if depth_img.ndim != 2:
        raise ValueError(f"depth_img는 2D여야 합니다. 현재 shape: {depth_img.shape}")

    h, w = depth_img.shape

    if target_size is None:
        new_w = int(w * scale)
        new_h = int(h * scale)
        target_size = (new_w, new_h)
    else:
        new_w, new_h = target_size

    interp_map = {
        "nearest": cv2.INTER_NEAREST,
        "area": cv2.INTER_AREA,
        "linear": cv2.INTER_LINEAR,
    }

    if interpolation not in interp_map:
        raise ValueError("interpolation은 'nearest', 'area', 'linear' 중 하나여야 합니다.")

    interp = interp_map[interpolation]

    depth_down = cv2.resize(
        depth_img,
        target_size,
        interpolation=interp
    )

    valid_down = None
    if valid_mask is not None:
        if valid_mask.shape != depth_img.shape:
            raise ValueError("valid_mask와 depth_img의 shape가 다릅니다.")

        valid_down = cv2.resize(
            valid_mask.astype(np.uint8),
            target_size,
            interpolation=cv2.INTER_NEAREST
        )

        valid_down = (valid_down > 0).astype(np.uint8)

        # 다운샘플된 depth에서도 0은 invalid로 처리
        valid_down = valid_down & (depth_down > 0).astype(np.uint8)

    info = {
        "original_shape": [int(h), int(w)],
        "downsampled_shape": [int(depth_down.shape[0]), int(depth_down.shape[1])],
        "scale": float(scale),
        "target_size": [int(new_w), int(new_h)],
        "interpolation": interpolation,
    }

    print("\n[DOWNSAMPLE DEPTH IMAGE]")
    print(f"original shape    : {depth_img.shape}")
    print(f"downsampled shape : {depth_down.shape}")
    print(f"interpolation     : {interpolation}")

    if show:
        depth_vis = depth_down.astype(np.float32)
        if valid_down is not None:
            depth_vis[valid_down == 0] = 0

        valid_pixels = depth_vis[depth_vis > 0]
        if valid_pixels.size > 0:
            vmin = np.percentile(valid_pixels, 1)
            vmax = np.percentile(valid_pixels, 99)
        else:
            vmin, vmax = 0, 1

        plt.figure(figsize=(8, 5))
        plt.imshow(depth_vis, cmap="gray", vmin=vmin, vmax=vmax)
        plt.title(title)
        plt.axis("off")
        plt.colorbar()
        plt.show()

    return depth_down, valid_down, info


# In[1012]:


valid_mask = (depth_filtered > 0).astype(np.uint8)

depth_down, valid_down, down_info = downsample_depth_image(
    depth_img=depth_filtered,
    scale=0.5,
    valid_mask=valid_mask,
    interpolation="nearest",
    show=True,
    title="Downsampled Depth"
)

print(down_info)


# In[1013]:


# from sklearn.cluster import DBSCAN

def downsample_depth_and_mask(depth_img, valid_mask, target_mask, scale=0.5):
    h, w = depth_img.shape
    new_w = int(w * scale)
    new_h = int(h * scale)

    depth_down = cv2.resize(
        depth_img,
        (new_w, new_h),
        interpolation=cv2.INTER_NEAREST
    )

    valid_down = cv2.resize(
        valid_mask.astype(np.uint8),
        (new_w, new_h),
        interpolation=cv2.INTER_NEAREST
    )

    target_down = cv2.resize(
        target_mask.astype(np.uint8),
        (new_w, new_h),
        interpolation=cv2.INTER_NEAREST
    )

    dbscan_valid = (
        (depth_down > 0) &
        (valid_down > 0) &
        (target_down > 0)
    ).astype(np.uint8)

    return depth_down, dbscan_valid


def dbscan_depth_image(
    depth_img,
    valid_mask,
    depth_scale=0.001,
    eps=22.0,
    min_samples=6,
    xy_scale_mm_per_px=3.0,
    min_depth_mm=100.0,
    max_depth_mm=1200.0,
    min_cluster_area=6,
    output_mode="color",
    show=True
):
    h, w = depth_img.shape

    depth_mm = depth_img.astype(np.float32) * depth_scale * 1000.0

    valid = (
        (valid_mask > 0) &
        (depth_img > 0) &
        (depth_mm >= min_depth_mm) &
        (depth_mm <= max_depth_mm)
    )

    ys, xs = np.where(valid)

    if len(xs) == 0:
        print("[WARN] DBSCAN 입력 픽셀이 없습니다.")
        return np.zeros((h, w, 3), dtype=np.uint8), []

    zs = depth_mm[ys, xs]

    features = np.column_stack([
        xs.astype(np.float32) * xy_scale_mm_per_px,
        ys.astype(np.float32) * xy_scale_mm_per_px,
        zs.astype(np.float32),
    ])

    labels = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        n_jobs=-1
    ).fit_predict(features)

    label_img = np.full((h, w), -1, dtype=np.int32)
    label_img[ys, xs] = labels

    result_color = np.zeros((h, w, 3), dtype=np.uint8)
    result_mask = np.zeros((h, w), dtype=np.uint8)

    clusters = []
    rng = np.random.default_rng(42)

    for label in sorted(set(labels)):
        if label == -1:
            continue

        mask = label_img == label
        area = int(np.sum(mask))

        if area < min_cluster_area:
            continue

        color = rng.integers(50, 255, size=3, dtype=np.uint8)
        result_color[mask] = color
        result_mask[mask] = 255

        y_idx, x_idx = np.where(mask)

        clusters.append({
            "label": int(label),
            "area": area,
            "bbox": [
                int(x_idx.min()),
                int(y_idx.min()),
                int(x_idx.max() - x_idx.min() + 1),
                int(y_idx.max() - y_idx.min() + 1),
            ],
            "centroid": [
                float(np.mean(x_idx)),
                float(np.mean(y_idx)),
            ],
            "depth_median_mm": float(np.median(depth_mm[mask])),
        })

    print(f"[DBSCAN] clusters: {len(clusters)}")

    if show:
        plt.figure(figsize=(8, 5))

        if output_mode == "mask":
            plt.imshow(result_mask, cmap="gray")
            plt.title("DBSCAN Mask")
        else:
            plt.imshow(result_color)
            plt.title("DBSCAN Color")

        plt.axis("off")
        plt.show()

    if output_mode == "mask":
        return result_mask, clusters
    else:
        return result_color, clusters


# In[1014]:


depth_scale = intrinsics["depth"]["depth_scale"]

valid_mask = (depth_filtered > 0).astype(np.uint8)

depth_down, dbscan_valid = downsample_depth_and_mask(
    depth_img=depth_filtered,
    valid_mask=valid_mask,
    target_mask=local_protrusion_mask,
    scale=0.5
)

plt.figure(figsize=(8, 5))
plt.imshow(depth_down, cmap="gray")
plt.title("Downsampled Depth")
plt.axis("off")
plt.show()

plt.figure(figsize=(8, 5))
plt.imshow(dbscan_valid, cmap="gray")
plt.title("DBSCAN Input Mask")
plt.axis("off")
plt.show()

dbscan_color, clusters = dbscan_depth_image(
    depth_img=depth_down,
    valid_mask=dbscan_valid,
    depth_scale=depth_scale,

    eps=22.0,
    min_samples=6,
    xy_scale_mm_per_px=3.0,

    min_depth_mm=100.0,
    max_depth_mm=1200.0,

    min_cluster_area=6,
    output_mode="color",
    show=True
)

clusters


# In[1015]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


def create_depth_clahe_mask(
    depth_img,
    valid_mask=None,
    depth_scale=0.001,

    min_depth_mm=100.0,
    max_depth_mm=1200.0,

    median_ksize=5,
    gaussian_ksize=5,

    clahe_clip_limit=2.0,
    clahe_tile_grid_size=8,

    threshold_mode="adaptive",   # "adaptive" or "otsu" or "fixed"
    fixed_threshold=120,

    adaptive_block_size=31,
    adaptive_C=-3,

    open_kernel=3,
    close_kernel=7,

    show=True
):
    if depth_img is None:
        raise ValueError("depth_img가 None입니다.")

    if depth_img.ndim != 2:
        raise ValueError("depth_img는 2D여야 합니다.")

    depth_mm = depth_img.astype(np.float32) * float(depth_scale) * 1000.0

    if valid_mask is None:
        valid = depth_img > 0
    else:
        valid = (valid_mask > 0) & (depth_img > 0)

    valid &= depth_mm >= min_depth_mm
    valid &= depth_mm <= max_depth_mm

    if np.sum(valid) == 0:
        print("[WARN] valid depth가 없습니다.")
        return np.zeros_like(depth_img, dtype=np.uint8), {}

    # valid 영역 기준 정규화
    d_min = np.percentile(depth_mm[valid], 1)
    d_max = np.percentile(depth_mm[valid], 99)

    depth_norm = np.zeros_like(depth_mm, dtype=np.uint8)

    norm_float = (depth_mm - d_min) / max(d_max - d_min, 1e-6)
    norm_float = np.clip(norm_float, 0, 1)

    # 가까운 물체가 밝게 보이도록 반전
    depth_norm[valid] = ((1.0 - norm_float[valid]) * 255).astype(np.uint8)

    # invalid는 0
    depth_norm[~valid] = 0

    # median
    if median_ksize is not None and median_ksize > 1:
        if median_ksize % 2 == 0:
            median_ksize += 1
        depth_med = cv2.medianBlur(depth_norm, median_ksize)
    else:
        depth_med = depth_norm.copy()

    # gaussian
    if gaussian_ksize is not None and gaussian_ksize > 1:
        if gaussian_ksize % 2 == 0:
            gaussian_ksize += 1
        depth_blur = cv2.GaussianBlur(depth_med, (gaussian_ksize, gaussian_ksize), 0)
    else:
        depth_blur = depth_med.copy()

    # CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip_limit,
        tileGridSize=(clahe_tile_grid_size, clahe_tile_grid_size)
    )
    depth_clahe = clahe.apply(depth_blur)

    depth_clahe[~valid] = 0

    # threshold
    if threshold_mode == "adaptive":
        if adaptive_block_size % 2 == 0:
            adaptive_block_size += 1

        mask = cv2.adaptiveThreshold(
            depth_clahe,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            adaptive_block_size,
            adaptive_C
        )

    elif threshold_mode == "otsu":
        _, mask = cv2.threshold(
            depth_clahe,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    elif threshold_mode == "fixed":
        _, mask = cv2.threshold(
            depth_clahe,
            fixed_threshold,
            255,
            cv2.THRESH_BINARY
        )

    else:
        raise ValueError("threshold_mode은 'adaptive', 'otsu', 'fixed' 중 하나여야 합니다.")

    mask[~valid] = 0

    # morphology
    if open_kernel is not None and open_kernel > 0:
        k_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (open_kernel, open_kernel)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

    if close_kernel is not None and close_kernel > 0:
        k_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (close_kernel, close_kernel)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

    info = {
        "d_min_mm": float(d_min),
        "d_max_mm": float(d_max),
        "median_ksize": median_ksize,
        "gaussian_ksize": gaussian_ksize,
        "clahe_clip_limit": clahe_clip_limit,
        "clahe_tile_grid_size": clahe_tile_grid_size,
        "threshold_mode": threshold_mode,
        "mask_pixel_count": int(np.sum(mask > 0)),
    }

    if show:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        axes[0].imshow(depth_norm, cmap="gray")
        axes[0].set_title("Normalized Depth")
        axes[0].axis("off")

        axes[1].imshow(depth_blur, cmap="gray")
        axes[1].set_title("Median + Gaussian")
        axes[1].axis("off")

        axes[2].imshow(depth_clahe, cmap="gray")
        axes[2].set_title("CLAHE Depth")
        axes[2].axis("off")

        axes[3].imshow(mask, cmap="gray")
        axes[3].set_title("Final Mask")
        axes[3].axis("off")

        plt.tight_layout()
        plt.show()

    return mask, info


# In[1016]:


depth_scale = intrinsics["depth"]["depth_scale"]
valid_mask = (depth_filtered > 0).astype(np.uint8)

clahe_mask, clahe_info = create_depth_clahe_mask(
    depth_img=depth_filtered,
    valid_mask=valid_mask,
    depth_scale=depth_scale,

    min_depth_mm=100.0,
    max_depth_mm=1200.0,

    median_ksize=5,
    gaussian_ksize=5,

    clahe_clip_limit=2.0,
    clahe_tile_grid_size=8,

    threshold_mode="adaptive",
    adaptive_block_size=31,
    adaptive_C=-3,

    open_kernel=3,
    close_kernel=7,

    show=True
)

print(clahe_info)


# In[ ]:




