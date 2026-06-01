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

# 욜로 확인용
def yolo_check(model_path, image_path):
    """
    YOLO 모델과 이미지 경로를 받아 추론을 수행하고,
    결과가 그려진 RGB 이미지를 반환합니다.
    """
    # 1. 모델 로드
    print(f"🤖 '{model_path}' 로딩 중...")
    model = YOLO(model_path)

    # 2. GPU 할당 확인
    model.to('cuda')
    print(f"🚀 모델이 사용하는 하드웨어: {model.device}")

    # 3. 테스트 이미지 추론 (save=False로 변경하여 디스크 저장 생략, 메모리에서 바로 처리)
    print(f"🔍 추론 시작 (RTX 2060 가동)...")
    results = model.predict(source=image_path, save=False, conf=0.5, device=0)

    # 4. 결과 요약 출력
    result = results[0] # 첫 번째 이미지에 대한 결과 객체
    print(f"\n✅ 발견된 객체 수: {len(result.boxes)}")
    if result.masks is not None:
        print(f"🎨 추출된 마스크(폴리곤) 수: {len(result.masks)}")

    # 5. 결과 이미지 배열 뽑아내기 및 색상 변환
    # result.plot() 은 예측 박스와 마스크가 그려진 OpenCV BGR 배열을 바로 뱉어냅니다!
    annotated_bgr = result.plot() 
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB) # BGR -> RGB 변환

    return annotated_rgb

#동적 컬러화
def depth_dynamic_colorization(depth_img):

    # Valid depth
    valid = depth_img > 0

    percentile_min=1
    percentile_max=99

    if np.count_nonzero(valid) == 0:
        raise ValueError("valid depth pixel이 없습니다.")

    d_min = np.percentile(depth_img[valid], percentile_min)
    d_max = np.percentile(depth_img[valid], percentile_max)

    if d_max <= d_min:
        raise ValueError(f"depth range가 비정상입니다. d_min={d_min}, d_max={d_max}")

    # Dynamic normalization
    depth_float = depth_img.astype(np.float32)
    depth_clipped = np.clip(depth_float, d_min, d_max)

    depth_norm = np.zeros_like(depth_img, dtype=np.uint8)
    depth_norm[valid] = (
        (depth_clipped[valid] - d_min) / (d_max - d_min) * 255
    ).astype(np.uint8)

    # Invalid는 검정색
    depth_color_bgr = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    depth_color_bgr[~valid] = (0, 0, 0)

    depth_color_rgb = cv2.cvtColor(depth_color_bgr, cv2.COLOR_BGR2RGB)

    # f"DEPTH Dynamic Colormap\nrange: {d_min:.1f} ~ {d_max:.1f}"
    d_range = [float(d_min), float(d_max)]

    return depth_color_rgb, d_range

# 시각화
def VISUALIZE_COLOR_AND_DEPTH(color_img_bgr, depth_img):

    # BGR -> RGB 변환 필요
    vis_color_img = cv2.cvtColor(color_img_bgr, cv2.COLOR_BGR2RGB)

    # vis_depth_img, depth_range = depth_dynamic_colorization(depth_img)

    # 뎁스 시각화
    vis_depth_img = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_img, alpha=0.025),
        cv2.COLORMAP_JET
    )

    # 비율에 맞춰서 이미지 비율 계산
    width = 18
    height = width * (3 / 8)
    figsize=(width, height)

    # 1행 2열 figsize에 맞춰서 설정
    fig, axes = plt.subplots(1, 2, figsize = figsize)

    # 컬러
    axes[0].imshow(vis_color_img)
    axes[0].set_title("VISUALIZED COLOR (BGR -> RGB)")
    axes[0].axis("off")

    # 뎁스
    axes[1].imshow(vis_depth_img)
    # axes[1].set_title(
    #     f"RAW Dynamic VISUALIZED DEPTH\n"
    #     f"Range: {depth_range[0]:.1f} ~ {depth_range[1]:.1f} [mm]"
    # )
    axes[1].set_title(f"VISUALIZED DEPTH (RAW)")
    axes[1].axis("off")

    # 시각화
    plt.tight_layout()
    plt.show()







### 시각화 대비용 함수
def compare_depth(color_filtered, depth_filtered_before, depth_filtered_after):
    # 글로벌 변수 - 카메라 원본 이미지
        # color_img_bgr
        # depth_img

    # 글로벌 변수 - 처리 완료된 각각의 컬러와 뎁스 저장
        # color_filtered
        # depth_filtered

    # 비율에 맞춰서 이미지 비율 계산
    width = 18
    height = width * (3 / 8)
    figsize = (width, height)

    # 1행 3열 figsize에 맞춰서 설정
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 컬러
    vis_color_img = cv2.cvtColor(color_filtered, cv2.COLOR_BGR2RGB)
    axes[0].imshow(vis_color_img)
    axes[0].set_title("VISUALIZED COLOR (BGR -> RGB)")
    axes[0].axis("off")

    # 뎁스 비포
    vis_depth_img, depth_range = depth_dynamic_colorization(depth_filtered_before)
    axes[1].imshow(vis_depth_img)
    axes[1].set_title(
        f"Dynamic VISUALIZED DEPTH [before]\n"
        f"Range: {depth_range[0]:.1f} ~ {depth_range[1]:.1f} [mm]"
    )
    axes[1].axis("off")

    # 뎁스 애프터
    vis_depth_img, depth_range = depth_dynamic_colorization(depth_filtered_after)
    axes[2].imshow(vis_depth_img)
    axes[2].set_title(
        f"Dynamic VISUALIZED DEPTH [after]\n"
        f"Range: {depth_range[0]:.1f} ~ {depth_range[1]:.1f} [mm]"
    )
    axes[2].axis("off")

    # 시각화
    plt.tight_layout()
    plt.show()


def compare_color(color_filtered_before, color_filtered_after, depth_filtered):
    # 글로벌 변수 - 카메라 원본 이미지
        # color_img_bgr
        # depth_img

    # 글로벌 변수 - 처리 완료된 각각의 컬러와 뎁스 저장
        # color_filtered
        # depth_filtered

    # 비율에 맞춰서 이미지 비율 계산
    width = 18
    height = width * (3 / 8)
    figsize = (width, height)

    # 1행 3열 figsize에 맞춰서 설정
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 컬러 비포
    vis_color_img = cv2.cvtColor(color_filtered_before, cv2.COLOR_BGR2RGB)
    axes[0].imshow(vis_color_img)
    axes[0].set_title("VISUALIZED COLOR [before] (BGR -> RGB)")
    axes[0].axis("off")

    # 컬러 애프터
    vis_color_img = cv2.cvtColor(color_filtered_after, cv2.COLOR_BGR2RGB)
    axes[1].imshow(vis_color_img)
    axes[1].set_title("VISUALIZED COLOR [after] (BGR -> RGB)")
    axes[1].axis("off")

    # 뎁스
    vis_depth_img, depth_range = depth_dynamic_colorization(depth_filtered)
    axes[2].imshow(vis_depth_img)
    axes[2].set_title(
        f"Dynamic VISUALIZED DEPTH\n"
        f"Range: {depth_range[0]:.1f} ~ {depth_range[1]:.1f} [mm]"
    )
    axes[2].axis("off")

    # 시각화
    plt.tight_layout()
    plt.show()



    