# INU vision 테스트 구조

### INU 비전에 대한 검증용 주피터 노트북 형식
### 여기에 기반한 라이브러리 변환 및 함수화
### 이후 ROS화에 사용할 로컬 함수 저장 구조

1차 목표: 검출된 “덩어리 객체”마다 가능한 모든 중간 결과와 6D 후보를 출력한다.
2차 목표: 나중에 옵션 플래그로 필요한 출력만 켜고 끈다.
3차 목표: 지금은 정확도보다 “디버깅 가능성”을 최우선으로 둔다.

---
좋아. 지금 워크플로우 방향은 **“한 방에 완벽한 마스크 하나 만들기”가 아니라, 여러 관점의 마스크를 병렬로 만들고 비교/조합하는 구조**로 가는 게 맞아.

전체 방향은 이렇게 정리하면 돼.

---

# 전체 워크플로우 방향

```text
D435i/f depth 획득
→ RealSense SDK 필터
→ OpenCV depth 전처리
→ 공통 depth_real 생성
→ 여러 마스크 branch 병렬 생성
→ 마스크 비교
→ 선택적으로 AND / OR / 제외
→ 최종 depth mask 생성
→ 실제 3D 계산은 depth_real에서 수행
```

---






# 관련 내용

### **PCL 노말 벡터 추정**

https://pcl.readthedocs.io/projects/tutorials/en/latest/normal_estimation.html#normal-estimation

### **FPFH 설명**

https://velog.io/@harms/Point-Registration-PFH-FPFH-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98

### **CVAT 욜로 툴**

https://github.com/cvat-ai/cvat

### **O3D 라이브러리 도큐먼트**

https://www.open3d.org/docs/release/

### **OPENCV 라이브러리 도큐먼트**

https://docs.opencv.org/4.x/






```
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

# 시각화 -------------------------------------------
def VISUALIZE_COLOR_AND_DEPTH(color_img_bgr, depth_img):

    # BGR -> RGB 변환 필요
    vis_color_img = cv2.cvtColor(color_img_bgr, cv2.COLOR_BGR2RGB)

    vis_depth_img, depth_range = depth_dynamic_colorization(depth_img)

    # # 뎁스 시각화
    # vis_depth_img = cv2.applyColorMap(
    #     cv2.convertScaleAbs(depth_img, alpha=0.025),
    #     cv2.COLORMAP_JET
    # )

    # 설정 -------------------------------------------

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
    axes[1].set_title(f"Dynamic VISUALIZED DEPTH (ColorMap Applied)\nRange: {depth_range[0]:.1f} ~ {depth_range[1]:.1f}")
    axes[1].axis("off")

    # 시각화
    plt.tight_layout()
    plt.show()
```

# **진행 단계**


## **단일 브릭**
1. 윗면 기준 4D
2. 뒤집힘 여부 확인
3. 옆면 정보 포함한 6D

## **조립체**


