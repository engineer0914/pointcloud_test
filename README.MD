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

# 1. 입력 단계: RealSense 필터

D435i/f에서 depth를 가져올 때 먼저 SDK 필터를 적용.

```text
Raw Depth
→ Spatial Filter
→ Temporal Filter
→ Align to Color
→ depth_img
```

역할:

```text
Spatial  : depth 표면 노이즈 완화
Temporal : 프레임 간 흔들림 완화
Align    : color / depth 좌표 일치
```

단, Temporal은 움직이는 물체에서는 잔상이 생길 수 있으니까 너무 강하게 쓰지는 않는 쪽.

---

# 2. 공통 depth 전처리

모든 실제 depth 기반 알고리즘의 공통 입력을 만든다.

```text
depth_img
→ Median
→ Gaussian
→ Valid Mask 생성
→ Invalid Erosion
→ depth_real
```

추천 의미:

```text
Median
- 튀는 점 제거
- salt-pepper depth 노이즈 제거

Gaussian
- 표면 흔들림 완화
- gradient/protrusion 안정화

Invalid erosion
- invalid depth 주변의 가짜 경계 제거
```

여기서 중요한 변수:

```python
depth_real = median + gaussian까지만 적용된 실제 depth
```

이건 실제 거리값을 유지해야 함.

즉, `normalize`, `CLAHE`, `adaptive threshold`가 적용된 이미지는 **마스크용**이지, 3D 좌표 계산용이 아님.

---

# 3. 마스크 Branch A: CLAHE + Adaptive Threshold

가장 기본적인 2D depth contrast 기반 마스크.

```text
depth_real
→ Normalize to uint8
→ CLAHE
→ Adaptive Threshold
→ Morphology
→ clahe_mask
```

역할:

```text
물체 후보를 넓게 잡는 기본 마스크
```

특징:

```text
장점:
- 시각적으로 잘 보임
- 구현 쉬움
- 빠름

단점:
- 실제 3D 의미는 약함
- 절벽/먼 배경/기울어진 바닥에 영향을 받을 수 있음
```

그래서 단독 최종 마스크보다는 **후보 마스크**로 쓰는 게 좋음.

---

# 4. 마스크 Branch B: 기존 Depth Gradient Component

기존에 네가 만든 방식.

```text
depth_real
→ Sobel/Scharr Gradient
→ Gradient Barrier
→ Barrier 제거 후 Connected Component
→ gradient_component_mask
```

역할:

```text
depth가 급격히 변하는 경계를 장벽으로 보고
valid depth 영역을 덩어리 단위로 나누기
```

특징:

```text
장점:
- 객체/바닥 경계, 단차, 절벽 경계 검출 가능
- component 단위로 면적, depth_std 필터링 가능

단점:
- raw depth z값 기준이라 카메라가 삐딱하면 바닥도 gradient를 먹음
```

그래서 이건 유지하되, 나중에는 plane-height 기반 gradient와 비교해서 안 좋으면 뺄 수 있음.

---

# 5. 마스크 Branch C: Local Protrusion

주변보다 카메라 쪽으로 튀어나온 영역을 잡는 방식.

```text
depth_real
→ local median / local background
→ local_background - depth
→ protrusion_mask
```

역할:

```text
주변보다 돌출된 부분 검출
```

특징:

```text
장점:
- 바닥 위 물체 후보 잡기 좋음
- 단순하고 직관적

단점:
- raw depth 기준이면 삐딱한 카메라 뷰에서 local background가 흔들림
- 절벽 모서리도 돌출처럼 잡힐 수 있음
```

그래서 이것도 나중에는 **plane-relative height 기반 protrusion**으로 바꾸는 게 더 좋음.

---

# 6. 마스크 Branch D: RANSAC + DBSCAN

3D point cloud 기반 덩어리 추출.

```text
depth_real
→ Point Cloud 변환
→ RANSAC Plane 추정
→ Plane 제거
→ DBSCAN Clustering
→ Cluster를 다시 2D mask로 projection
→ cluster_mask
```

역할:

```text
바닥 평면 제거 후 실제 3D 공간에서 떨어진 덩어리만 잡기
```

특징:

```text
장점:
- 3D 기준이라 실제 객체 덩어리 검출에 강함
- 바닥 제거에 효과적
- 2D에서는 붙어 보여도 3D에서 떨어진 물체 분리 가능

단점:
- RANSAC이 실패하면 전체가 흔들림
- DBSCAN eps 튜닝 필요
- 다운샘플링이 강하면 작은 구조가 사라질 수 있음
```

그래서 이 branch는 강력하지만 **신뢰도 체크 후 사용하는 보조 branch**로 두는 게 좋다.

---

# 7. 핵심 신규 Branch: Plane Coordinate Height Grid

지금 우리가 가장 중요하게 잡은 방향.

```text
depth_real
→ RANSAC Plane 추정
→ Plane Sanity Check
→ Plane Coordinate Frame 생성
→ 3D point를 u-v-h 좌표로 변환
→ Height Grid 생성
```

여기서 의미는:

```text
u, v : 바닥 평면 위 좌표
h    : 바닥 평면 normal 방향 높이
```

즉 기존 depth z값이 아니라:

```text
바닥 기준으로 얼마나 위로 솟았는가?
```

를 보는 구조.

---

## 이 Branch에서 파생되는 마스크들

```text
height_grid
├─ height_positive_mask
├─ height_gradient_magnitude_mask
├─ height_gradient_component_mask
├─ height_local_protrusion_mask
├─ height_CLAHE_adaptive_mask
└─ closed_region_mask
```

이게 앞으로 메인 방향이 될 가능성이 높음.

---

# 8. Height Positive Mask

가장 단순하고 강력한 기준.

```text
height_grid > 일정 높이
→ 바닥보다 위로 솟은 물체 후보
```

예:

```text
height > 4 mm
height < 150 mm
```

역할:

```text
바닥 평면보다 위에 있는 것만 남김
```

이건 raw depth보다 훨씬 논리적임.

---

# 9. Height Gradient Magnitude

기존 depth gradient를 대체할 수 있는 방식.

```text
height_grid
→ Scharr / Sobel
→ height_gradient_magnitude
```

의미:

```text
바닥 기준 높이가 급격히 바뀌는 부분
```

장점:

```text
삐딱한 카메라 뷰에서도 바닥 기울기 영향을 줄임
물체 경계, 단차, 스터드, 절벽 경계가 더 논리적으로 잡힘
```

기존 depth gradient보다 이쪽이 더 좋을 가능성이 큼.

---

# 10. Height Local Protrusion

기존 protrusion을 height 기준으로 바꾼 것.

```text
height_grid
→ local median height
→ height - local_background
→ height_protrusion_mask
```

의미:

```text
주변보다 바닥 normal 방향으로 더 솟은 부분
```

기존 depth protrusion은:

```text
주변보다 카메라에 가까운가?
```

였다면, 새 방식은:

```text
주변보다 바닥 위로 솟았는가?
```

가 됨.

이게 네 목적에는 더 맞음.

---

# 11. Closed Region Filter

절벽/낭떠러지/이미지 밖으로 이어지는 열린 영역을 제거하기 위한 후처리.

```text
mask
→ contour / connected component
→ border touching 제거
→ solidity / extent 검사
→ closed-like region만 유지
```

역할:

```text
닫힌 물체 영역만 남기기
열린 절벽/배경/이미지 경계 연결 영역 제거
```

이건 CLAHE mask, gradient component mask, cluster mask, height mask 어디에도 적용 가능.

---

# 12. RANSAC 신뢰도 체크

이게 중요함.

RANSAC은 항상 잘 되지 않는다. 그래서 반드시 검사해야 함.

```text
RANSAC plane 추정
→ inlier ratio 검사
→ inlier bbox 크기 검사
→ normal 방향 검사
→ 이전 프레임 normal과 비교 가능하면 비교
```

판단:

```text
신뢰 가능:
→ plane height branch 사용

불안정:
→ height branch 비활성화
→ 기존 depth 기반 mask들로 fallback
```

즉 RANSAC은 “무조건 쓰는 기준”이 아니라:

```text
성공하면 강력하게 쓰고,
실패하면 버리는 branch
```

로 둬야 함.

---

# 13. 최종 조합 방식

처음에는 모든 마스크를 따로 생성해서 비교.

```text
clahe_mask
gradient_component_mask
protrusion_mask
cluster_mask
plane_height_mask
height_protrusion_mask
height_gradient_mask
closed_region_mask
```

그다음 조합.

## 넓은 후보 확인

```text
OR 조합
```

```python
candidate_mask = mask_a OR mask_b OR mask_c
```

의미:

```text
어느 branch라도 잡은 영역을 모두 보기
```

## 확실한 영역 확인

```text
AND 조합
```

```python
final_mask = mask_a AND mask_b AND mask_c
```

의미:

```text
여러 기준을 동시에 만족하는 확실한 영역만 남기기
```

처음에는 이걸 추천:

```text
1단계:
각 mask 개별 시각화

2단계:
OR mask로 후보 범위 확인

3단계:
AND mask로 확실한 영역 확인

4단계:
잘못 잡는 branch는 제거
```

---

# 최종 추천 워크플로우

```text
[Input]
D435i/f depth
→ spatial filter
→ temporal filter
→ align to color

[Common Preprocess]
→ median
→ gaussian
→ invalid erosion
→ depth_real

[Basic 2D Mask]
depth_real
→ normalize
→ CLAHE
→ adaptive threshold
→ clahe_mask

[Depth Shape Masks]
depth_real
→ depth gradient component
→ depth_gradient_mask

depth_real
→ local protrusion
→ depth_protrusion_mask

[3D Cluster Mask]
depth_real
→ RANSAC plane
→ plane remove
→ DBSCAN
→ cluster_mask

[Plane Height Branch]
depth_real
→ RANSAC plane
→ plane reliability check
→ plane coordinate frame
→ height_grid

height_grid
→ positive height mask
→ height gradient mask
→ height protrusion mask
→ height component mask
→ project back to image
→ plane_height_mask

[Closed Region]
각 mask 또는 combined mask
→ border-touch 제거
→ closed-like region 유지

[Combination]
개별 mask 시각화
→ OR 비교
→ AND 비교
→ 필요 없는 branch 제거

[Final]
final_mask를 depth_real에 적용
→ depth_masked
→ point cloud / pose estimation / DBSCAN / PCA / OBB
```

---

# 지금 방향의 핵심 요약

```text
1. CLAHE/adaptive는 후보를 잘 보이게 만드는 2D branch

2. Gradient/protrusion은 형상 기반 branch

3. RANSAC/DBSCAN은 3D 덩어리 기반 branch

4. Plane coordinate height grid는 삐딱선 뷰 대응용 핵심 branch

5. RANSAC은 무조건 믿지 말고 reliability check 후 사용

6. 최종 판단은 여러 mask를 만들고 시각화한 뒤 선택/조합
```

내 판단으로는 앞으로 메인은 이거야.

```text
depth_real 기반 기존 마스크들 유지
+
plane-relative height grid 기반 마스크 추가
+
RANSAC 실패 시 fallback 구조
```

이렇게 가면 지금 네가 걱정하는 **삐딱선 뷰, 절벽, 먼 배경, 바닥 기울기, 정규화 범위 문제**를 가장 체계적으로 다룰 수 있어.



# icp route

------------------------------------
PIC SET
------------------------------------

YOLO V8 ROI 기반 영역 제한

PASS THROUGH FILTER

AND 연산 및 ROI 마스크 추출

뎁스맵 영역 제한

메디안 필터링

포인트 클라우드화

ROI 기반 크롭

아웃 라이어 제거

DBSCAN 클러스터링

------------------------------------
COARSE FITTING
------------------------------------

복셀 다운 샘플링

1차 위치 조정
컨벡스헐 + PCA

색상 구분
PCA 장축 추출
3등분, 각구간별 포인트 클라우드들의 RGB 색조 비율에 따라 로컬 x축 180도 회전 여부 지정

2차 위치 조정
fpfh 특징점 추출 + RANSAC 기반 정렬

------------------------------------
FINE FITTING
------------------------------------

KDTREE 인덱싱 구축

Trimmed CAD 거리 제한

point cloud ICP
=icp시 cad를 뎁스에 맞추기


--------------------------------------------



# YOLO = CVAT TOOL



### 🚀 최종 6D 포즈 추정 파이프라인 (Code-Level Blueprint)

#### 🛡️ [ STEP 1: PIC SET ] (2D-3D 센서 퓨전 및 전처리)

2D 상태에서 지저분한 데이터를 싹 다 날려버려, 3D 연산 부하를 최소화하는 단계입니다.

1. **YOLOv8-Seg 추론:** RGB 이미지에서 듀플로 조립체의 2D 바이너리 마스크 추출
2. **Depth Z축 임계값 마스크:** 바닥(테이블) 높이 이상으로 솟아오른 영역만 1로 마스킹
3. **⭐ 교차 검증 (AND 연산):** `YOLO 마스크 & Depth 마스크`로 허공의 로봇 팔과 바닥의 그림자를 동시 제거
4. **뎁스맵 영역 제한:** 원본 뎁스 이미지에 최종 마스크를 곱해(Element-wise) 배경 데이터를 수학적 `0`으로 소거
5. **메디안 필터링 (2D):** 마스킹된 뎁스맵의 소금-후추 픽셀 노이즈 제거
6. **포인트 클라우드화 (3D 변환):** 카메라 Intrinsics를 적용해 순도 100% 3D 데이터로 변환 (이미 배경이 다 날아갔으므로 PassThrough 필터는 생략 가능!)
7. **아웃라이어 제거:** 허공에 뜬 미세 3D 노이즈 제거
8. **DBSCAN 클러스터링:** 최종적으로 가장 크고 깨끗한 단일 타겟 덩어리 확보 (`Target_Fine`으로 저장)

#### 🧭 [ STEP 2: COARSE FITTING ] (대략적 위치 및 방향 정렬)

무거운 3D 우주 공간에서 CAD 모델을 타겟 근처로 훅! 끌어다 놓는 단계입니다.

1. **복셀 다운 샘플링:** 연산 속도를 위해 타겟을 듬성듬성하게 압축 (`Target_Coarse`)
2. **1차 위치 조정 (PCA + 컨벡스 헐):** 전체적인 덩어리의 중심과 주축(방향)을 대략적으로 매칭
3. **⭐ 색상 기반 대칭 보정:** PCA 장축을 3등분하여 RGB 비율 확인 $\rightarrow$ 역순일 경우 로컬 $X$축 기준 180도 회전 적용 (위아래 완벽 정렬)
4. **안전한 이동:** CAD 모델이 뎁스 껍데기 아래로 푹 꺼지지 않도록 Z축 최상단(또는 바운딩 박스) 기준으로 끌어올림
5. **2차 위치 조정 (FPFH + RANSAC):** 듀플로 특유의 모서리/굴곡 지문을 추출해 밀리미터 단위로 초기 뼈대 강제 결합

#### 🎯 [ STEP 3: FINE FITTING ] (0.1mm 단위 초정밀 밀착)

어긋난 표면을 톱니바퀴처럼 완벽하게 맞물려 잠그는(Lock-in) 단계입니다.

1. **고해상도 타겟 로드:** 아까 STEP 1에서 저장해둔 빽빽한 `Target_Fine`과 CAD 모델을 준비
2. **KD-Tree 인덱싱 자동 구축:** 초고속 이웃 탐색을 위한 공간 분할
3. **⭐ 거리 제한 설정 (Trimmed):** 반쪽짜리 카메라 뷰의 한계를 극복하기 위해, 안 보이는 CAD 뒷면 점들을 매칭에서 강제 제외
4. **Point-to-Plane ICP 진행:** CAD의 면(Plane)을 뎁스 표면에 완벽하게 밀착 슬라이딩시켜 최종 $4 \times 4$ 변환 행렬($T$) 도출

#### 🤖 [ STEP 4: OUTPUT ] (로봇 제어용 출력)

1. **90도 스냅 (Snap) 보정:** $T$ 행렬의 회전각을 로봇이 잡기 가장 편한 90도 정방향 단위로 반올림하여 미세 회전 유격 제거
2. **ROS2 TF 브로드캐스팅:** 최종 오일러 각(Roll, Pitch, Yaw)과 XYZ 좌표를 로봇 팔(매니퓰레이터) 제어 노드로 전송

---
