# INU vision 테스트 구조

### INU 비전에 대한 검증용 주피터 노트북 형식
### 여기에 기반한 라이브러리 변환 및 함수화
### 이후 ROS화에 사용할 로컬 함수 저장 구조

1차 목표: 검출된 “덩어리 객체”마다 가능한 모든 중간 결과와 6D 후보를 출력한다.
2차 목표: 나중에 옵션 플래그로 필요한 출력만 켜고 끈다.
3차 목표: 지금은 정확도보다 “디버깅 가능성”을 최우선으로 둔다.

---

내가 추천하는 최종 구체화 파이프라인

Stage 0. Camera
카메라 리스트업
RGB/depth stream 확인
aligned depth 확보
intrinsic 로드
depth scale 확인
Stage 1. Depth seed
depth valid range filtering
median filter
temporal median optional
invalid mask 생성
depth discontinuity 계산
point cloud 변환
outlier removal
DBSCAN 또는 local depth cluster
2D projection mask 생성

출력:

M_depth_seed
depth_clusters
depth_confidence
Stage 2. RANSAC availability
객체 후보 주변 ring depth 분석
valid_ratio
depth_std
plane_residual
inlier_ratio

출력:

ransac_confidence
table_plane if available
Stage 3. Image contour
RGB ROI 생성
gray/V/L 채널 전처리
CLAHE 약하게
Gaussian or bilateral
Canny
morphology
contour 추출

출력:

M_gray_edge
contour_candidates
Stage 4. Color mask
HSV 변환
Saturation threshold
색상별 hue mask
reflection mask 분리
morphology
color component labeling

출력:

M_hsv
color_components
dominant_colors
Stage 5. Fusion
M_seed = M_depth_seed AND M_hsv
M_candidate = M_hsv OR dilate(M_depth_seed)
barrier = strong RGB edge OR strong depth edge
M_final = reconstruction / region growing

출력:

final object mask
final contour
object ROI
Stage 6. Object classification
색상 cluster 개수
YOLO box 개수
height layer 개수
solidity
rectangularity
edge density
projection complexity

출력:

single_score
assembly_score
unknown_score
Stage 7. Single brick pose
mask 내부 point cloud crop
outlier removal
normal estimation
region/plane normal 추출
orthogonal projection
contour/Hough/MPP/minAreaRect yaw
z_axis = table/top normal
x_axis = projected yaw
y_axis = z × x

출력:

T_camera_object
pose_confidence
Stage 8. Assembly pose
전체 cluster 중심
orthogonal projection footprint
PCA/minAreaRect/Hough 대표 yaw
height histogram
color/YOLO component 분석

출력:

assembly_center
representative_yaw
assembly_pose
assembly_confidence
component_info

---



1. create_valid_depth_mask
2. apply_depth_median_filter
3. create_near_object_seed_mask

4. convert_to_hsv
5. create_duplo_hsv_mask

6. clean_binary_mask
7. create_fusion_seed_mask

8. create_fusion_candidate_mask
9. refine_candidate_mask
10. extract_object_components
11. analyze_object_basic