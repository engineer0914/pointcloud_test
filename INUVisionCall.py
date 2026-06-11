import os
import yaml
import pprint
import glob
import torch
import cv2
import open3d as o3d
from sklearn.cluster import DBSCAN
from ultralytics import YOLO
import pyrealsense2 as rs
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import numpy as np
import copy
from scipy.spatial.transform import Rotation as R

import INUVisionLib as ivl

def search_wide(color_rgb, depth, intrinsics, scale, V_visualize=True):

    if color_rgb is None or depth is None or intrinsics is None or scale is None:
        raise RuntimeError("RealSense 캡처 실패: color/depth/intrinsics/scale 중 None이 있습니다.")

    color_img_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)

    # ### **YOLO V8 세그멘테이션 기준 영역 잡기**
    # color_rgb, depth, intrinsics, scale

    # 욜로 검출
    model = YOLO("yolo_models/duplo_2_low_2/weights/best.pt")
    # model = YOLO("yolo_models/manip_segmentor_0528.pt")
    target_classes = [0, 1, 3, 4, 5, 6, 8, 9]

    results, mask_binary, vis_yolo = ivl.detect_objects_yolo(
        model= model, 
        color_img_bgr=color_img_bgr, 
        target_classes=target_classes, 
        visualize=V_visualize
    )

    # results (list): YOLO 모델의 원본 추론 결과 객체 리스트
    # mask_binary (ndarray): 검출된 모든 객체의 마스크를 하나로 합친 이진 마스크 (0 or 1, 형태: H x W)
    # vis_yolo (ndarray): 바운딩 박스와 라벨이 그려진 시각화용 이미지 (BGR)



    # ### 바운딩 박스가 겹치는 부분을 억제하는 전처리 추가

    # 겹치는 마스크 깔끔하게 정리 (해상도 640x480 기준, 70% 겹치면 삭제, 시각화 켬)
    final_objects, clean_mask = ivl.filter_overlapping_masks(
        results=results, 
        overlap_threshold=0.70, 
        img_shape=(640, 480), 
        visualize=V_visualize
    )

    # final_detected_objects (list): 억제 후 살아남은 최종 객체들의 리스트. 
    #                                 각 요소는 dict 형태 (class_id, class_name, confidence, mask)
    # final_combined_mask (ndarray): 병합된 최종 전체 ROI 마스크 (0 or 1, uint8)



    # print(f"\n✅ 최종 검출된 유효 객체/군집 수: {len(final_objects)}개")
    # final_combined_mask = np.zeros((480, 640), dtype=np.uint8)
    # for obj in final_objects:
    #     print(f" - Name: {obj['class_name']}")
    #     final_combined_mask = np.logical_or(final_combined_mask, obj["mask"]).astype(np.uint8)



    # ### Ransac 바닥 검출 후 높이 + 비율 기반 ID 수정

    # 1. 이전 단계에서 얻은 데이터: 
    # color_rgb, depth, intrinsics, scale
    # final_objects, clean_mask

    # =================================================================
    # [STEP 1 & 2 통합] DBSCAN+RANSAC 바닥 추정 및 40mm 돌출 맵 사영
    # =================================================================
    # 기존의 바닥 다림질과 돌출 맵 추출 과정이 하나의 함수로 처리됩니다.
    # 반환된 closed_mask가 곧 40mm 이상 돌출된 객체의 2D 마스크(mask_40mm_2d)입니다.
    mask_40mm_2d, refined_color, contours, plane_model = ivl.extract_3d_protruding_objects(
        depth_img=depth, 
        color_img_bgr=color_img_bgr, 
        intrinsics=intrinsics, 
        depth_scale=scale, 
        yolo_combined_mask=clean_mask,
        depth_trunc=5.0,
        height_threshold=0.040,
        visualize=False
    )

    # print(f"\n✅ 최종 검출된 유효 객체/군집 수: {len(final_objects)}개")
    # final_combined_mask = np.zeros((480, 640), dtype=np.uint8)
    # for obj in final_objects:
    #     print(f" - Name: {obj['class_name']}")
    #     final_combined_mask = np.logical_or(final_combined_mask, obj["mask"]).astype(np.uint8)

    # =================================================================
    # [STEP 3] 최종 OBB 기반 ID 판독 및 교정
    # =================================================================

    final_objects_before = copy.deepcopy(final_objects)

    final_objects, result_vis_img = ivl.correct_object_ids(
        detected_objects=final_objects, 
        mask_high_2d=mask_40mm_2d, 
        color_img_bgr=color_img_bgr, 
        ratio_threshold=1.5, 
        overlap_threshold=0.20, 
        visualize=False
    )

    # final_objects_after = copy.deepcopy(final_objects)

    # =================================================================
    # [STEP 3-1] ID 교정 후 마스크 내부 채우기
    # =================================================================

    # objects_filled = copy.deepcopy(final_objects_after)

    # mask_before = np.zeros(color_img_bgr.shape[:2], dtype=np.uint8)
    # mask_after = np.zeros(color_img_bgr.shape[:2], dtype=np.uint8)

    # for obj in objects_filled:
    #     original_mask = (obj["mask"] > 0).astype(np.uint8)

    #     filled_mask = ivl.fill_object_mask_holes(original_mask)

    #     obj["mask"] = filled_mask.astype(bool)

    #     mask_before = np.logical_or(mask_before, original_mask > 0).astype(np.uint8)
    #     mask_after = np.logical_or(mask_after, filled_mask > 0).astype(np.uint8)


    # =================================================================
    # [STEP 3-1] ID 교정 후 마스크 Convex Hull로 내부 채우기
    # =================================================================

    objects_hull = copy.deepcopy(final_objects)

    mask_before = np.zeros(color_img_bgr.shape[:2], dtype=np.uint8)
    mask_hull_after = np.zeros(color_img_bgr.shape[:2], dtype=np.uint8)

    for obj in objects_hull:
        original_mask = (obj["mask"] > 0).astype(np.uint8)

        # Convex Hull로 객체 영역 재생성
        hull_mask = ivl.fill_object_mask_by_convex_hull(
            original_mask,
            min_area=20
        )

        # 객체별 mask를 hull 결과로 교체
        obj["mask"] = hull_mask.astype(bool)

        # 전체 before / after 통합 마스크
        mask_before = np.logical_or(mask_before, original_mask > 0).astype(np.uint8)
        mask_hull_after = np.logical_or(mask_hull_after, hull_mask > 0).astype(np.uint8)


    # =================================================================
    # [STEP 4] Convex Hull + 마스크 기준으로 바닥/PCD 재생성
    # =================================================================

    pcd_data, plane_data, floor_pcd = ivl.build_floor_scene_data_from_depth(
        depth_img=depth,
        intrinsics=intrinsics,
        depth_scale=scale,
        object_mask_01=mask_hull_after,
        depth_trunc=5.0,
        voxel_size=0.003,
        plane_dist_thresh=0.015,
        floor_height_eps=0.005,
        visualize=False
    )


    # =================================================================
    # [STEP 5] Convex Hull 객체 기준 3D OBB 생성
    # =================================================================

    objects_obb, vis_3d, overlay_3d, vis_2d_rgb, obb_results = ivl.generate_3d_obbs_from_hull_objects(
        objects=objects_hull,
        refined_mask_01=mask_hull_after,
        pcd_data=pcd_data,
        plane_data=plane_data,
        intrinsics=intrinsics,
        color_img_rgb=color_rgb,
        floor_pcd=floor_pcd,
        min_height=0.024,
        max_height_limit=0.12,
        height_percentile=95,
        visualize_2d=False
    )

    # =================================================================
    # [STEP] 3D OBB 기준 객체 좌표계 + Camera 기준 RPY 계산
    # =================================================================

    pose_results = []
    axes_geometries = []

    plane_normal = plane_data["normal"]

    for idx, obj in enumerate(objects_obb):
        obb_3d = obj.get("obb_3d", None)
        class_name = obj.get("class_name", "unknown")

        pose = ivl.estimate_pose_axes_from_obb3d(
            obb_3d=obb_3d,
            plane_normal=plane_normal,
            class_name=class_name,
            axis_size=0.04
        )

        if pose is None:
            print(f"[SKIP] idx {idx}: pose 계산 실패")
            continue

        obj["pose_cam"] = pose
        pose_results.append({
            "idx": idx,
            "class_name": class_name,
            "center_mm": pose["center_mm"],
            "roll_deg": pose["roll_deg"],
            "pitch_deg": pose["pitch_deg"],
            "yaw_deg": pose["yaw_deg"],
            "R_obj_cam": pose["R_obj_cam"],
        })

        axes_geometries.append(pose["axes_3d"])

        c = pose["center_mm"]

        # print(
        #     f"idx {idx:02d} | {class_name:20s} | "
        #     f"center(mm)=({c[0]:7.1f}, {c[1]:7.1f}, {c[2]:7.1f}) | "
        #     f"RPY(deg)=({pose['roll_deg']:7.2f}, "
        #     f"{pose['pitch_deg']:7.2f}, "
        #     f"{pose['yaw_deg']:7.2f})"
        # )

        # 기존 3D OBB geometry에 좌표축 추가
    vis_3d_with_axes = vis_3d + axes_geometries
    overlay_3d_with_axes = overlay_3d + axes_geometries

    if V_visualize:
        print("\n[INFO] 3D OBB + Object Coordinate Axes 표시")
        o3d.visualization.draw_geometries(
            vis_3d_with_axes,
            window_name="3D OBB + Object XYZ Axes"
        )

    color_o3d = o3d.geometry.Image(color_rgb)
    depth_o3d = o3d.geometry.Image(cv2.medianBlur(depth, 5))

    o3d_intr = o3d.camera.PinholeCameraIntrinsic(
        int(intrinsics.width),
        int(intrinsics.height),
        float(intrinsics.fx),
        float(intrinsics.fy),
        float(intrinsics.ppx),
        float(intrinsics.ppy)
    )

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=1.0 / float(scale),
        depth_trunc=5.0,
        convert_rgb_to_intensity=False
    )

    rgb_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d_intr
    )

    rgb_pcd = rgb_pcd.voxel_down_sample(voxel_size=0.0015)

    final_overlay_elements = [rgb_pcd] + overlay_3d_with_axes

    if V_visualize:
        print("\n[INFO] RGB-D PointCloud + 3D OBB + Object Axes 표시")
        o3d.visualization.draw_geometries(
            final_overlay_elements,
            window_name="RGB-D PointCloud + Object XYZ Axes"
        )


    pose_table, class_index = ivl.build_class_sorted_pose_index(
        objects_obb=objects_obb,
        use_pose_cam=True,
        remove_c_prefix=True,
        remove_side2=False,
        verbose=True
    )

    return pose_table, class_index

class VisionManager:
    def __init__(self):
        # 1. 상태(데이터) 보관함 초기화
        self.color_rgb = None
        self.depth = None
        self.intrinsics = None
        self.scale = None
        
        self.pose_table = None
        self.class_index = None

        # 2. ID -> 클래스 이름 매핑 딕셔너리
        self.id_to_class = {
            1: "2x2_red", 2: "2x2_green", 3: "2x2_blue", 4: "2x2_yellow",
            5: "4x2_red", 6: "4x2_green", 7: "4x2_blue", 8: "4x2_yellow",
            13: "Magnet",
            34: "Battery",
            81: "estop",
            241: "traffic light",
            442: "carrot",
            462: "small tree",
            711: "hammer",
            4482: "big carrot",
            8518: "burger",
            46262: "bigtree",
            48132: "icecream"
        }

    # ==========================================
    # 함수 1. 카메라 호출 함수
    # ==========================================
    def capture_camera(self, mode="mid_50", visualize=True):
        print("[INFO] 카메라 데이터 캡처 중...")
        devices = ivl.get_realsense_ids()
        
        if len(devices) == 0:
            raise RuntimeError("연결된 RealSense 카메라가 없습니다.")
            
        target_serial = list(devices.keys())[0]

        # 캡처한 데이터를 클래스 내부 보관함(self)에 저장
        self.color_rgb, self.depth, self.intrinsics, self.scale = ivl.capture_realsense_data(
            serial_number=target_serial, 
            mode=mode, 
            visualize=visualize
        )
        return self.color_rgb, self.depth, self.intrinsics, self.scale

    # ==========================================
    # 함수 2. 서치 함수
    # ==========================================
    def run_search(self, visualize=False):
        print("[INFO] 전체 객체 탐색(Search Wide) 실행 중...")
        if self.color_rgb is None:
            raise RuntimeError("카메라 데이터가 없습니다. 먼저 capture_camera()를 실행하세요.")

        # 보관함에 있던 카메라 데이터를 꺼내서 서치 함수에 넣음
        self.pose_table, self.class_index = search_wide(
            self.color_rgb, self.depth, self.intrinsics, self.scale, V_visualize=visualize
        )
        return self.pose_table, self.class_index

    # ==========================================
    # 함수 3. 서치 결과 기반 위치 반환 함수 (ID 변환 포함)
    # ==========================================
    def get_pose_by_id(self, target_id, local_id=0):
        if self.class_index is None:
            raise RuntimeError("탐색된 인덱스가 없습니다. 먼저 run_search()를 실행하세요.")

        # 1. 입력받은 ID를 문자열 클래스 이름으로 변환
        target_class_name = self.id_to_class.get(target_id)
        
        if target_class_name is None:
            print(f"[ERROR] 등록되지 않은 ID 번호입니다: {target_id}")
            return None

        print(f"\n[INFO] 타겟 ID [{target_id}] ➔ 클래스명 ['{target_class_name}'] 변환 완료")

        # 2. 변환된 이름으로 6D 포즈 추출
        pose = ivl.get_nearest_6d_pose_by_class(
            class_index=self.class_index,
            target_class_name=target_class_name,
            local_id=local_id
        )

        # 3. 결과 출력 및 반환
        if pose is not None:
            x, y, z = pose["x_mm"], pose["y_mm"], pose["z_mm"]
            roll, pitch, yaw = pose["roll_deg"], pose["pitch_deg"], pose["yaw_deg"]

            print("--- 6D Pose Result ---")
            print(f"class: {pose['class_name']}")
            print(f"local_id: {pose['local_id']}")
            print(f"global_idx: {pose.get('global_idx', 'N/A')}")
            print(f"XYZ mm: {x:.1f}, {y:.1f}, {z:.1f}")
            print(f"RPY deg: {roll:.2f}, {pitch:.2f}, {yaw:.2f}")
            print("----------------------")
            return pose
        else:
            print(f"[WARNING] 시야에서 '{target_class_name}' 객체를 찾을 수 없습니다.")
            return None



# ==========================================
# 4. 단독 실행용 테스트 코드 (if __name__ == "__main__":)
# ==========================================
if __name__ == "__main__":
    print("\n[INFO] ivc.py 라이브러리 단독 테스트 모드 실행\n")
    
    # ---------------------------------------------------------
    # 테스트 방법 1: 클래스를 이용한 깔끔한 테스트
    # ---------------------------------------------------------
    vision = VisionManager()
    
    try:
        vision.capture_camera(visualize=False)
        vision.run_search(visualize=False)
        
        # 4x2_blue (ID 7) 찾기 테스트
        test_pose = vision.get_pose_by_id(target_id=7, local_id=0)
        
        if test_pose:
            print("클래스를 이용한 포즈 추출 성공!")
            
    except Exception as e:
        print(f"[ERROR] 테스트 중 오류 발생: {e}")