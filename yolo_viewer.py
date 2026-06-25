import cv2
import numpy as np
import pyrealsense2 as rs  # [NEW] 리얼센스 라이브러리 임포트
from ultralytics import YOLO

def main():
    # ==========================================
    # 1. 두 개의 YOLO 모델 경로 설정 및 로드
    # ==========================================
    # model_path_1 = 'yolo_models/Component_Model_ver1.0/Model_s_ver2.0/best.pt'
    # model_path_2 = 'yolo_models/Model_s_ver3.0/Model_s_ver3.0/best.pt'

    model_path_1 = 'yolo_models/Block_m_ver1.0/Block_s_ver1.0/best.pt'
    model_path_2 = 'yolo_models/Block_m_ver2.0/Block_m_ver2.0/best.pt'

    print(f"[INFO] 모델 1 로드 중: {model_path_1}")
    model1 = YOLO(model_path_1)
    
    print(f"[INFO] 모델 2 로드 중: {model_path_2}")
    model2 = YOLO(model_path_2)

    # ==========================================
    # 2. RealSense D435 카메라 파이프라인 설정 (RGB만)
    # ==========================================
    print("[INFO] RealSense 카메라를 연결합니다...")
    pipeline = rs.pipeline()
    config = rs.config()

    # 해상도 640x480, 프레임 30fps, OpenCV 색상 규격(BGR8)으로 RGB 스트림만 켬
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    try:
        pipeline.start(config)
        print("[INFO] RealSense 카메라 연결 성공!")
    except Exception as e:
        print(f"[ERROR] 카메라를 열 수 없습니다. 연결을 확인하세요: {e}")
        return

    print("[INFO] 실시간 듀얼 추론을 시작합니다. (종료하려면 'q' 입력)")

    # ==========================================
    # 3. 실시간 추론 루프
    # ==========================================
    try:
        while True:
            # 리얼센스에서 프레임 세트 기다리기
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                continue

            # RealSense 프레임 데이터를 넘파이 배열(이미지)로 변환
            frame = np.asanyarray(color_frame.get_data())

            # 모델 1 추론 및 결과 시각화
            results1 = model1(frame, stream=False, verbose=False) 
            annotated_frame1 = results1[0].plot()

            # 모델 2 추론 및 결과 시각화
            results2 = model2(frame, stream=False, verbose=False)
            annotated_frame2 = results2[0].plot()

            # 각 화면 상단에 모델 이름(라벨) 달아주기
            cv2.putText(annotated_frame1, "Model 1: ver2.0", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame1, f"{model_path_1}", (20, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

            cv2.putText(annotated_frame2, "Model 2: ver3.0", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame2, f"{model_path_2}", (20, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

            # 두 화면을 좌우로 나란히 붙이기
            combined_frame = np.hstack((annotated_frame1, annotated_frame2))

            # 화면 출력
            cv2.imshow("Dual YOLO RealSense Viewer", combined_frame)

            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] 뷰어를 종료합니다.")
                break

    finally:
        # 에러가 나거나 q를 눌러서 종료할 때 자원 안전하게 해제
        print("[INFO] 파이프라인을 닫습니다.")
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()