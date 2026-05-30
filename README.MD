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
