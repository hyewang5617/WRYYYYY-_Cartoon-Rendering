# Cartoon Rendering using OpenCV

OpenCV와 Python을 이용하여 입력 이미지를 만화(cartoon) 스타일로 변환하는 프로그램입니다.

---

## 📌 개요

본 프로젝트는 컴퓨터비전 수업 과제로, 이미지 처리 기법을 활용하여 일반 이미지를 만화 스타일로 변환하는 것을 목표로 합니다.

강의에서 배운 다양한 이미지 처리 기법을 조합하여 구현하였습니다.

---

## ⚙️ 사용 기술

- Python
- OpenCV
- NumPy

---

## 🧠 알고리즘 설명

본 프로그램은 다음과 같은 단계로 동작합니다:

1. **Grayscale 변환**
   - 컬러 이미지를 흑백 이미지로 변환

2. **Median Blur (노이즈 제거)**
   - 잡음을 제거하면서 경계를 보존

3. **Adaptive Thresholding (윤곽선 추출)**
   - 이미지의 윤곽선을 검출

4. **Bilateral Filter (색상 부드럽게 처리)**
   - 경계를 유지하면서 색상을 부드럽게 만듦

5. **Color Quantization (색상 단순화)**
   - 색상 수를 줄여 만화 느낌 강화

6. **이미지 결합**
   - 윤곽선 + 색상 이미지를 결합하여 최종 결과 생성

---

## 🖼️ 실행 방법

```bash
pip install opencv-python numpy
python main.py input.jpg