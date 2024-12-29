## import argparse
## import sys
## import insightface

# STEP 1 : import modules
import numpy as np  # numpy : 선형대수 라이브러리, 행렬을 처리, image = 비트맵 2차원 행렬
import cv2 # opencv : numpy를 처리
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image # 샘플 이미지

# 불필요한 코드 삭제 : ##
## 버전 체크 삭제
## assert insightface.__version__>='0.3'
## parser = argparse.ArgumentParser(description='insightface app test')
## general
## parser.add_argument('--ctx', default=0, type=int, help='ctx id, <0 means using cpu')
## parser.add_argument('--det-size', default=640, type=int, help='detection size')
## args = parser.parse_args()

# STEP 2 : create inference object(instance)
app = FaceAnalysis() # 모델을 자동으로 다운 받아 준다
# 위의 내용과 일치시킨다
# ctx_id : Context ID, 기본적으로 CPU나 GPU를 지정할 때 사용, 만약 -1로 설정하면 GPU를 사용하지 않고 CPU에서 작업을 처리
# det_size : Detection Size, 입력 이미지 또는 데이터를 처리할 때의 크기를 의미
app.prepare(ctx_id=0, det_size=(640, 640))

# STEP 3 : load data
img1 = cv2.imread("iu01.jpg")
img2 = cv2.imread("iu02.jpg")


# STEP 4 : inference
faces1 = app.get(img1)
faces2 = app.get(img2)
assert len(faces1)==1 # 샘플 이미지에 얼굴 개수 확인
assert len(faces2)==1 # 샘플 이미지에 얼굴 개수 확인


# STEP 5 : post processing
face_feat1 = faces1[0].normed_embedding # 임베딩의 값의 정규화 : -1 -> 1
face_feat2 = faces2[0].normed_embedding
face_feat1 = np.array(face_feat1, dtype=np.float32)
face_feat2 = np.array(face_feat2, dtype=np.float32)

sims = np.dot(face_feat1, face_feat2.T) # 코사인 유사도 계산
print(sims) # sims 값 : 0 ~ 0.4 다른사람, 0.4 ~ 같은사람의 유사, 1 같은 사진

# STEP 3
## img = ins_get_image('t1')

# STEP 4
## faces = app.get(img) # 이때 임베딩 결과가 들어온다

# STEP 5-1 : Save result image
# rimg = app.draw_on(img, faces)
# cv2.imwrite("./t1_output.jpg", rimg)

# STEP 5-2 : face recognition
# then print all-to-all face similarity
## feats = []
## for face in faces:
##    feats.append(face.normed_embedding)
## feats = np.array(feats, dtype=np.float32)