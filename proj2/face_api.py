# STEP 1 : import modules
import numpy as np  
import cv2 
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image 

# STEP 2 : create inference object(instance)
face = FaceAnalysis() 
face.prepare(ctx_id=0, det_size=(640, 640))

from fastapi import FastAPI, File, UploadFile

app = FastAPI()

import cv2
import numpy as np
@app.post("/uploadfile/")
async def create_upload_file(file1: UploadFile, file2: UploadFile):
    # STEP 3: Load the input image.
    contents1 = await file1.read()
    contents2 = await file2.read()
    nparr1 = np.fromstring(contents1, np.uint8)  # 변환
    nparr2 = np.fromstring(contents2, np.uint8)  # 변환
    img1 = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)

    # STEP 4 : inference
    faces1 = face.get(img1)
    faces2 = face.get(img2)
    assert len(faces1)==1 # 샘플 이미지에 얼굴 개수 확인
    assert len(faces2)==1 # 샘플 이미지에 얼굴 개수 확인


    # STEP 5 : post processing
    face_feat1 = faces1[0].normed_embedding # 임베딩의 값의 정규화 : -1 -> 1
    face_feat2 = faces2[0].normed_embedding
    face_feat1 = np.array(face_feat1, dtype=np.float32)
    face_feat2 = np.array(face_feat2, dtype=np.float32)

    sims = np.dot(face_feat1, face_feat2.T) # 코사인 유사도 계산
    print(sims)

    if sims > 0.4:
        return {"result" : "동일인입니다"}
    else:
        return {"result" : "동일인이 아닙니다"}