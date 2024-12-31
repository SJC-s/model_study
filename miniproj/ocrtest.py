# STEP 1 : import module
import easyocr
import cv2

# STEP 2 : create inference object
# 기본적으로 detector 들어있음(기본값 중국어 ch_sim)
reader = easyocr.Reader(['ko','en']) # this needs to run only once to load the model into memory

# STEP 3 : load data
data = 'ccc.png'

# # 이미지 불러오기
# img = cv2.imread(data)

# # 대비 증가와 밝기 조정
# alpha = 1.3  # 대비
# beta = 50    # 밝기
# adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

# # 이미지를 그레이스케일로 변환
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # 이진화 적용
# _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)


# # 결과 저장
# cv2.imwrite('adjusted_image.jpg', binary)


# STEP 4 : inference
result = reader.readtext(data, detail=0)
print(result)

# STEP 5 : post processing(후처리)