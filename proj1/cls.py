# 이미지를 다운로드 하는 코드
# import urllib.request

IMAGE_FILENAMES = ['burger.jpg', 'cat.jpg']

# for name in IMAGE_FILENAMES:
#   url = f'https://storage.googleapis.com/mediapipe-tasks/image_classifier/{name}'
#   urllib.request.urlretrieve(url, name)



# 이미지를 확인해주는 예제 코드
# import cv2
# # from google.colab.patches import cv2_imshow # 코랩용
# import math

# DESIRED_HEIGHT = 480
# DESIRED_WIDTH = 480

# def resize_and_show(image):
#   h, w = image.shape[:2]
#   if h < w:
#     img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
#   else:
#     img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
# #   cv2_imshow(img)
#   cv2.imshow("test", img)
#   cv2.waitKey(0)


# # Preview the images.

# images = {name: cv2.imread(name) for name in IMAGE_FILENAMES}
# for name, image in images.items():
#   print(name)
#   resize_and_show(image)

# 코드 보단 주석에 집중 - 모든 라이브러리에서 통용되는 과정(STEP)
# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision

# STEP 2: Create an ImageClassifier object.
base_options = python.BaseOptions(model_asset_path='models\\efficientnet_lite0.tflite') # 경로
options = vision.ImageClassifierOptions(base_options=base_options, max_results=4) # 1000가지 중의 max_results 개
classifier = vision.ImageClassifier.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file('burger.jpg')

# STEP 4: Classify the input image.
classification_result = classifier.classify(image) # forward(), inference(), get() 추론 이름

# STEP 5: Process the classification result. In this case, visualize it.
top_category = classification_result.classifications[0].categories[0]
print(f"{top_category.category_name} ({top_category.score:.2f})")
