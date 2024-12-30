from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

# TrOCR 모델과 프로세서 로드
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

# 이미지에서 텍스트를 추출하는 함수
def extract_text_from_image(image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values

    # 텍스트 생성
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    
    # 생성된 텍스트 디코딩
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

# 텍스트에서 전력 정보를 추출하는 함수
def extract_power(text):
    # 예시: 텍스트에서 'power: 20W'와 같은 형식으로 전력 값을 추출
    words = text.split()
    for i, word in enumerate(words):
        if word.lower() in ['power', 'watt', 'w']:
            try:
                power_value = float(words[i+1].replace('W', '').replace('w', ''))
                return power_value
            except ValueError:
                return None
    return None

# 두 텍스트를 비교하여 적합/부적합 판별하는 함수
def compare_power(text_1, text_2):
    charger_power = extract_power(text_1)
    device_power = extract_power(text_2)
    
    if charger_power is None or device_power is None:
        return "전력 정보를 찾을 수 없음"
    
    if charger_power == device_power:
        return "적합"
    else:
        return "부적합"

# 이미지 경로 설정
image_path_1 = "charger.jpg"  # 충전기 정보가 담긴 이미지 경로
image_path_2 = "dock.png"   # 기기 정보가 담긴 이미지 경로

# 첫 번째 이미지에서 텍스트 추출
text_1 = extract_text_from_image(image_path_1)
print("text1 : "  + text_1)

# 두 번째 이미지에서 텍스트 추출
text_2 = extract_text_from_image(image_path_2)
print("text2 : "  + text_2)

# 전력 비교 결과 출력
#result = compare_power(text_1, text_2)
#print(f"전력 매칭 결과: {result}")