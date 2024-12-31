import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import easyocr
from PIL import Image, ImageEnhance, ImageOps
import re
import logging
from pathlib import Path
import numpy as np
import cv2

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ocr_debug.log')
    ]
)
logger = logging.getLogger(__name__)

class TextExtractor:
    def __init__(self):
        logger.info("TextExtractor 초기화 중...")
        try:
            # 한글+영어 리더와 중국어+영어 리더 각각 생성
            self.kr_reader = easyocr.Reader(['en', 'ko'])
            self.cn_reader = easyocr.Reader(['en', 'ch_sim'])
            logger.info("OCR 모델 로드 완료")
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            raise

    def extract_text(self, image_path):
        image_path = Path(image_path)
        logger.info(f"이미지 처리 중: {image_path}")
        
        if not image_path.exists():
            logger.error(f"이미지를 찾을 수 없음: {image_path}")
            return ""
            
        try:
            # 한글+영어 OCR 처리
            logger.info("한글+영어 OCR 처리 시작")
            kr_results = self.kr_reader.readtext(str(image_path))
            kr_text = ' '.join([result[1] for result in kr_results])
            logger.info(f"한글+영어 인식 결과: {kr_text}")
            
            # 중국어+영어 OCR 처리
            logger.info("중국어+영어 OCR 처리 시작")
            cn_results = self.cn_reader.readtext(str(image_path))
            cn_text = ' '.join([result[1] for result in cn_results])
            logger.info(f"중국어+영어 인식 결과: {cn_text}")
            
            # 두 결과 합치기
            combined_text = f"{kr_text} {cn_text}"
            logger.info(f"최종 결합된 텍스트: {combined_text}")
            
            # 결과를 파일로 저장
            debug_dir = Path("debug_images")
            debug_dir.mkdir(exist_ok=True)
            base_name = image_path.stem
            
            with open(debug_dir / f"{base_name}_ocr_result.txt", "w", encoding="utf-8") as f:
                f.write("=== OCR 결과 ===\n")
                f.write(f"한글+영어 인식 결과: {kr_text}\n")
                f.write(f"중국어+영어 인식 결과: {cn_text}\n")
                f.write(f"최종 결합된 텍스트: {combined_text}")
            
            return combined_text
            
        except Exception as e:
            logger.error(f"이미지 처리 실패 {image_path}: {e}")
            logger.exception("상세 에러:")
            return ""

def extract_voltage_current(text):
    logger.info("전압과 전류 추출 중")
    logger.debug(f"분석할 텍스트: {text}")
    
    voltage_patterns = [
        r'(DC).*?(\d+\.?\d*)\s*V',
        r'(\d+\.?\d*)\s*V\s*[\u23D8=]',
        r'Input:\s*(\d+\.?\d*)\s*V',
        r'Output:\s*(\d+\.?\d*)\s*V',
        r'(\d+\.?\d*)\s*V\s*[-~]\s*',
        r'(\d+\.?\d*)\s*[Vv]',
        r'.*?(\d+\.?\d*)\s*V'
    ]

    current_patterns = [
        r'(DC).*?(\d+\.?\d*)\s*(mA|A)',
        r'(\d+\.?\d*)\s*(mA|A)\s*[\u23D8=]',
        r'[\u23D8=]\s*(\d+\.?\d*)\s*(mA|A)',
        r'Input:.*?(\d+\.?\d*)\s*(mA|A)',
        r'Output:.*?(\d+\.?\d*)\s*(mA|A)',
        r'(\d+\.?\d*)\s*(mA|A)',
        r'.*?(\d+\.?\d*)\s*(mA|A)'
    ]

    try:
        # DC 값이 포함된 패턴만 우선적으로 검색
        logger.debug("DC 값 우선 탐색 시작")

        # 전압 추출
        voltage = None
        for pattern in voltage_patterns:
            voltage_match = re.search(pattern, text, re.IGNORECASE)
            if voltage_match and "DC" in voltage_match.group(0):
                voltage = float(voltage_match.group(2))
                logger.debug(f"매칭된 전압 패턴: {pattern}")
                logger.debug(f"추출된 전압값: {voltage}V")
                break

        # 전류 추출
        current = None
        for pattern in current_patterns:
            current_match = re.search(pattern, text, re.IGNORECASE)
            if current_match and "DC" in current_match.group(0):
                current_value = float(current_match.group(2))
                unit = current_match.group(3).lower()
                current = current_value / 1000 if unit == 'ma' else current_value
                logger.debug(f"매칭된 전류 패턴: {pattern}")
                logger.debug(f"추출된 전류값: {current}A")
                break

        logger.info(f"추출된 값 - 전압: {voltage}V, 전류: {current}A")
        return voltage, current

    except Exception as e:
        logger.error(f"값 추출 중 오류 발생: {e}")
        logger.error(f"에러 발생 텍스트: {text}")
        return None, None


def check_compatibility(charger_v, charger_a, device_v, device_a):
   if None in (charger_v, charger_a, device_v, device_a):
       return "값 추출 실패"
   
   voltage_compatible = abs(charger_v - device_v) < 0.1  # 0.1V 오차 허용
   current_compatible = charger_a >= device_a
   
   if voltage_compatible and current_compatible:
       return "적합"
   elif not voltage_compatible:
       return "부적합 (전압 불일치)"
   else:
       return "부적합 (전류 부족)"

def preprocess_image(image_path, output_path):
    """
    텍스트 추출 성능을 높이기 위한 간단한 이미지 전처리 함수.
    """

    # 1. 이미지 불러오기
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 이미지 전체 밝기와 표준편차 측정
    meanVal, stdVal = cv2.meanStdDev(gray)
    meanVal = meanVal[0][0]
    if meanVal > 128:
        # 흰색 글자를 검정색으로 만들기 위해 색 반전
        meanVal = cv2.bitwise_not(gray)
    stdVal = stdVal[0][0]

    # 3. 알파(대비), 베타(밝기) 동적 설정
    #    - 아래는 임의의 기준 예시
    if meanVal < 80:
        beta = 50
    elif meanVal > 180:
        beta = 10
    else:
        beta = 30

    if stdVal < 50:
        alpha = 1.6
    elif stdVal > 100:
        alpha = 1.2
    else:
        alpha = 1.3

    # 4. 밝기·대비 조정
    adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    # 5. 샤프닝 커널 적용
    kernel_sharpening = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ])
    sharpened = cv2.filter2D(adjusted, -1, kernel_sharpening)

    # 6. 결과 저장
    cv2.imwrite(output_path, sharpened)



def main():
   # 디버그 디렉토리 생성
   debug_dir = Path("debug_images")
   debug_dir.mkdir(exist_ok=True)
   
   try:
       extractor = TextExtractor()
       
       # 이미지 경로 설정
       preprocess_image("charger.jpg", "charger2.jpg")
       preprocess_image("ccc.png", "ccc2.png")
       charger_path = Path("charger2.jpg")
       device_path = Path("ccc2.png")


       # 텍스트 추출
       logger.info("충전기 이미지 처리 중...")
       charger_text = extractor.extract_text(charger_path)
       logger.info(f"충전기 텍스트: {charger_text}")
       
       logger.info("기기 이미지 처리 중...")
       device_text = extractor.extract_text(device_path)
       logger.info(f"기기 텍스트: {device_text}")
       
       # 전압/전류 값 추출
       charger_v, charger_a = extract_voltage_current(charger_text)
       device_v, device_a = extract_voltage_current(device_text)
       
       # 호환성 검사
       result = check_compatibility(charger_v, charger_a, device_v, device_a)
       
       # 결과 출력
       logger.info("\n=== 분석 결과 ===")
       logger.info(f"충전기 원본 텍스트: {charger_text}")
       logger.info(f"기기 원본 텍스트: {device_text}")
       logger.info(f"충전기: {charger_v}V, {charger_a}A")
       logger.info(f"기기: {device_v}V, {device_a}A")
       logger.info(f"호환성: {result}")
       
       # 결과를 파일로 저장
       with open(debug_dir / "final_result.txt", "w", encoding="utf-8") as f:
           f.write("=== 분석 결과 ===\n")
           f.write(f"충전기 원본 텍스트: {charger_text}\n")
           f.write(f"기기 원본 텍스트: {device_text}\n")
           f.write(f"충전기: {charger_v}V, {charger_a}A\n")
           f.write(f"기기: {device_v}V, {device_a}A\n")
           f.write(f"호환성: {result}\n")
       
   except Exception as e:
       logger.error(f"프로그램 실행 중 오류 발생: {e}")
       logger.exception("상세 에러:")

if __name__ == "__main__":
   main()