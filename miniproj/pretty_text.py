import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from pydub import AudioSegment
import os


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]

# 파일 이름 패턴 정의
file_prefix = "SJ_W_4B_01_"
file_extension = ".mp3"
file_range = range(1, 5)  # 1부터 4까지 처리


# 반복문으로 파일 처리
for i in file_range:
    file_name = f"{file_prefix}{i}{file_extension}"
    print(f"Processing file: {file_name}")
    
    try:
        # 오디오 파일 처리
        result = pipe(file_name, return_timestamps=True)
        transcribed_text = result["text"]
             
        print(transcribed_text)
    
    except Exception as e:
        print(f"Error processing file {file_name}: {e}")
