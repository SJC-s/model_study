from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the model and tokenizer
model_name = "beomi/korean-dialogue-generation"  # This is a hypothetical model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set up a conversation
conversation = "사용자: 안녕하세요, 오늘 날씨가 어때요?\n시스템: 안녕하세요! 오늘은 맑고 화창한 날씨입니다. 외출하기 좋은 날이에요.\n사용자: 그렇군요. 점심 메뉴 추천해주세요.\n시스템:"

# Tokenize the input
input_ids = tokenizer.encode(conversation, return_tensors="pt")

# Generate a response
output = model.generate(input_ids, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2)

# Decode and print the response
response = tokenizer.decode(output[0], skip_special_tokens=True)
print(response)
