from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

model = GPT2LMHeadModel.from_pretrained("withU/kogpt2-emotion-chatbot")
tokenizer = PreTrainedTokenizerFast.from_pretrained("withU/kogpt2-emotion-chatbot")

input_ids = tokenizer.encode("배고파", add_special_tokens=False, return_tensors="pt")
output_sequences = model.generate(input_ids=input_ids, do_sample=True, max_length=80, num_return_sequences=4)
for generated_sequence in output_sequences:
    generated_sequence = generated_sequence.tolist()
    print("GENERATED SEQUENCE : {0}".format(tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)))
