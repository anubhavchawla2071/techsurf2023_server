import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the pre-trained GPT-2 large model and tokenizer
model_name = "gpt2-large"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

max_length=100
def generate_gpt2_large(heading):
    # Tokenize the heading
    input_ids = tokenizer.encode(heading, return_tensors="pt")

    # Generate text
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=2, pad_token_id=tokenizer.eos_token_id,num_beams=6, no_repeat_ngram_size=2,temperature=1.2)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Post-process to end at the last complete sentence
    last_period_idx = generated_text.rfind('.')
    if last_period_idx != -1:
        generated_text = generated_text[:last_period_idx + 1]

    return generated_text
