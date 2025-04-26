from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

prompt = "Hello, how are you?"

output = generator(prompt, max_length=50, num_return_sequences=5)

print(output[0]["generated_text"])


