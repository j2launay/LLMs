from transformers import AutoModelForCausalLM, AutoTokenizer

print("TEST")

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
model_id = "mistralai/Mistral-7B-Instruct-v0.1"
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_id)
print("ici")
model = AutoModelForCausalLM.from_pretrained(model_id)
print("TE")
text = "Hello my name is Julien Delaunay, I am a phd researcher in explainable AI. I have a girlfriend named Pauline Roose "
print(text)
inputs = tokenizer(text, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))