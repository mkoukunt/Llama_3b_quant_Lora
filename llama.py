from transformers import AutoModelForCausalLM, AutoTokenizer
# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("./Llama3.2-3B")
model = AutoModelForCausalLM.from_pretrained("./Llama3.2-3B",trust_remote_code=True)
# Generate text
input_text = "What is the capital of France?"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(inputs["input_ids"], max_length=50)
# Print the result
print(tokenizer.decode(outputs[0], skip_special_tokens=True))