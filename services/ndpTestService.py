

import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from flask import Flask, request, jsonify
SYSTEM_PROMPT = """
You are an helpful assistant understands the user question and provides the correct tool name to call along with  the arguments to pass to the tool.
"""
app = Flask(__name__)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", # Or 'fp4'
    bnb_4bit_compute_dtype=torch.float16, # Or torch.bfloat16
    bnb_4bit_use_double_quant=True
)
peft_model_id = "../Laala-3.2-3B-ndp-inst-api"
config = PeftConfig.from_pretrained(peft_model_id)
print(config.base_model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,
                                            return_dict = True,
                                            quantization_config=bnb_config,
                                            device_map = 'auto')
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

model = PeftModel.from_pretrained(model,peft_model_id)
model.to("cuda")
model.eval()

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.get_json()
    print(data)
    question = data.get('question')
    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    chat = tokenizer.apply_chat_template(chat, tokenize=False)
    inputs = tokenizer(chat, return_tensors="pt", return_attention_mask=True)
    inputs.to("cuda")
    config = GenerationConfig(do_sample=True, temperature=0.1)
    outputs = model.generate(**inputs, max_length=1000, pad_token_id=tokenizer.eos_token_id, generation_config=config)
    text = tokenizer.batch_decode(outputs)[0]
    print(text)
    startIdx = text.index("<|start_header_id|>assistant<|end_header_id|>") + len(
        "<|start_header_id|>assistant<|end_header_id|>") + 1
    text = text[startIdx:].replace("<|eot_id|>", "")
    print(text)
    return text
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)