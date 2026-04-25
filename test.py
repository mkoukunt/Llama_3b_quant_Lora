

import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
SYSTEM_PROMPT = """
You are a helpful assistant reads  the user request , analyze it and provides the correct api to call .
Write a answer that appropriately provides the correct api to call.
"""

def format_input(entry):
    instruction_text = (
        f"You are an helpful assistant provides the correct url to call to fulfill user request. "
       f"Write a answer that appropriately provides the  url to call."
        f"\n\n### question:\n{entry['question']}"
    )
    #input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text #+ input_text
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", # Or 'fp4'
    bnb_4bit_compute_dtype=torch.float16, # Or torch.bfloat16
    bnb_4bit_use_double_quant=True
)
peft_model_id = "./Laala-3.2-3B-inst-api"
config = PeftConfig.from_pretrained(peft_model_id)
print(config.base_model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,
                                            return_dict = True,
                                            quantization_config=bnb_config,
                                            device_map = 'auto')
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

model = PeftModel.from_pretrained(model,peft_model_id)
model.to("cuda")
entry={
        "question": "update a device for a user in the domain mgoud",
        "answer": "The capital of India is New Delhi."
    }
chat = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": entry["question"]},
]
chat = tokenizer.apply_chat_template(chat, tokenize=False)
inputs = tokenizer(chat, return_tensors="pt", return_attention_mask=True)
inputs.to("cuda")
model.eval()
config = GenerationConfig(do_sample=True, temperature=0.1)
outputs = model.generate(**inputs, max_length=1000,pad_token_id=tokenizer.eos_token_id, generation_config=config )

text = tokenizer.batch_decode(outputs)[0]
print(text)