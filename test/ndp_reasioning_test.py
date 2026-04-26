

import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
SYSTEM_PROMPT = """
You are an helpful assistant converts user request into individual tasks.
Write a answer that appropriately provides the  individual tasks
"""

def format_input(entry):
    instruction_text = (
        f"You are an helpful assistant converts user request into individual tasks."
        f"Write a answer that appropriately provides the  individual tasks"
        f"\n\n### question:\n{entry['question']}"
    )

    instruction_text=(
        f"You are an helpful assistant read and analyze the  user question and converts it into sequence of tasks. "
        f"Think through the user question . Make sure to first add your step by step thought process within <think> </think> tags. Then, return your sequence of tasks in the following format: <guess> tesk1 > task2> </guess>."
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
peft_model_id = "../Laala-3.2-3B-ndp-inst-task"
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
        "question": "verify the snom-m500 model defaults in  the domain  josh and validate the model defaults for the model polycom-t46s for the domain  mgoud",
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