





import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", # Or 'fp4'
    bnb_4bit_compute_dtype=torch.float16, # Or torch.bfloat16
    bnb_4bit_use_double_quant=True
)
peft_model_id = "./Laala-3.2-3B-quant"
config = PeftConfig.from_pretrained(peft_model_id)
print(config.base_model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,
                                            return_dict = True,
                                            quantization_config=bnb_config,
                                            device_map = 'auto')
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

model = PeftModel.from_pretrained(model,peft_model_id)
model.to("cuda")

inputs = tokenizer("t boot, phones attempt to retrieve a configuration file from a provisioning service. The location of the configuration file can be set via Dynamic Host", return_tensors="pt", return_attention_mask=True)
inputs.to("cuda")
model.eval()
config = GenerationConfig(do_sample=True, temperature=0.1)
outputs = model.generate(**inputs, max_length=500,pad_token_id=tokenizer.eos_token_id, generation_config=config )

text = tokenizer.batch_decode(outputs)[0]
print(text)