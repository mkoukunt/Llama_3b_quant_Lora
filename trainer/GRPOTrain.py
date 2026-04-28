
import time

from datasets import load_dataset
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model
from torch.utils.data import Dataset

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json
import torch
from torch import nn
from trl import GRPOTrainer
from trl.rewards import accuracy_reward

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", # Or 'fp4'
    bnb_4bit_compute_dtype=torch.float16, # Or torch.bfloat16
    bnb_4bit_use_double_quant=True
)
peft_model_id = "../Laala-3.2-3B-ndp-inst-api"
config = PeftConfig.from_pretrained(peft_model_id)
print(config.base_model_name_or_path)

SYSTEM_PROMPT = """
You are an helpful assistant understands the user question and provides the correct tool name to call along with  the arguments to pass to the tool.
"""

model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,
                                            return_dict = True,
                                            quantization_config=bnb_config,
                                            device_map = 'auto')

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

lora_config = LoraConfig(
    task_type="CAUSAL_LM",#"SEQ_2_SEQ_LM",
    r=16,#8,
    lora_alpha=32,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    lora_dropout=0.01,
)


for param in model.parameters():
  param.requires_grad = False
  if param.ndim == 1:
    param.data = param.data.to(torch.float32)

  model.gradient_checkpointing_enable()
  model.enable_input_require_grads()

  class CastOutputToFloat(nn.Sequential):
    def forward(self, x) :
      return super().forward(x).to(torch.float32)
  #model.lm_head = CastOutputToFloat(model.lm_head)
  model.lm_head.weight.data = model.lm_head.weight.data.to(torch.float32)

model = PeftModel.from_pretrained(model,peft_model_id)
model.to("cuda")

def download_and_load_file(file_path, url):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

file_path = "../data/ndp-instruction-rl.json"
url = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)
data = download_and_load_file(file_path, url)
train_data = data

def extract_hash_answer(text: str) -> str | None:
    # Extracts the answer if it follows a '####' delimiter.
    if "####" not in text:
        return None
    return text.split("####")[1].strip()




class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        # Pre-tokenize texts
        self.encoded_texts = []
        for entry in data:
            dt={'prompt':[{'role': 'system', 'content': SYSTEM_PROMPT},# One-shot examples can be added here if desired.
            {'role': 'user', 'content': entry['question']}], 'answer': entry['answer']}

            self.encoded_texts.append(dt)




    def __getitem__(self, index):
        #print(self.encoded_texts[index])
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)

def custom_collate_fn(
        batch,
        pad_token_id=128001,
        ignore_index=-100,
        allowed_max_length=None,
        device="cpu"
):
    # Find the longest sequence in the batch
    batch_max_length = max(len(item) + 1 for item in batch)
    print(batch_max_length)
    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        # Pad sequences to max_length
        padded = (
                new_item + [pad_token_id] *
                (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets

        # New: Replace all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # New: Optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs and targets to tensors and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

from functools import partial
customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=1024)
from torch.utils.data import DataLoader


num_workers = 0
batch_size = 8

torch.manual_seed(123)

train_dataset = InstructionDataset(train_data, tokenizer)

def get_gsm8k_questions(split="train") -> Dataset:
    # Load and process the GSM8K dataset.
    data = train_dataset
    data = data.map(lambda x: {
        'prompt': tokenizer .apply_chat_template([
            {'role': 'system', 'content': SYSTEM_PROMPT},
            # One-shot examples can be added here if desired.
            {'role': 'user', 'content': x['question']}
        ]),
        'answer': extract_hash_answer(x['answer'])
    })
    return data


start_time = time.time()

torch.manual_seed(123)


num_epochs = 10

#dataset = load_dataset("./data/api-task-split-data.json", split="train")


#tokenizer.chat_template="you are a assistant"
def extract_xml_answer(text: str) -> str:
    # Extracts the content within <answer> tags.
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    print(prompts)
    # Compares the model's output with the expected answer.
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-' * 20, f"Question:\n{q}",
          f"\nAnswer:\n{answer[0]}",
          f"\nResponse:\n{responses[0]}",
          f"\nExtracted:\n{extracted_responses[0]}")
    return [safe_compare(r,a) for r, a in zip(extracted_responses, answer)]
def safe_compare(r, a):
    try:
        # Attempt to parse both JSON strings
        return 2.0 if json.loads(r) == json.loads(a) else 0.0
    except (json.JSONDecodeError, TypeError, ValueError):
        # Return 0.0 if decoding fails
        return 0.0
tokenizer.chat_template="{%- for message in messages %} {{- '<|' + message['role'] + |>\n' }}    {{- message['content'] + eos_token }}{%- endfor %}{%- if add_generation_prompt %}    {{- '<|assistant|>\n' }}{%- endif %}"
trainer = GRPOTrainer(
    model=model,
    reward_funcs=correctness_reward_func,
    train_dataset=train_dataset,


)
trainer.train()

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")
model.save_pretrained("../Laala-3.2-3B-quant-inst-rl")
tokenizer.save_pretrained("../Laala-3.2-3B-quant-inst-rl")