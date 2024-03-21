from dataclasses import dataclass, field
from typing import Optional
import torch
from tqdm import trange
from transformers import AutoModelForCausalLM,HfArgumentParser,AutoTokenizer,GenerationConfig
from helper.prompter import Prompter
import numpy as np 
import pandas as pd
import os
from transformers.integrations import is_deepspeed_zero3_enabled


def load_model(model_default):
        # bnb_config = BitsAndBytesConfig(
        #   load_in_4bit=True,
        #   bnb_4bit_quant_type="nf4",
        #   bnb_4bit_compute_dtype=torch.float16,
        #   )
        tokenizer = AutoTokenizer.from_pretrained(model_default)
        if device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                model_default,
                low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
        elif device == "cpu":
            model = AutoModelForCausalLM.from_pretrained(
                model_default,
                device_map={"": "cpu"},
                low_cpu_mem_usage=True,
            )
        return model,tokenizer

generation_config = GenerationConfig(
            temperature=0.2,
            top_p=0.96,
            top_k=50,
            do_sample=True,
            typical_p=1.0,
        )
device ="cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class ScriptArguments:
    
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """

    """
    Data env
    """
    dataset: Optional[str] = field(metadata={"help": "the model name"})
    base_model: Optional[str] = field(metadata={"help": "the dataset name"})
    prompter_name : Optional[str] = field(metadata={"help": "the prompter_name"})
    name_csv: Optional[str] = field(metadata={"help": "the output directory"})
    adapter: Optional[str] = field(default=None,metadata={"help": "the output directory"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


prompter = Prompter(script_args.prompter_name)
model,tokenizer = load_model(script_args.base_model)
if script_args.adapter != None:
    model.load_adapter(script_args.adapter)


df = pd.read_csv(script_args.dataset)
test_list = df['input'].tolist()
output_list = df['output'].tolist()
df=pd.DataFrame(columns=['word','response'])
# generation_config = GenerationConfig(
#             temperature=0.2,
#             do_sample=True,
#         )
for index in trange(len(test_list), desc="Generating Responses"):
    text = test_list[index]
    context = prompter.generate_prompt("สร้างประโยคโฆษณาโดยใช้คำที่กำหนดให้", text)
    batch = tokenizer(context, return_tensors='pt').to(device)
    output_text = output_list[index]
    max_length = tokenizer(output_text, return_tensors='pt').to(device)['input_ids'].shape[1]
    # print(max_length)
    with torch.no_grad():
        output_tokens = model.generate(
            input_ids=batch["input_ids"],
            # generation_config=generation_config,
            no_repeat_ngram_size=4,
            # early_stopping =True,
            max_new_tokens=max_length+20, # 512
            pad_token_id=tokenizer.eos_token_id,
            )
    response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    response = prompter.get_response(response)
    dfnew = pd.DataFrame({'word': text, 'response': response}, index=[0])
    df = pd.concat([df, dfnew])

#check script_args.output_dir exists

df.to_csv(f'gentest/output_{script_args.name_csv}.csv', index=False)