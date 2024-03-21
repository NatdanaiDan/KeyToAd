from dataclasses import dataclass, field
from typing import Optional
from transformers.integrations import is_deepspeed_zero3_enabled
import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from helper.prompter import Prompter
from trl import SFTTrainer
import wandb
import time
import bitsandbytes as bnb
import numpy as np
import evaluate
import os
from unsloth import FastLanguageModel

@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """

    """
    Data env
    """
    model_name: Optional[str] = field(metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(metadata={"help": "the dataset name"})
    output_dir: Optional[str] = field(metadata={"help": "the output directory"})
    prompter_name: Optional[str] = field(metadata={"help": "the prompter_name"})
    use_8bit: Optional[bool] = field(default=False, metadata={"help": "the 8bit"})
    fp_16: Optional[bool] = field(default=False, metadata={"help": "the 8bit"})

    """
    LoRa Config
    """
    lora_r: Optional[float] = field(default=32, metadata={"help": "the lora_r"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora_alpha"})
    lora_dropout: Optional[float] = field(
        default=0.05, metadata={"help": "the lora_dropout"}
    )

    """
    Training Arguments
    """
    per_device_train_batch_size: Optional[int] = field(
        default=1, metadata={"help": "the per_device_train_batch_size"}
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=1, metadata={"help": "the per_device_eval_batch_size"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=32, metadata={"help": "the gradient_accumulation_steps"}
    )
    max_grad_norm: Optional[float] = field(
        default=1.0, metadata={"help": "the max_grad_norm"}
    )
    learning_rate: Optional[float] = field(
        default=5e-5, metadata={"help": "the learning_rate"}
    )
    weight_decay: Optional[float] = field(
        default=0.001, metadata={"help": "the weight_decay"}
    )
    optimizer_name: Optional[str] = field(
        default="adamw_hf", metadata={"help": "the optimizer_name"}
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine", metadata={"help": "the lr_scheduler_type"}
    )
    warmup_ratio: Optional[float] = field(
        default=0.0025, metadata={"help": "the warmup_ratio"}
    )
    group_by_length: Optional[bool] = field(
        default=False, metadata={"help": "the group_by_length"}
    )
    epochs: Optional[int] = field(default=5, metadata={"help": "the num_train_epochs"})
    deepspeed: Optional[str] = field(default=None, metadata={"help": "the deepspeed"})
    local_rank: Optional[int] = field(default=0)
    flash_attention: Optional[bool] = field(
        default=False, metadata={"help": "the flash_attention"}
    )
    neftune_noise_alpha: Optional[float] = field(
        default=None, metadata={"help": "the neftune_noise_alpha"}
    )
    name_project: Optional[str] = field(
        default="keytoad_test1", metadata={"help": "the name_project"}
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

###########    Prompter  ################

print(script_args.use_8bit)
prompter = Prompter(script_args.prompter_name)

########### wandb init ################
wandb.login(key="d69f749409d3fc33eef9678cf1257b17c99644e5")
run = wandb.init(
    project=f"keytoad_test1",
    job_type="training",
    anonymous="allow",
    name=f"{script_args.name_project}",
)

###########    Load model   ################

if script_args.fp_16:
    use_fp16 = True
    use_bf16 = False
else:
    use_fp16 = not torch.cuda.is_bf16_supported()
    use_bf16 = torch.cuda.is_bf16_supported()

device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
print(device_map)
torch_dtype = torch.bfloat16 if use_bf16 else torch.float16


model,tokenizer = FastLanguageModel.from_pretrained(
    model_name = script_args.model_name,
    # mex_seq_length = 2048,
    dtype = torch_dtype,
    device_map = device_map,
)


model.config.use_cache = False
###########    Load dataset   ################

try:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_eos_token = True
    tokenizer.add_bos_token, tokenizer.add_eos_token
    tokenizer.padding_side = "right"
except:
    pass

dataset = load_dataset("csv", data_files=script_args.dataset_name, split="train")
dataset = dataset.train_test_split(test_size=0.11, shuffle=True, seed=42)


def prepare_dataset(example):
    full_prompt = prompter.generate_prompt(
        example["instruction"],
        example["input"],
        example["output"],
    )

    example["text"] = full_prompt + tokenizer.eos_token
    return example


dataset = dataset.map(
    prepare_dataset, num_proc=16, remove_columns=["instruction", "input", "output"]
)
print("*" * 20)
print(dataset)
print("*" * 20)

###########    Training Arguments   ################


training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optim=script_args.optimizer_name,
    learning_rate=script_args.learning_rate,
    weight_decay=script_args.weight_decay,
    lr_scheduler_type=script_args.lr_scheduler_type,
    warmup_ratio=script_args.warmup_ratio,
    max_grad_norm=script_args.max_grad_norm,
    num_train_epochs=script_args.epochs,
    group_by_length=script_args.group_by_length,
    # fp16 = True,
    fp16=use_fp16,
    bf16=use_bf16,
    logging_steps=20,
    evaluation_strategy="epoch",
    save_total_limit=1,
    save_strategy="epoch",
    max_steps=-1,
    report_to="wandb",
    deepspeed=script_args.deepspeed,
    load_best_model_at_end=True,
    seed=42,
)


###########    Load LoRa   ################


def find_target_modules(model):
    # Initialize a Set to Store Unique Layers
    unique_layers = set()

    # Iterate Over All Named Modules in the Model
    for name, module in model.named_modules():
        # print(name,module)
        # Check if the Module Type Contains 'Linear4bit'
        if "Linear" in str(type(module)):
            # Extract the Type of the Layer
            layer_type = name.split(".")[-1]

            # Add the Layer Type to the Set of Unique Layers
            unique_layers.add(layer_type)

    # Return the Set of Unique Layers Converted to a List
    # remove 'lm_head' from list
    unique_layers.remove("lm_head")
    return list(unique_layers)


print(model)
print(find_target_modules(model))
model = FastLanguageModel.get_peft_model(
    model,
    r=script_args.lora_r,
    lora_alpha =script_args.lora_alpha,
    lora_dropout =script_args.lora_dropout,
    bias="none",
    use_gradient_checkpointing = True,
    random_state = 42,
    target_modules = find_target_modules(model),
)


###########    SFTTrainer   ################

# rouge = evaluate.load("rouge")


# def compute_metrics(eval_preds):
#     labels_ids = eval_preds.label_ids
#     labels_ids = np.where(labels_ids != -100, labels_ids, tokenizer.pad_token_id)
#     pred_ids = eval_preds.predictions
#     pred_ids = np.where(pred_ids != -100, pred_ids, tokenizer.pad_token_id)

#     pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
#     label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
#     result = rouge.compute(
#         predictions=pred_str,
#         references=label_str,
#         rouge_types=["rouge1", "rouge2", "rougeL"],
#     )
#     return result


# # Create a preprocessing function to extract out the proper logits from the model output
# def preprocess_logits_for_metrics(logits, labels):
#     if isinstance(logits, tuple):
#         logits = logits[0]
#     return logits.argmax(dim=-1)


print("SFTTrainer")
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    neftune_noise_alpha=script_args.neftune_noise_alpha,
    max_seq_length=2048,
    dataset_text_field="text",
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    tokenizer=tokenizer,
    # compute_metrics=compute_metrics,
    # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)


###########    Training   ################
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

train_result = trainer.train()


# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# model.config.use_cache = True
trainer.save_model(script_args.output_dir)
trainer.log_metrics("train", train_result.metrics)
trainer.save_metrics("train", train_result.metrics)
trainer.save_state()
