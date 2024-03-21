#!/bin/bash
#SBATCH -A KEYTOAD
#SBATCH --gres=gpu:1
#SBATCH --job-name=unsloth
#SBATCH -t 3-00:00:00
#SBATCH -N 1
#SBATCH -o outputbatch/o%j.txt
#SBATCH -e outputbatch/e%j.txt
#SBATCH --mem 48G



source activate ktd
#####
#### Experiment 1 compare model in cosmetic gpt4 and kwpostag
#####

# where="cosmetic_kwgpt4_train"
# model="scb10x/typhoon-7b"
# model_save="typhoon"
# name_project="E1_gpt4_unsloth"
# python train_unsloth.py \
#     --model_name ${model} \
#     --dataset_name "/home/natdanai/Download/dataset/train/$where.csv" \
#     --output_dir "model/E1/${model_save}_${where}_${name_project}" \
#     --prompter_name mistral \
#     --epochs 10 \
#     --name_project ${name_project} \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size  4 \
#     --gradient_accumulation_steps 4

# where="cosmetic_kwpostag_train"
# model="scb10x/typhoon-7b"
# model_save="typhoon"
# name_project="E1_kwpostag_unsloth"
# python train_unsloth.py \
#     --model_name ${model} \
#     --dataset_name "/home/natdanai/Download/dataset/train/$where.csv" \
#     --output_dir "model/E1/${model_save}_${where}_${name_project}" \
#     --prompter_name mistral \
#     --epochs 10 \
#     --name_project ${name_project} \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size  4 \
#     --gradient_accumulation_steps 4
    
# where="cosmetic_kwgpt4_train"
# model="scb10x/typhoon-7b"
# model_save="typhoon"
# name_project="E1_gpt4"
# python train.py \
#     --model_name ${model} \
#     --dataset_name "/home/natdanai/Download/dataset/train/$where.csv" \
#     --output_dir "model/E1/${model_save}_${where}" \
#     --prompter_name mistral \
#     --flash_attention True \
#     --epochs 10 \
#     --name_project ${name_project}


# where="cosmetic_kwpostag_train"
# model="scb10x/typhoon-7b"
# model_save="typhoon"
# name_project="E1_kwpostag"
# python train.py \
#     --model_name ${model} \
#     --dataset_name "/home/natdanai/Download/dataset/train/$where.csv" \
#     --output_dir "model/E1/${model_save}_${where}" \
#     --prompter_name mistral \
#     --flash_attention True \
#     --epochs 10 \
#     --name_project ${name_project}

#####
#### Experiment 2 neftune_noise_alpha
#####

# where="cosmetic_kwgpt4_train"
# model="scb10x/typhoon-7b"
# model_save="typhoon"
# name_project="E2_neftune_gpt4"
# python train_unsloth.py \
#     --model_name ${model} \
#     --dataset_name "/home/natdanai/Download/dataset/train/$where.csv" \
#     --output_dir "model/E2/${model_save}_${where}_${name_project}" \
#     --prompter_name mistral \
#     --epochs 10 \
#     --name_project ${name_project} \
#     --neftune_noise_alpha 5 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size  4 \
#     --gradient_accumulation_steps 4 


# where="cosmetic_kwpostag_train"
# model="scb10x/typhoon-7b"
# model_save="typhoon"
# name_project="E2_neftune_kwpostag"
# python train_unsloth.py \
#     --model_name ${model} \
#     --dataset_name "/home/natdanai/Download/dataset/train/$where.csv" \
#     --output_dir "model/E2/${model_save}_${where}_${name_project}" \
#     --prompter_name mistral \
#     --epochs 10 \
#     --name_project ${name_project} \
#     --neftune_noise_alpha 5 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size  4 \
#     --gradient_accumulation_steps 4 

# ### Experiment 3

# where="cosmetic_kwgpt4_train_shuffle"
# model="scb10x/typhoon-7b"
# model_save="typhoon"
# name_project="E3_shuffle_gpt4"
# python train_unsloth.py \
#     --model_name ${model} \
#     --dataset_name "/home/natdanai/Download/dataset/train/$where.csv" \
#     --output_dir "model/E3/${model_save}_${name_project}" \
#     --prompter_name mistral \
#     --epochs 10 \
#     --name_project ${name_project} \
#     --neftune_noise_alpha 5 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size  4 \
#     --gradient_accumulation_steps 4 

# where="cosmetic_kwgpt4_train_shuffle_augment"
# model="scb10x/typhoon-7b"
# model_save="typhoon"
# name_project="E3_shuffle_augment_gpt4"
# python train_unsloth.py \
#     --model_name ${model} \
#     --dataset_name "/home/natdanai/Download/dataset/train/$where.csv" \
#     --output_dir "model/E3/${model_save}_${name_project}" \
#     --prompter_name mistral \
#     --epochs 10 \
#     --name_project ${name_project} \
#     --neftune_noise_alpha 5 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size  4 \
#     --gradient_accumulation_steps 4 


#### Experiment 4

# where="cosmetic_kwgpt4_train_shuffle"
# model="pythainlp/wangchanglm-7.5B-sft-enth"
# model_save="wangchanglm"
# name_project="E4_wangchanglm_shuffle_gpt4"
# python train.py \
#     --model_name "${model}" \
#     --dataset_name "/home/natdanai/Download/dataset/train/${where}.csv" \
#     --output_dir "model/E4/${model_save}_${name_project}" \
#     --prompter_name xglm \
#     --epochs 10 \
#     --neftune_noise_alpha 5 \
#     --name_project ${name_project}

# where="cosmetic_kwgpt4_train_shuffle_augment"
# model="pythainlp/wangchanglm-7.5B-sft-enth"
# model_save="wangchanglm"
# name_project="E4_wangchanglm_shuffle_augment_gpt4"
# python train.py \
#     --model_name "${model}" \
#     --dataset_name "/home/natdanai/Download/dataset/train/${where}.csv" \
#     --output_dir "model/E4/${model_save}_${name_project}" \
#     --prompter_name xglm \
#     --epochs 10 \
#     --neftune_noise_alpha 5 \
#     --name_project ${name_project}

# where="cosmetic_kwgpt4_train_shuffle"
# model="openthaigpt/openthaigpt-1.0.0-beta-13b-chat-hf"
# model_save="openthaigpt"
# name_project="E4_openthaigpt_shuffle_gpt4"
# python train_unsloth.py \
#     --model_name ${model} \
#     --dataset_name "/home/natdanai/Download/dataset/train/$where.csv" \
#     --output_dir "model/E4/${model_save}_${name_project}" \
#     --prompter_name llama_v2 \
#     --epochs 10 \
#     --name_project ${name_project} \
#     --neftune_noise_alpha 5 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size  4 \
#     --gradient_accumulation_steps 4 

# where="cosmetic_kwgpt4_train_shuffle_augment"
# model="openthaigpt/openthaigpt-1.0.0-beta-13b-chat-hf"
# model_save="openthaigpt"
# name_project="E4_openthaigpt_shuffle_augment_gpt4"
# python train_unsloth.py \
#     --model_name ${model} \
#     --dataset_name "/home/natdanai/Download/dataset/train/$where.csv" \
#     --output_dir "model/E4/${model_save}_${name_project}" \
#     --prompter_name llama_v2 \
#     --epochs 10 \
#     --name_project ${name_project} \
#     --neftune_noise_alpha 5 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size  4 \
#     --gradient_accumulation_steps 4 

#### Experiment 5

# where="funirture_kwgpt4_train_shuffle"
# model="scb10x/typhoon-7b"
# model_save="typhoon"
# name_project="E5_typhoon_shuffle_gpt4"
# python train_unsloth.py \
#     --model_name ${model} \
#     --dataset_name "/home/natdanai/Download/dataset/train/$where.csv" \
#     --output_dir "model/E5/${model_save}_${name_project}" \
#     --prompter_name mistral \
#     --epochs 10 \
#     --name_project ${name_project} \
#     --neftune_noise_alpha 5 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size  4 \
#     --gradient_accumulation_steps 4 

# where="funirture_kwgpt4_train_shuffle_augment"
# model="scb10x/typhoon-7b"
# model_save="typhoon"
# name_project="E5_typhoon_shuffle_augment_gpt4"
# python train_unsloth.py \
#     --model_name ${model} \
#     --dataset_name "/home/natdanai/Download/dataset/train/$where.csv" \
#     --output_dir "model/E5/${model_save}_${name_project}" \
#     --prompter_name mistral \
#     --epochs 10 \
#     --name_project ${name_project} \
#     --neftune_noise_alpha 5 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size  4 \
#     --gradient_accumulation_steps 4 

# where="funirture_kwgpt4_train_shuffle"
# model="pythainlp/wangchanglm-7.5B-sft-enth"
# model_save="wangchanglm"
# name_project="E5_wangchanglm_shuffle_gpt4"
# python train.py \
#     --model_name "${model}" \
#     --dataset_name "/home/natdanai/Download/dataset/train/${where}.csv" \
#     --output_dir "model/E5/${model_save}_${name_project}" \
#     --prompter_name xglm \
#     --epochs 10 \
#     --neftune_noise_alpha 5 \
#     --name_project ${name_project}

# where="funirture_kwgpt4_train_shuffle_augment"
# model="pythainlp/wangchanglm-7.5B-sft-enth"
# model_save="wangchanglm"
# name_project="E5_wangchanglm_shuffle_augment_gpt4"
# python train.py \
#     --model_name "${model}" \
#     --dataset_name "/home/natdanai/Download/dataset/train/${where}.csv" \
#     --output_dir "model/E5/${model_save}_${name_project}" \
#     --prompter_name xglm \
#     --epochs 10 \
#     --neftune_noise_alpha 5 \
#     --name_project ${name_project}

# where="funirture_kwgpt4_train_shuffle"
# model="openthaigpt/openthaigpt-1.0.0-beta-13b-chat-hf"
# model_save="openthaigpt"
# name_project="E5_openthaigpt_shuffle_gpt4"
# python train_unsloth.py \
#     --model_name ${model} \
#     --dataset_name "/home/natdanai/Download/dataset/train/$where.csv" \
#     --output_dir "model/E5/${model_save}_${name_project}" \
#     --prompter_name llama_v2 \
#     --epochs 10 \
#     --name_project ${name_project} \
#     --neftune_noise_alpha 5 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size  4 \
#     --gradient_accumulation_steps 4 

# where="funirture_kwgpt4_train_shuffle_augment"
# model="openthaigpt/openthaigpt-1.0.0-beta-13b-chat-hf"
# model_save="openthaigpt"
# name_project="E5_openthaigpt_shuffle_augment_gpt4"
# python train_unsloth.py \
#     --model_name ${model} \
#     --dataset_name "/home/natdanai/Download/dataset/train/$where.csv" \
#     --output_dir "model/E5/${model_save}_${name_project}" \
#     --prompter_name llama_v2 \
#     --epochs 10 \
#     --name_project ${name_project} \
#     --neftune_noise_alpha 5 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size  4 \
#     --gradient_accumulation_steps 4