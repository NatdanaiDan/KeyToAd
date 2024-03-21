#!/bin/bash
#SBATCH -A testktd
#SBATCH --gres=gpu:1
#SBATCH --job-name=KTDT0
#SBATCH -t 3-00:00:00
#SBATCH -N 1
#SBATCH -o experiment/OT0%j.txt
#SBATCH -e experiment/ET0%j.txt
#SBATCH --mem 64G
#SBATCH --exclude prism-5
source activate ktd

#### E0

name_csv="E0_wangchanglm_cosmetic_gpt4_test"
python gen_totest.py --base_model pythainlp/wangchanglm-7.5B-sft-enth --prompter_name xglm --dataset /home/natdanai/Download/dataset/test/cosmetic_kwgpt4_test.csv --name_csv $name_csv
# name_csv="E0_wangchanglm_cosmetic_noinput_gpt4_test"
# python gen_totest.py --base_model pythainlp/wangchanglm-7.5B-sft-enth --adapter /home/natdanai/allcheckpoint/checkpointwithoutkeyword --prompter_name xglm --dataset /home/natdanai/Download/dataset/test/cosmetic_kwgpt4_test.csv --name_csv $name_csv



#### E1 compare model in cosmetic gpt4 and kwpostag
# name_csv="E1_typhoon_cosmetic_kwgpt4_test"
# python gen_totest.py --base_model scb10x/typhoon-7b --adapter /home/natdanai/KeyToad/script/model/E1/typhoon_cosmetic_kwgpt4_train_E1_gpt4_unsloth --prompter_name mistral --dataset /home/natdanai/Download/dataset/test/cosmetic_kwgpt4_test.csv --name_csv $name_csv
# name_csv="E1_typhoon_cosmetic_kwpostag_test"
# python gen_totest.py --base_model scb10x/typhoon-7b --adapter /home/natdanai/KeyToad/script/model/E1/typhoon_cosmetic_kwpostag_train_E1_kwpostag_unsloth --prompter_name mistral --dataset /home/natdanai/Download/dataset/test/cosmetic_kwgpt4_test.csv --name_csv $name_csv

# ### E2 Compare model in neftune_noise_alpha
# name_csv="E2_typhoon_cosmetic_kwgpt4_neftune_test"
# python gen_totest.py --base_model scb10x/typhoon-7b --adapter /home/natdanai/KeyToad/script/model/E2/typhoon_cosmetic_kwgpt4_train_E2_neftune_gpt4 --prompter_name mistral --dataset /home/natdanai/Download/dataset/test/cosmetic_kwgpt4_test.csv --name_csv $name_csv
# name_csv="E2_typhoon_cosmetic_kwpostag_neftune_test"
# python gen_totest.py --base_model scb10x/typhoon-7b --adapter /home/natdanai/KeyToad/script/model/E2/typhoon_cosmetic_kwpostag_train_E2_neftune_kwpostag --prompter_name mistral --dataset /home/natdanai/Download/dataset/test/cosmetic_kwgpt4_test.csv --name_csv $name_csv

# #### E3 Compare model in augment

# name_csv="E3_typhoon_cosmetic_gpt4_shuffle_test"
# python gen_totest.py --base_model scb10x/typhoon-7b --adapter /home/natdanai/KeyToad/script/model/E3/typhoon_E3_shuffle_gpt4 --prompter_name mistral --dataset /home/natdanai/Download/dataset/test/cosmetic_kwgpt4_test.csv --name_csv $name_csv

# name_csv="E3_typhoon_cosmetic_gpt4_shuffle_test_augment"
# python gen_totest.py --base_model scb10x/typhoon-7b --adapter /home/natdanai/KeyToad/script/model/E3/typhoon_E3_shuffle_gpt4 --prompter_name mistral --dataset /home/natdanai/Download/dataset/test/cosmetic_kwgpt4_test_augment.csv --name_csv $name_csv

# name_csv="E3_typhoon_cosmetic_gpt4_shuffle_augment_test"
# python gen_totest.py --base_model scb10x/typhoon-7b --adapter /home/natdanai/KeyToad/script/model/E3/typhoon_E3_shuffle_augment_gpt4 --prompter_name mistral --dataset /home/natdanai/Download/dataset/test/cosmetic_kwgpt4_test.csv --name_csv $name_csv

# name_csv="E3_typhoon_cosmetic_gpt4_shuffle_augment_test_augment"
# python gen_totest.py --base_model scb10x/typhoon-7b --adapter /home/natdanai/KeyToad/script/model/E3/typhoon_E3_shuffle_augment_gpt4 --prompter_name mistral --dataset /home/natdanai/Download/dataset/test/cosmetic_kwgpt4_test_augment.csv --name_csv $name_csv

# name_csv="E3_typhoon_cosmetic_gpt4_test_augment"
# python gen_totest.py --base_model scb10x/typhoon-7b --adapter /home/natdanai/KeyToad/script/model/E2/typhoon_cosmetic_kwgpt4_train_E2_neftune_gpt4 --prompter_name mistral --dataset /home/natdanai/Download/dataset/test/cosmetic_kwgpt4_test_augment.csv --name_csv $name_csv




#### E4 Compare model in augment cosmetic domain

# name_csv="E4_wangchanglm_cosmetic_gpt4_shuffle_test"
# python gen_totest.py --base_model pythainlp/wangchanglm-7.5B-sft-enth --adapter /home/natdanai/KeyToad/script/model/E4/wangchanglm_E4_wangchanglm_shuffle_gpt4 --prompter_name xglm --dataset /home/natdanai/Download/dataset/test/cosmetic_kwgpt4_test.csv --name_csv $name_csv

# name_csv="E4_wangchanglm_cosmetic_gpt4_shuffle_test_augment"
# python gen_totest.py --base_model pythainlp/wangchanglm-7.5B-sft-enth --adapter /home/natdanai/KeyToad/script/model/E4/wangchanglm_E4_wangchanglm_shuffle_gpt4 --prompter_name xglm --dataset /home/natdanai/Download/dataset/test/cosmetic_kwgpt4_test_augment.csv --name_csv $name_csv

# name_csv="E4_wangchanglm_cosmetic_gpt4_shuffle_augment_test"
# python gen_totest.py --base_model pythainlp/wangchanglm-7.5B-sft-enth --adapter /home/natdanai/KeyToad/script/model/E4/wangchanglm_E4_wangchanglm_shuffle_augment_gpt4 --prompter_name xglm --dataset /home/natdanai/Download/dataset/test/cosmetic_kwgpt4_test.csv --name_csv $name_csv

# name_csv="E4_wangchanglm_cosmetic_gpt4_shuffle_augment_test_augment"
# python gen_totest.py --base_model pythainlp/wangchanglm-7.5B-sft-enth --adapter /home/natdanai/KeyToad/script/model/E4/wangchanglm_E4_wangchanglm_shuffle_augment_gpt4 --prompter_name xglm --dataset /home/natdanai/Download/dataset/test/cosmetic_kwgpt4_test_augment.csv --name_csv $name_csv

# name_csv="E4_openthaigpt_cosmetic_gpt4_shuffle_test"
# python gen_totest.py --base_model openthaigpt/openthaigpt-1.0.0-beta-13b-chat-hf --adapter /home/natdanai/KeyToad/script/model/E4/openthaigpt_E4_openthaigpt_shuffle_gpt4 --prompter_name llama_v2 --dataset /home/natdanai/Download/dataset/test/cosmetic_kwgpt4_test.csv --name_csv $name_csv

# name_csv="E4_openthaigpt_cosmetic_gpt4_shuffle_test_augment"
# python gen_totest.py --base_model openthaigpt/openthaigpt-1.0.0-beta-13b-chat-hf --adapter /home/natdanai/KeyToad/script/model/E4/openthaigpt_E4_openthaigpt_shuffle_gpt4 --prompter_name llama_v2 --dataset /home/natdanai/Download/dataset/test/cosmetic_kwgpt4_test_augment.csv --name_csv $name_csv

# name_csv="E4_openthaigpt_cosmetic_gpt4_shuffle_augment_test"
# python gen_totest.py --base_model openthaigpt/openthaigpt-1.0.0-beta-13b-chat-hf --adapter /home/natdanai/KeyToad/script/model/E4/openthaigpt_E4_openthaigpt_shuffle_augment_gpt4 --prompter_name llama_v2 --dataset /home/natdanai/Download/dataset/test/cosmetic_kwgpt4_test.csv --name_csv $name_csv

# name_csv="E4_openthaigpt_cosmetic_gpt4_shuffle_augment_test_augment"
# python gen_totest.py --base_model openthaigpt/openthaigpt-1.0.0-beta-13b-chat-hf --adapter /home/natdanai/KeyToad/script/model/E4/openthaigpt_E4_openthaigpt_shuffle_augment_gpt4 --prompter_name llama_v2 --dataset /home/natdanai/Download/dataset/test/cosmetic_kwgpt4_test_augment.csv --name_csv $name_csv

# #### E5 Compare domain cosmetic done test in furniture

# name_csv="E5_typhoon_furniture_gpt4_shuffle_test"
# python gen_totest.py --base_model scb10x/typhoon-7b --adapter /home/natdanai/KeyToad/script/model/E5/typhoon_E5_typhoon_shuffle_gpt4 --prompter_name mistral --dataset /home/natdanai/Download/dataset/test/funirture_kwgpt4_test.csv --name_csv $name_csv

# name_csv="E5_typhoon_furniture_gpt4_shuffle_test_augment"
# python gen_totest.py --base_model scb10x/typhoon-7b --adapter /home/natdanai/KeyToad/script/model/E5/typhoon_E5_typhoon_shuffle_gpt4 --prompter_name mistral --dataset /home/natdanai/Download/dataset/test/funirture_kwgpt4_test_augment.csv --name_csv $name_csv

# name_csv="E5_typhoon_furniture_gpt4_shuffle_augment_test"
# python gen_totest.py --base_model scb10x/typhoon-7b --adapter /home/natdanai/KeyToad/script/model/E5/typhoon_E5_typhoon_shuffle_augment_gpt4 --prompter_name mistral --dataset /home/natdanai/Download/dataset/test/funirture_kwgpt4_test.csv --name_csv $name_csv

# name_csv="E5_typhoon_furniture_gpt4_shuffle_augment_test_augment"
# python gen_totest.py --base_model scb10x/typhoon-7b --adapter /home/natdanai/KeyToad/script/model/E5/typhoon_E5_typhoon_shuffle_augment_gpt4 --prompter_name mistral --dataset /home/natdanai/Download/dataset/test/funirture_kwgpt4_test_augment.csv --name_csv $name_csv


# #####
# name_csv="E5_openthaigpt_furniture_gpt4_shuffle_test"
# python gen_totest.py --base_model openthaigpt/openthaigpt-1.0.0-beta-13b-chat-hf --adapter /home/natdanai/KeyToad/script/model/E5/openthaigpt_E5_openthaigpt_shuffle_gpt4 --prompter_name llama_v2 --dataset /home/natdanai/Download/dataset/test/funirture_kwgpt4_test.csv --name_csv $name_csv

# name_csv="E5_openthaigpt_furniture_gpt4_shuffle_test_augment"
# python gen_totest.py --base_model openthaigpt/openthaigpt-1.0.0-beta-13b-chat-hf --adapter /home/natdanai/KeyToad/script/model/E5/openthaigpt_E5_openthaigpt_shuffle_gpt4 --prompter_name llama_v2 --dataset /home/natdanai/Download/dataset/test/funirture_kwgpt4_test_augment.csv --name_csv $name_csv

# name_csv="E5_openthaigpt_furniture_gpt4_shuffle_augment_test"
# python gen_totest.py --base_model openthaigpt/openthaigpt-1.0.0-beta-13b-chat-hf --adapter /home/natdanai/KeyToad/script/model/E5/openthaigpt_E5_openthaigpt_shuffle_augment_gpt4 --prompter_name llama_v2 --dataset /home/natdanai/Download/dataset/test/funirture_kwgpt4_test.csv --name_csv $name_csv

# name_csv="E5_openthaigpt_furniture_gpt4_shuffle_augment_test_augment"
# python gen_totest.py --base_model openthaigpt/openthaigpt-1.0.0-beta-13b-chat-hf --adapter /home/natdanai/KeyToad/script/model/E5/openthaigpt_E5_openthaigpt_shuffle_augment_gpt4 --prompter_name llama_v2 --dataset /home/natdanai/Download/dataset/test/funirture_kwgpt4_test_augment.csv --name_csv $name_csv


# #####
# name_csv="E5_wangchanglm_furniture_gpt4_shuffle_test"
# python gen_totest.py --base_model pythainlp/wangchanglm-7.5B-sft-enth --adapter /home/natdanai/KeyToad/script/model/E5/wangchanglm_E5_wangchanglm_shuffle_gpt4 --prompter_name xglm --dataset /home/natdanai/Download/dataset/test/funirture_kwgpt4_test.csv --name_csv $name_csv

# name_csv="E5_wangchanglm_furniture_gpt4_shuffle_test_augment"
# python gen_totest.py --base_model pythainlp/wangchanglm-7.5B-sft-enth --adapter /home/natdanai/KeyToad/script/model/E5/wangchanglm_E5_wangchanglm_shuffle_gpt4 --prompter_name xglm --dataset /home/natdanai/Download/dataset/test/funirture_kwgpt4_test_augment.csv --name_csv $name_csv

# name_csv="E5_wangchanglm_furniture_gpt4_shuffle_augment_test"
# python gen_totest.py --base_model pythainlp/wangchanglm-7.5B-sft-enth --adapter /home/natdanai/KeyToad/script/model/E5/wangchanglm_E5_wangchanglm_shuffle_augment_gpt4 --prompter_name xglm --dataset /home/natdanai/Download/dataset/test/funirture_kwgpt4_test.csv --name_csv $name_csv

# name_csv="E5_wangchanglm_furniture_gpt4_shuffle_augment_test_augment"
# python gen_totest.py --base_model pythainlp/wangchanglm-7.5B-sft-enth --adapter /home/natdanai/KeyToad/script/model/E5/wangchanglm_E5_wangchanglm_shuffle_augment_gpt4 --prompter_name xglm --dataset /home/natdanai/Download/dataset/test/funirture_kwgpt4_test_augment.csv --name_csv $name_csv
