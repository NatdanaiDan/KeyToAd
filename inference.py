import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig,BitsAndBytesConfig
from helper.prompter import Prompter
import gc
import time
import os
model_default = [
    "pythainlp/wangchanglm-7.5B-sft-enth",
    "openthaigpt/openthaigpt-1.0.0-beta-13b-chat-hf",
    "scb10x/typhoon-7b",
]

peft_model_id_cosmetic= [
    "SADATO/wangchanglm_cosmetic",
    "SADATO/openthaigpt_cosmetic",
    "SADATO/typhoon_cosmetic",
]
peft_model_id_furniture= [
    "SADATO/wangchanglm_furniture",
    "SADATO/openthaigpt_furniture",
    "SADATO/typhoon_furniture",
]
promp_list=['xglm','llama_v2','mistral']



# Function to load the model, called only once during the session
def load_model(model_select,adapter_model):
    st.session_state.model, st.session_state.tokenizer = None, None
    torch.cuda.empty_cache()
    gc.collect()
    with st.status("Downloading data...", expanded=True) as status:
        st.write("Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(model_select)
        if st.session_state.device == "cuda":
        #     bnb_config = BitsAndBytesConfig(
        #         load_in_4bit=True,
        #         bnb_4bit_quant_type="nf4",
        #         bnb_4bit_compute_dtype=torch.float16,
        #   )
            model = AutoModelForCausalLM.from_pretrained(
                model_select,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                # load_in_8bit=True,
                device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)},
                # quantization_config=bnb_config,
                low_cpu_mem_usage=True,
            )
        elif st.session_state.device == "cpu":
            model = AutoModelForCausalLM.from_pretrained(
                model_select,
                device_map={"": "cpu"},
                low_cpu_mem_usage=True,
            )
        st.write("Adapter model...")
        model.load_adapter(adapter_model)
        status.update(label="Download complete!", state="complete", expanded=False)
        return model, tokenizer




# Check if the model is already loaded in session_state
if "model" not in st.session_state:
    st.session_state.model, st.session_state.tokenizer = None, None
if "prompter" not in st.session_state:
    st.session_state.prompter = None
if "response" not in st.session_state:
    st.session_state.response = ""

if "device" not in st.session_state:
    st.session_state.device = "cuda" if torch.cuda.is_available() else "cpu"
if "model_index" not in st.session_state:
    st.session_state.model_index = None

with st.container():
    st.title("KeyToad")
    model_options = ["Select a model", "WangChanGLM", "Openthaigpt", "Typhoon", "Unload model"]
    selected_model = st.selectbox("Choose a model", model_options)
    # type_options = ["Select a type", "Cosmetic", "Furniture"]
    # selected_type = st.selectbox("Choose a type", type_options)

    if selected_model == "Unload model":
        if st.button("Unload Model"):
            st.session_state.model, st.session_state.tokenizer = None, None
            st.session_state.prompter = None
            st.session_state.response = ""
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()


    elif selected_model != "Select a model":
        index_model = model_options.index(selected_model)-1
        type_options = ["Select a type", "Cosmetic", "Furniture"]
        selected_type = st.selectbox("Choose a type", type_options)
        index_type = type_options.index(selected_type)
        if st.button("Load Model") and  selected_type != "Select a type":
            if st.session_state.model is None:
                if index_type == 1:
                    st.session_state.model, st.session_state.tokenizer = load_model(model_default[index_model],peft_model_id_cosmetic[index_model])
                elif index_type == 2:
                    st.session_state.model, st.session_state.tokenizer = load_model(model_default[index_model],peft_model_id_furniture[index_model])
                st.session_state.prompter = Prompter(promp_list[index_model])
            else:
                st.write("Model already loaded! Need to Unload Model before loading new model")


with st.container():
    temp = st.slider("Temperature", 0.0, 1.5, 0.7, key="temp")
    top_k = st.slider("Top K", 0, 100, 50, key="top_k")
    top_p = st.slider("Top P", 0.0, 1.0, 0.7, key="top_p")
    max_length = st.slider("Max Length", 32, 128, 64, key="max_length")


with st.container():
    st.write("ตัวอย่างของ Input สวยงาม,ลิปสติก")
    prompt = st.text_input("prompt", value="", key="prompt")
    instructions = st.text_input("instructions", value="สร้างประโยคโฆษณาโดยใช้คำที่กำหนดให้", key="instructions")
    
with st.container():
    if st.button("🙋‍♀️", key="button"):
        generation_config = GenerationConfig(
            temperature=temp,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            typical_p=1.0,
        )
        with st.spinner(""):
            try:
                prompt_gen = st.session_state.prompter.generate_prompt(
                    instructions,prompt
                )
                print(prompt_gen)
                batch = st.session_state.tokenizer(prompt_gen, return_tensors="pt").to(
                    st.session_state.device
                )
                st.session_state.model.config.use_cache = False
                st.session_state.model.eval()
                with torch.no_grad():
                    output_tokens = st.session_state.model.generate(
                        input_ids=batch["input_ids"],
                        pad_token_id=st.session_state.tokenizer.eos_token_id,
                        max_new_tokens=max_length,
                        min_length=len(batch["input_ids"][0]) + 32,
                        no_repeat_ngram_size=4,
                        generation_config=generation_config,
                    )
                response = st.session_state.tokenizer.decode(
                    output_tokens[0], skip_special_tokens=True
                )
                st.session_state.response = st.session_state.prompter.get_response(response)
            except Exception as e:
                st.write(e)
                st.session_state.response = "Error"

st.write("Response:", st.session_state.response)