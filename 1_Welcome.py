import streamlit as st
from transformers import AutoConfig, AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM
import openvino as ov

if "activity" not in st.session_state:
    st.session_state.activity = ""

ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""}

st.set_page_config(page_title="Rancangan Pengajaran Harian")

st.title("Rancangan Pengajaran Harian")

year = st.selectbox("Form", ("Form 4", "Form 5"))
subject = st.selectbox("Subject", ("Biology","Chemistry","Physics"))
chapter = st.selectbox("Chapter", ("Chapter 1: Fundamental of Biology", "Chapter 2: Physiology of Humans and Animals"))

objective = st.text_input("Objective")
material = st.multiselect("Materials", ["Textbook", "Presentation Slides", "Video"],["Textbook"])
focus = st.radio("Focus",["Simple", "Moderate", "Difficult"])

generate = st.button("Generate")

if generate:
    prompt = f'''
    You are a teacher who would like to create activities in class.
    Based on {subject} subject, {chapter} and the objective of the class is {objective},
    Generate an activity suitable for the subject above for students to better understand the chapter, while pairing it with {material}, and the activity level should be {focus}.
    '''

    model_dir = "/home/user/openvino_notebooks/notebooks/llm-chatbot/llama-2-chat-7b/INT8_compressed_weights"
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    ov_model = OVModelForCausalLM.from_pretrained(
        model_dir,
        device="CPU",
        ov_config=ov_config,
        config=AutoConfig.from_pretrained(model_dir, trust_remote_code=True),
        trust_remote_code=True,
    )

    tokenizer_kwargs = {"add_special_tokens": False}
    input_tokens = tok(prompt, return_tensors="pt", **tokenizer_kwargs)
    answer = ov_model.generate(**input_tokens, max_new_tokens=250)
    
    st.session_state.activity = tok.batch_decode(answer, skip_special_tokens=True)[0]
    
    st.switch_page("pages/2_Activity.py")
