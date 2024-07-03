import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import time

# Function to get GPU information
def get_gpu_info():
    num_gpus = torch.cuda.device_count()
    gpu_info = []
    for gpus in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(gpus)
        gpu_memory = torch.cuda.get_device_properties(gpus).total_memory / 1e9
        gpu_info.append(f"GPU {gpus} name: {gpu_name}, memory: {gpu_memory} GB")
    return gpu_info

# Function to run inference
def run_inference(user_input):
    chat = [
        { "role": "user", "content": "I'm looking for an AI solution to help me with my business."},
        { "role": "assistant", "content": "Our AI solution can help you with your business by providing insights into your data." },
        { "role": "user", "content": "I'm interested in having a chatbot for my website." },
        { "role": "assistant", "content": "Our chatbot solution can help you engage with your customers and provide support 24/7." },
        { "role": "user", "content": "I'm looking for an automated solution to help me with my marketing campaigns." },
        { "role": "assistant", "content": "Our AI marketing solution can help you optimize your marketing campaigns and reach your target audience more effectively." },
        { "role": "user", "content": user_input },
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    start_time = time.time()
    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=300)
    result = tokenizer.decode(outputs[0])
    inference_time = time.time() - start_time
    return result, f"Inference time: {inference_time:.2f} seconds"

# Setup model and tokenizer
model_id = "gg-hf/gemma-2b-it"
dtype = torch.bfloat16
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    quantization_config=quantization_config
)

# Gradio interface
gpu_info = get_gpu_info()
gpu_info_str = "\n".join(gpu_info)

def infer(user_input):
    result, inference_time = run_inference(user_input)
    return result, inference_time

with gr.Blocks() as demo:
    gr.Markdown("# AI Solution Salesperson")
    with gr.Row():
        with gr.Column():
            gr.Markdown("## GPU Information")
            gpu_info_box = gr.Textbox(value=gpu_info_str, interactive=False, lines=5)
        with gr.Column():
            gr.Markdown("## Input for LLM")
            input_textbox = gr.Textbox(label="Enter your text here")
            submit_button = gr.Button("Submit")
    
    result_textbox = gr.Textbox(label="Generated Response", interactive=False)
    inference_time_textbox = gr.Textbox(label="Inference Time", interactive=False)
    
    submit_button.click(infer, inputs=[input_textbox], outputs=[result_textbox, inference_time_textbox])

demo.launch()
