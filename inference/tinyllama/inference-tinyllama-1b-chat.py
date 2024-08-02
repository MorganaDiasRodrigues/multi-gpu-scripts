import gradio as gr
import torch
from transformers import pipeline
import time
from huggingface_hub import login
login("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")

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
    start_time = time.time()

    messages = [
        {
            "role": "system",
            "content": "You are a sales expert in AI and technology. You are chatting with a potential customer who is interested in AI solutions for their business. The customer is asking about the services you offer. Respond to the customer's inquiries and provide information about the AI solutions you offer.",
        },
        {"role": "user", "content": "I'm looking for an AI solution to help me with my business."},
        {"role": "assistant", "content": "Arrr matey! Our AI solution can help ye with yer business by providin' insights into yer data."},
        {"role": "user", "content": "I'm interested in having a chatbot for my website."},
        {"role": "assistant", "content": "Ahoy! Our chatbot solution can help ye engage with yer customers and provide support 24/7."},
        {"role": "user", "content": "I'm looking for an automated solution to help me with my marketing campaigns."},
        {"role": "assistant", "content": "Aye aye! Our AI marketing solution can help ye optimize yer marketing campaigns and reach yer target audience more effectively."},
        {"role": "user", "content": user_input},
    ]

    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    
    result = outputs[0]["generated_text"]
    inference_time = time.time() - start_time
    return result, f"Inference time: {inference_time:.2f} seconds"

# Setup pipeline
pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v0.6", torch_dtype=torch.bfloat16, device_map="auto")

# Gradio interface
gpu_info = get_gpu_info()
gpu_info_str = "\n".join(gpu_info)

def infer(user_input):
    result, inference_time = run_inference(user_input)
    return result, inference_time

with gr.Blocks() as demo:
    gr.Markdown("# AI Solution Sales Expert")
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

demo.launch(share=True)
