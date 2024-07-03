import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time
import torch

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
    # Predefined prompt
    input_text = """You're an salesperson expert in AI solutions.
Your task is to output your answer as a salesperson.
### Example 1
- Customer
    - Content: "I'm looking for an AI solution to help me with my business."
- Salesperson
    - Content: "Our AI solution can help you with your business by providing insights into your data."

### Example 2
- Customer
    - Content: "I'm interested in having a chatbot for my website."
- Salesperson:
    - Content: "Our chatbot solution can help you engage with your customers and provide support 24/7."

### Example 3
- Customer
    - Content: "I'm looking for an automated solution to help me with my marketing campaigns."
- Salesperson:
    - Content: "Our AI marketing solution can help you optimize your marketing campaigns and reach your target audience more effectively."

### Input
- Role: "Customer"
    - Content: """ + user_input

    start_time = time.time()
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
    outputs = model.generate(**input_ids, max_length=400, num_beams=5, renormalize_logits=True)
    result = tokenizer.decode(outputs[0])
    inference_time = time.time() - start_time
    return result, f"Inference time: {inference_time:.2f} seconds"

# Setup model and tokenizer
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b-it",
    device_map="auto",
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