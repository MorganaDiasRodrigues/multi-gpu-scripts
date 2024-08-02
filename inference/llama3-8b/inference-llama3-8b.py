import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
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
    messages = [
        { "role": "user", "content": "I'm looking for an AI solution to help me with my business."},
        { "role": "assistant", "content": "Our AI solution can help you with your business by providing insights into your data." },
        { "role": "user", "content": "I'm interested in having a chatbot for my website." },
        { "role": "assistant", "content": "Our chatbot solution can help you engage with your customers and provide support 24/7." },
        { "role": "user", "content": "I'm looking for an automated solution to help me with my marketing campaigns." },
        { "role": "assistant", "content": "Our AI marketing solution can help you optimize your marketing campaigns and reach your target audience more effectively." },
        { "role": "user", "content": user_input }
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, 
                                        add_generation_prompt=True)
    
    start_time = time.time()

    # Check if eos_token_id is defined, otherwise set to a default value
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = tokenizer.convert_tokens_to_ids("")

    terminators = [eos_token_id]
    
    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    result = tokenizer.decode(response, skip_special_tokens=True)
    inference_time = time.time() - start_time
    return result, f"Inference time: {inference_time:.2f} seconds"

# Setup model and tokenizer
base_model = "meta-llama/Meta-Llama-3-8B-Instruct"
torch_dtype = torch.float16
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)

print("🤗 Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config
)
print("🤗 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model)

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

demo.launch(share=True)
