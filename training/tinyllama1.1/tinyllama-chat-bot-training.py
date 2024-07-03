import gradio as gr
import torch
from datasets import load_dataset, Dataset, DatasetDict
from trl import setup_chat_format, SFTTrainer
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model
import time

# Download base model before launching the UI
base_model = "TinyLlama/TinyLlama-1.1B-Chat-v0.6"
torch_dtype = torch.float16
attn_implementation = "eager"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)

print("🤗 Loading the model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="balanced",
    attn_implementation=attn_implementation
)
tokenizer = AutoTokenizer.from_pretrained(base_model)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)
model = get_peft_model(model, peft_config)
model, tokenizer = setup_chat_format(model, tokenizer)
print("🤗 Model and tokenizer loaded successfully!")

# Function to get GPU information
def get_gpu_info():
    num_gpus = torch.cuda.device_count()
    gpu_info = []
    for gpus in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(gpus)
        gpu_memory = torch.cuda.get_device_properties(gpus).total_memory / 1e9
        gpu_info.append(f"GPU {gpus} name: {gpu_name}, memory: {gpu_memory} GB")
    return gpu_info

# Function to transform dataset
def transform_dataset(sales_dataset, tokenizer):
    def format_chat_template(conversations):
        formatted_conversations = {}
        for key, value in conversations.items():
            if value is None:
                continue
            if key in ['0', '2', '4', '6', '8', '10', '12', '14', '16', '18']:
                role = "user"
            else:
                role = "assistant"
            conversation_json = [{"role": role, "content": value.replace('Salesman: ', '').replace('Customer: ', '')}]
            formatted_conversations[key] = tokenizer.apply_chat_template(conversation_json, tokenize=False)
        return formatted_conversations

    formatted_dataset = []
    for conversation in sales_dataset['train']:
        formatted_conversation = format_chat_template(conversation)
        combined_dialogue = ''.join([v for v in formatted_conversation.values() if v is not None])
        formatted_dataset.append({"text": combined_dialogue})
    
    return Dataset.from_pandas(pd.DataFrame(formatted_dataset))

# Function to run training
def run_training(number_of_dialogues, train_batch_size, eval_batch_size):
    sales_dataset = load_dataset('goendalf666/sales-conversations')
    formatted_sales_dataset = transform_dataset(sales_dataset, tokenizer)
    dataset_dict = DatasetDict({"train": formatted_sales_dataset})
    dataset = dataset_dict["train"].select(range(number_of_dialogues)).train_test_split(test_size=0.1)

    training_arguments = TrainingArguments(
        output_dir="tinyllama-sales-chatbot",
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        num_train_epochs=1,
        evaluation_strategy="steps",
        eval_steps=0.2,
        logging_steps=100,
        warmup_steps=10,
        logging_strategy="steps",
        learning_rate=2e-4,
        fp16=False,
        bf16=False,
        group_by_length=True
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        dataset_text_field="text",
        peft_config=peft_config,
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
    )

    print("🚀 Training the model...")
    start_time = time.time()
    train_output = trainer.train()

    # Extract relevant metrics
    metrics = train_output.metrics
    train_runtime = metrics['train_runtime']
    train_steps_per_second = metrics['train_steps_per_second']
    global_step = train_output.global_step

    # Calculate the total number of steps for 3000 datapoints
    current_datapoints = number_of_dialogues
    total_datapoints = 3000
    total_steps = (total_datapoints * global_step) / current_datapoints

    # Calculate the time for the total steps
    estimated_time_seconds = total_steps / train_steps_per_second
    estimated_time_minutes = estimated_time_seconds / 60
    total_time = time.time() - start_time

    return {
        "train_runtime": train_runtime,
        "train_steps_per_second": train_steps_per_second,
        "global_step": global_step,
        "total_time": total_time,
        "estimated_time_seconds": estimated_time_seconds,
        "estimated_time_minutes": estimated_time_minutes
    }

# Gradio interface
gpu_info = get_gpu_info()
gpu_info_str = "\n".join(gpu_info)

def infer(number_of_dialogues, train_batch_size, eval_batch_size):
    result = run_training(number_of_dialogues, train_batch_size, eval_batch_size)
    return (
        f"Training completed in {result['train_runtime']:.2f} seconds\n"
        f"Estimated training time for 3000 datapoints: {result['estimated_time_seconds']:.2f} seconds "
        f"({result['estimated_time_minutes']:.2f} minutes)\n"
        f"Total training time: {result['total_time']:.2f} seconds"
    )

with gr.Blocks() as demo:
    gr.Markdown("# Training Dashboard")
    with gr.Row():
        with gr.Column():
            gr.Markdown("## GPU Information")
            gpu_info_box = gr.Textbox(value=gpu_info_str, interactive=False, lines=5)
        with gr.Column():
            gr.Markdown("## Training Configuration")
            number_of_dialogues = gr.Number(label="Number of dialogues", value=50)
            train_batch_size = gr.Number(label="Training batch size", value=1)
            eval_batch_size = gr.Number(label="Evaluation batch size", value=1)
            submit_button = gr.Button("Train")
    
    training_output_box = gr.Textbox(label="Training Output", interactive=False, lines=10)

    submit_button.click(infer, inputs=[number_of_dialogues, train_batch_size, eval_batch_size], outputs=training_output_box)

demo.launch(share=True)
