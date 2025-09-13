import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import os
from datetime import datetime

# We load our custom model blueprint from pree.py
from pree import PreeForCausalLM

def main():
    # --- 1. Load your PRE-TRAINED model and tokenizer ---
    model_path = "./pree-code-llm-final"
    print(f"[{datetime.now()}] Loading the base model you trained from: {model_path}")

    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = PreeForCausalLM.from_pretrained(model_path)
    print(f"[{datetime.now()}] Base model loaded successfully.")

    # --- 2. Load the High-Quality Dataset from your LOCAL FILE ---
    # The path should match the folder and filename you created.
    local_dataset_path = "./CodeAlpaca-20k/code_alpaca_20k.json"
    print(f"[{datetime.now()}] Loading the CodeAlpaca dataset from local file: {local_dataset_path}")
    
    # --- The Definitive Fix ---
    # We now load the dataset from the JSON file you downloaded.
    # This is fast, reliable, and requires no internet or authentication.
    if not os.path.exists(local_dataset_path):
        raise FileNotFoundError(
            f"Dataset file not found at {local_dataset_path}. "
            "Please ensure the file is in the 'CodeAlpaca-20k' folder."
        )
    
    dataset = load_dataset("json", data_files=local_dataset_path, split="train")
    # --- End of Fix ---

    # --- 3. Format the dataset ---
    # The JSON file has 'instruction', 'input', and 'output' columns.
    def format_prompt(example):
        # We combine instruction and input for a more robust prompt.
        if example.get("input"):
            instruction = f"{example['instruction']}\n\nInput:\n{example['input']}"
        else:
            instruction = example['instruction']
            
        return {
            "text": f"### Instruction:\n{instruction}\n\n### Response:\n{example['output']}"
        }

    print(f"[{datetime.now()}] Formatting the dataset...")
    formatted_dataset = dataset.map(format_prompt)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=384)
        
    print(f"[{datetime.now()}] Tokenizing the formatted dataset...")
    tokenized_dataset = formatted_dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    # --- 4. Configure and Run the Fine-Tuning Trainer ---
    output_dir = "./pree-code-llm-finetuned"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        save_strategy="epoch",
        learning_rate=1e-5,
        fp16=True,
        optim="adamw_8bit",
        gradient_checkpointing=True,
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    print(f"[{datetime.now()}] Starting fine-tuning...")
    trainer.train()
    print(f"[{datetime.now()}] Fine-tuning complete.")

    # --- 5. Save the new, much smarter model ---
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[{datetime.now()}] Your new fine-tuned model is saved to: {output_dir}")

if __name__ == "__main__":
    main()

