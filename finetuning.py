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

    # Load the tokenizer you already trained
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the 1B parameter model you spent two weeks training
    model = PreeForCausalLM.from_pretrained(model_path)
    print(f"[{datetime.now()}] Base model loaded successfully.")

    # --- 2. Load the High-Quality Fine-Tuning Dataset ---
    print(f"[{datetime.now()}] Loading the official CodeAlpaca dataset for fine-tuning...")
    
    # --- START OF THE DEFINITIVE FIX ---
    # Using the official, stable dataset you found. This will work.
    dataset = load_dataset("HuggingFaceH4/CodeAlpaca_20K", split="train")
    # --- END OF THE DEFINITIVE FIX ---

    # --- 3. Format the dataset into an instruction-following format ---
    # The dataset has columns 'prompt' and 'completion'
    def format_instruction(example):
        return {
            "text": f"### Instruction:\n{example['prompt']}\n\n### Response:\n{example['completion']}"
        }

    print(f"[{datetime.now()}] Formatting the dataset...")
    formatted_dataset = dataset.map(format_instruction)

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
