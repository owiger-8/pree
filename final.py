import torch
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import os
from datetime import datetime

# It imports the model blueprint from your 'pree.py' file
from pree import PreeConfig, PreeForCausalLM

# --- Tokenizer Training (This function remains the same) ---
def train_tokenizer(dataset, vocab_size=65536):
    tokenizer_path = "./pree-tokenizer"
    if os.path.exists(os.path.join(tokenizer_path, "tokenizer.json")):
        print(f"[{datetime.now()}] tokenizer.json already exists. Skipping training.")
        return
    
    print(f"[{datetime.now()}] Training a new tokenizer...")
    tokenizer = ByteLevelBPETokenizer()
    def batch_iterator(batch_size=1000):
        for i in range(0, len(dataset), batch_size):
            yield dataset[i : i + batch_size]["text"]
    tokenizer.train_from_iterator(
        batch_iterator(), vocab_size=vocab_size, min_frequency=2,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
    )
    os.makedirs(tokenizer_path, exist_ok=True)
    
    tokenizer_file_path = os.path.join(tokenizer_path, "tokenizer.json")
    tokenizer.save(tokenizer_file_path)
    
    print(f"[{datetime.now()}] Tokenizer training complete and saved to {tokenizer_file_path}.")

# --- 2. Dataset Preparation ---
def prepare_dataset(block_size=384): # We are keeping the smaller block size
    print(f"[{datetime.now()}] Loading and preparing dataset...")

    auth_token = "" #enter your token here
    

    dataset_stream = load_dataset(
        "bigcode/the-stack-dedup", 
        data_dir="data/python", split="train", streaming=True, token=auth_token
    )
    
    from datasets import Dataset
    tokenizer_path = "./pree-tokenizer"
    if not os.path.exists(os.path.join(tokenizer_path, "tokenizer.json")):
        print(f"[{datetime.now()}] tokenizer.json not found. Training on a 50k sample...")
        tokenizer_sample = list(dataset_stream.take(50000))
        dataset_for_tokenizer = Dataset.from_dict({"text": [item['content'] for item in tokenizer_sample]})
        train_tokenizer(dataset_for_tokenizer)
    
    print(f"[{datetime.now()}] Loading trained tokenizer into a Transformers-compatible object...")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(tokenizer_path, "tokenizer.json"),
        bos_token="<s>", eos_token="</s>", pad_token="<pad>",
        unk_token="<unk>", mask_token="<mask>",
    )
    
    def tokenize_function(examples):
        return tokenizer(
            examples["content"], truncation=True, max_length=block_size,
        )
    
    print(f"[{datetime.now()}] Taking a 50k sample for the smoke test run...")
    dataset_sample = list(dataset_stream.take(50000))
    main_dataset = Dataset.from_list(dataset_sample)
    
    print(f"[{datetime.now()}] Tokenizing the dataset sample...")
    tokenized_dataset = main_dataset.map(
        tokenize_function, batched=True, remove_columns=list(main_dataset.features)
    )

    print(f"[{datetime.now()}] Dataset preparation complete.")
    return tokenized_dataset, tokenizer

if __name__ == "__main__":
    tokenized_dataset, tokenizer = prepare_dataset(block_size=384)

    print(f"[{datetime.now()}] Initializing model...")
    config = PreeConfig(
        vocab_size=tokenizer.vocab_size, bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
    )
    model = PreeForCausalLM(config)
    print(f"[{datetime.now()}] Model created with ~{sum(p.numel() for p in model.parameters())/1e9:.2f}B parameters.")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print(f"[{datetime.now()}] Configuring training arguments...")
    training_args = TrainingArguments(
        output_dir="./pree-code-llm", overwrite_output_dir=True,
        num_train_epochs=1, per_device_train_batch_size=1,
        gradient_accumulation_steps=8, save_steps=1000,
        save_total_limit=2, prediction_loss_only=True,
        learning_rate=2e-5, fp16=True,
        # --- START OF THE FIX ---
        # This is the final optimization.
        optim="adamw_8bit",
        # --- END OF THE FIX ---
        gradient_checkpointing=True, logging_steps=100,
    )

    trainer = Trainer(
        model=model, args=training_args,
        data_collator=data_collator, train_dataset=tokenized_dataset,
    )

    print(f"[{datetime.now()}] Starting training...")
    trainer.train()
    print(f"[{datetime.now()}] Training complete.")

    trainer.save_model("./pree-code-llm-final")
    tokenizer.save_pretrained("./pree-code-llm-final")
    print(f"[{datetime.now()}] Final model and tokenizer saved.")

