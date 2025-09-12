import torch
from transformers import PreTrainedTokenizerFast
from pree import PreeForCausalLM, PreeConfig

def main():
    model_path = "./pree-code-llm-final"
    device = "cpu"  # Use CPU to avoid CUDA issues
    
    try:
        # Load tokenizer
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
        print("Tokenizer loaded successfully")
        
        # Try to load the model
        try:
            model = PreeForCausalLM.from_pretrained(model_path)
            print("Model loaded from pretrained")
        except:
            print("Could not load pretrained model, creating new one...")
            config = PreeConfig(vocab_size=tokenizer.vocab_size)
            model = PreeForCausalLM(config)
        
        model.to(device)
        model.eval()
        print("Model ready")
        
        while True:
            prompt = input("\nPrompt: ").strip()
            if prompt.lower() in ['quit', 'exit']:
                break
            
            # Simple tokenization without attention mask issues
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            
            # Very simple generation
            with torch.no_grad():
                try:
                    # Generate one token at a time to debug
                    for i in range(20):
                        outputs = model(input_ids)
                        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
                        input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=1)
                        
                        # Decode and print as we go
                        generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                        print(f"\r{generated_text}", end="", flush=True)
                        
                        if next_token.item() == tokenizer.eos_token_id:
                            break
                    print()  # New line after generation
                    
                except Exception as e:
                    print(f"\nError during generation: {e}")
                    break
                    
    except Exception as e:
        print(f"Error loading model: {e}")
        print("The model may not be properly trained or saved.")

if __name__ == "__main__":
    main()