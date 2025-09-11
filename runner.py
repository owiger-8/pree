import torch
from transformers import PreTrainedTokenizerFast
from pree import PreeForCausalLM

def main():
    model_path = "./pree-code-llm-final"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model and tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    model = PreeForCausalLM.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    print("Model loaded. Enter prompts (type 'quit' to exit):")
    
    while True:
        prompt = input("\nPrompt: ").strip()
        if prompt.lower() in ['quit', 'exit']:
            break
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        if 'token_type_ids' in inputs:
            del inputs['token_type_ids']
        
        # Generate with simpler settings
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,  # Reduced for testing
                do_sample=True,
                temperature=0.8,
                top_k=40,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode and print
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(generated_text)

if __name__ == "__main__":
    main()