import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_text(prompt, model_path, max_length=512, temperature=0.7):
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    model.eval()
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.1
        )
    
    # Decode and return
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    model_path = "/home/fuljanic/gemma2/output_hrcak_2/model_epoch_2"
    prompt = "Bacterial resistance to antibiotics is"  # Replace with your prompt
    
    generated_text = generate_text(prompt, model_path)
    print(f"Prompt: {prompt}")
    print(f"Generated text:\n{generated_text}")