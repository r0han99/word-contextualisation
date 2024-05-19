from transformers import GPT2Tokenizer, GPT2LMHeadModel

def load_model(model_path):
    # Load the model and tokenizer from the saved directory
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    return model, tokenizer

def generate_text(model, tokenizer, input_word, max_length=50):
    # Prepare the prompt
    prompt = f"Word: {input_word} Sentence:"
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate output
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
    
    # Decode generated sequence to text
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Print the generated text
    print(text)

# Load the model
model_path = './final_model_v2'  # Ensure this path is correct
model, tokenizer = load_model(model_path)

# Test with an example word
test_word = 'transparency'
generate_text(model, tokenizer, test_word)
