from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import pandas as pd
import torch


# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load pre-trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Set the pad token
tokenizer.pad_token = tokenizer.eos_token


# prepared_data = prepare_data(data)
# Example function to prepare data
def prepare_data(df):
    formatted_data = []
    for _, row in df.iterrows():
        for i in range(1, 9):  # Loop through Usage 1 to Usage 8
            usage_col = f'Usage {i}'
            if usage_col in row and not pd.isna(row[usage_col]):
                text = f"Word: {row['index']} Sentence: {row[usage_col]}"
                encoded = tokenizer.encode_plus(
                    text,
                    add_special_tokens=True,
                    max_length=512,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                # Ensure input_ids are also used as labels
                encoded['labels'] = encoded['input_ids'].clone()
                formatted_data.append(encoded)
    return formatted_data

# Prepare the dataset
data = pd.read_csv("./g1_usage_extended.csv")  # Assuming you have extended data in g1_extended_data.csv
prepared_data = prepare_data(data)



# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=prepared_data,  # This should be your tokenized and formatted dataset
)

# Train the model
trainer.train()

# Save the final model and tokenizer
model.save_pretrained('./final_model_v2')
tokenizer.save_pretrained('./final_model_v2')
