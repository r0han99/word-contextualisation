#Example function to prepare data
def prepare_data(df):
    formatted_data = []
    for _, row in df.iterrows():
        for i in range(1, 4):  # Loop through Usage 1 to Usage 3
            text = f"Word: {row['index']} Sentence: {row[f'Usage {i}']}"
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
data = pd.read_csv("./g1_data.csv")