import os
import pandas as pd
import spacy
from collections import Counter

# Load the pre-trained NER model
nlp = spacy.load("en_core_web_sm")

def process_batch(batch, entity_counter):
    for text in batch:
        doc = nlp(text)
        for ent in doc.ents:
            entity_counter[(ent.text, ent.label_)] += 1

def process_column(df, column_name, output_file_path, batch_size=500000):
    if column_name not in df.columns:
        raise ValueError(f"The CSV file does not contain a '{column_name}' column.")
    
    df[column_name] = df[column_name].fillna('').astype(str)
    entity_counter = Counter()
    num_batches = len(df) // batch_size + 1

    for i in range(num_batches):
        print(f"Processing batch {i+1} of {num_batches}...")
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df))
        batch = df[column_name][start_idx:end_idx]
        process_batch(batch, entity_counter)

    output_lines = [f"{text} {label} {count}" for (text, label), count in entity_counter.items()]

    with open(output_file_path, 'w') as f:
        for line in output_lines:
            f.write(line + "\n")

    print(f"Entity frequencies for {column_name} saved to {output_file_path}")

def main():
    file_path = 'data/period_4.csv'
    output_file_path = 'entity_frequencies_4.txt'

    df = pd.read_csv(file_path)
    process_column(df, 'prompt', output_file_path)

if __name__ == "__main__":
    main()
