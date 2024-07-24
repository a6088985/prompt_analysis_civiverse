import os
import pandas as pd
import numpy as np
import torch
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from src.preprocessing import preprocess_and_split

def get_writable_directory():
    directories = ['./plots', '/path/to/plots']
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            return directory
        except OSError:
            continue
    raise Exception("No writable directory found.")

def save_visualization(fig, path):
    try:
        fig.write_image(path)
        print(f"Visualization saved to {path}")
    except Exception as e:
        print(f"Failed to save visualization to {path}: {e}")

def main():
    # Load the CSV file into a pandas dataframe
    file_path = 'data/final_positive.csv'
    df = pd.read_csv(file_path)
    print("Loaded data with shape:", df.shape)

    output_dir = get_writable_directory()
    print(f"Saving visualizations to {output_dir}")

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    print("Loaded SentenceTransformer model")

    df['specifier'] = df['specifier'].fillna('').astype(str)
    embeddings = sentence_model.encode(df['specifier'].tolist(), show_progress_bar=True)
    print("Generated embeddings")

    np.save(os.path.join(output_dir, 'embeddings_specifier.npy'), embeddings)
    topic_model = BERTopic()
    topics, probabilities = topic_model.fit_transform(df['specifier'], embeddings)
    np.save(os.path.join(output_dir, 'topics_specifier.npy'), topics)

    df['topic'] = topics
    df_filtered = df[df['topic'] != -1]
    all_topics = topic_model.get_topics()

    print("All topics:")
    for topic_id, words in all_topics.items():
        if topic_id != -1:
            print(f"Topic {topic_id}: {words}")

    top_30_topics = topic_model.get_topic_info().head(30)
    topic_words = {}
    seen_words = set()

    for topic in top_30_topics['Topic']:
        if topic != -1:
            words = topic_model.get_topic(topic)
            if words:
                word_list = [word[0] for word in words[:10]]
                word_tuple = tuple(word_list)
                if word_tuple not in seen_words:
                    seen_words.add(word_tuple)
                    topic_words[topic] = word_list

    for topic, words in topic_words.items():
        print(f"Topic {topic}: {words}")

if __name__ == "__main__":
    main()
