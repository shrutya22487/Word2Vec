import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import json

# Word2VecDataset class
class Word2VecDataset(Dataset):
    def __init__(self, corpus, context_window, word_to_idx, idx_to_word):
        self.context_window = context_window
        self.word_to_idx = word_to_idx  # Use prebuilt word-to-index mapping
        self.idx_to_word = idx_to_word  # Use prebuilt index-to-word mapping
        self.data = self.preprocess_data(corpus)

    def preprocess_data(self, corpus):
        data = []
        for key in corpus:
            tokens = corpus[key]
            for i in range(len(tokens)):
                target = tokens[i]
                context = tokens[max(0, i - self.context_window):i] + tokens[i + 1:i + 1 + self.context_window]
                target_idx = self.word_to_idx[target]
                context_idx = [self.word_to_idx[word] for word in context if word in self.word_to_idx]
                for ctx in context_idx:
                    data.append((target_idx, ctx))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Word2VecModel class
class Word2VecModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2VecModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, target):
        embed = self.embedding(target)
        output = self.output_layer(embed)
        return output


# Training function
def train(model, dataloader, criterion, optimizer, epochs):
    model.train()
    losses = []

    for epoch in range(epochs):
        total_loss = 0
        for target, context in dataloader:
            target = target.long()
            context = context.long()

            optimizer.zero_grad()
            predictions = model(target)
            loss = criterion(predictions, context)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

    return losses

# Cosine similarity utility
def get_triplets(model, word_to_idx, idx_to_word, top_n=3):
    embeddings = model.embedding.weight.data.cpu().numpy()
    similarities = cosine_similarity(embeddings)
    triplets = []

    for word, idx in word_to_idx.items():
        similar_indices = np.argsort(similarities[idx])[::-1][:top_n + 1]
        dissimilar_index = np.argsort(similarities[idx])[0]

        similar_words = [idx_to_word[i] for i in similar_indices if i != idx][:top_n]
        dissimilar_word = idx_to_word[dissimilar_index]
        triplets.append((word, similar_words, dissimilar_word))

    return triplets

def check_word_similarity(model, word_to_idx, idx_to_word, word, top_n=3):
    if word not in word_to_idx:
        print(f"The word '{word}' is not in the vocabulary.")
        return

    embeddings = model.embedding.weight.data.cpu().numpy()
    similarities = cosine_similarity(embeddings)

    word_idx = word_to_idx[word]
    similar_indices = np.argsort(similarities[word_idx])[::-1][:top_n + 1]
    dissimilar_index = np.argsort(similarities[word_idx])[0]

    similar_words = [idx_to_word[i] for i in similar_indices if i != word_idx][:top_n]
    dissimilar_word = idx_to_word[dissimilar_index]

    print(f"Word: {word}")
    print(f"Similar Words: {similar_words}")
    print(f"Dissimilar Word: {dissimilar_word}")


if __name__ == "__main__":

    file_path = "../tokenized_data.json"
    with open(file_path, "r", encoding="utf-8") as file:
        corpus = json.load(file)

    vocab_file_path = "../vocabulary_86.txt"

    with open(vocab_file_path, "r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f]  # Read each line and strip whitespace

    # Create mappings
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}

    # Hyperparameters
    context_window = 3
    embedding_dim = 50
    batch_size = 16
    epochs = 250
    learning_rate = 0.003

    # Dataset and DataLoader
    dataset = Word2VecDataset(corpus, context_window, word_to_idx=word_to_idx, idx_to_word=idx_to_word)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model, loss, optimizer
    model = Word2VecModel(len(dataset.word_to_idx), embedding_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    losses = train(model, dataloader, criterion, optimizer, epochs)

    # Save model checkpoint
    torch.save(model.state_dict(), "word2vec_checkpoint.pth")

    # Plot training loss
    plt.plot(range(1, epochs + 1), losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()

    # Generate triplets
    triplets = get_triplets(model, dataset.word_to_idx, dataset.idx_to_word)
    for triplet in triplets[:5]:
        print("Word:", triplet[0])
        print("Similar:", triplet[1])
        print("Dissimilar:", triplet[2])
        print()

    # Check similar and dissimilar words for a specific word
    word_to_check = "happy"  # Replace with your word of choice
    check_word_similarity(model, dataset.word_to_idx, dataset.idx_to_word, word_to_check, top_n=3)

