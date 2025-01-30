import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

# Word2VecDataset for CBOW
import torch.nn.functional as F

class Word2VecDataset(Dataset):
    def __init__(self, corpus, context_window, word_to_idx, idx_to_word):
        self.context_window = context_window
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.max_context_size = 2 * context_window
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

                if context_idx:
                    data.append((context_idx, target_idx))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        context_tensor = torch.tensor(context, dtype=torch.long)

        # Pad or truncate context to the maximum context size
        padded_context = F.pad(context_tensor,
                               (0, self.max_context_size - len(context_tensor)),  # Pad at the end
                               value=0)[:self.max_context_size]  # Truncate if necessary

        return padded_context, torch.tensor(target, dtype=torch.long)



# Word2VecModel for CBOW
class Word2VecModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context):
        aggregated = self.embedding(context).mean(dim=1)
        output = self.output_layer(aggregated)
        return output


# Training function
def train(model, dataloader, criterion, optimizer, epochs):
    model.train()
    losses = []

    for epoch in tqdm(range(epochs)):
        total_loss = 0
        for context, target in dataloader:
            target = target.long()
            context = context.long()

            optimizer.zero_grad()
            predictions = model(context)
            loss = criterion(predictions, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        # print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

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

if __name__ == "__main__":

    file_path = "../tokenized_data.json"
    with open(file_path, "r", encoding="utf-8") as file:
        corpus = json.load(file)

    vocab_file_path = "../vocabulary_86.txt"

    with open(vocab_file_path, "r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f]

    # Create mappings
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}

    # Hyperparameters
    context_window = 4
    embedding_dim = 400
    batch_size = 1024
    epochs = 200
    learning_rate = 0.001

    # Dataset and DataLoader
    dataset = Word2VecDataset(corpus, context_window, word_to_idx=word_to_idx, idx_to_word=idx_to_word)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model, loss, optimizer
    model = Word2VecModel(len(dataset.word_to_idx), embedding_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Save model checkpoint
    torch.save(model.state_dict(), "word2vec_checkpoint.pth")

    # Train the model
    losses = train(model, dataloader, criterion, optimizer, epochs)

    # Plot training loss
    plt.plot(range(1, epochs + 1), losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig("training_loss_task_2.png")
    plt.show()

    # Generate triplets
    triplets = get_triplets(model, dataset.word_to_idx, dataset.idx_to_word)
    for triplet in triplets[:20]:
        print("Word:", triplet[0])
        print("Similar:", triplet[1])
        print("Dissimilar:", triplet[2])
        print()