import torch
import torch.nn as nn
import torch.optim as optim
from torch.ao.nn.quantized import Dropout
from torch.nn.functional import dropout
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import torch.nn.functional
import task1

vocabulary_size = 5000
context_window = 2
embedding_dim = 500
batch_size = 1024
epochs = 40
lr = 0.001
dropout_rate = 0
word_index_mapping, index_word_mapping = {}, {}
train_dataloader = []
val_dataloader = []

class Word2VecDataset(Dataset):
    def __init__(self, corpus):
        self.preprocessed_data = []
        self.corpus = corpus

    def __len__(self):
        return len(self.preprocessed_data)

    def __getitem__(self, index):
        target, context = self.preprocessed_data[index]
        context_tensor = torch.tensor(context, dtype=torch.long)

        total_padding = 2 * context_window - len(context_tensor)

        left_padding = total_padding // 2
        right_padding = total_padding - left_padding

        context = torch.nn.functional.pad(context_tensor, (left_padding, right_padding), value=0)
        return torch.tensor(target, dtype=torch.long), context

    def preprocess_data(self, word_index_mapping):
        data = []
        for key in self.corpus:
            tokens = self.corpus[key]

            n = len(tokens)

            for i in range(0, n):
                target = tokens[i]

                # if target not in word_index_mapping:
                #     print(f"Warning: '{target}' not found in vocabulary, skipping.")
                #     continue  # Skip this word

                tokens_before_word = tokens[max(0, i - context_window):i]
                tokens_after_word = tokens[i + 1: min(n + 1, i + 1 + context_window)]

                context = tokens_before_word + tokens_after_word

                target_index = word_index_mapping[target]
                context_index = []
                for word in context:
                    if word in word_index_mapping:
                        context_index.append(word_index_mapping[word])

                if len(context_index) > 0:
                    data.append((target_index, context_index))

        self.preprocessed_data = data


class Word2VecModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(embedding_dim, vocab_size)

        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, context):
        embedded = self.embedding(context)  # [batch_size, context_length, embedding_dim]
        aggregated = embedded.mean(dim=1)     # [batch_size, embedding_dim]
        aggregated = self.dropout(aggregated)   # Apply dropout
        out = self.linear(aggregated)           # [batch_size, vocab_size]
        return out

    def train_model(self, model, criterion, optimizer):
        loss_list, val_loss = [], []

        for _ in tqdm(range(epochs)):
            total_loss = 0
            model.train()  # enable this if we r able to implement some dropout thingy

            for target, context in train_dataloader:
                target = target.long()
                context = context.long()

                pred = model.forward(context)
                loss = criterion(pred, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            number_of_samples = len(train_dataloader)
            avg_loss = total_loss / number_of_samples
            loss_list.append(avg_loss)

            total_loss = 0
            model.eval()

            with torch.no_grad():
                for target, context in val_dataloader:
                    target = target.long()
                    context = context.long()

                    pred = model.forward(context)
                    loss = criterion(pred, target)
                    total_loss += loss.item()

            avg_val_loss = total_loss / len(val_dataloader)
            val_loss.append(avg_val_loss)

        return loss_list, val_loss

    # def get_triplets(self, model):
    #     embeddings = model.network[0].weight.data.cpu().numpy()
    #     similarities = cosine_similarity(embeddings)
    #     triplets = []
    #
    #     for item in word_index_mapping:
    #         word, index = item[0], item[1]
    #
    #         similar, triplet = [], []
    #         similar_indices = np.argsort(similarities[index])[::-1]
    #         similar_indices = similar_indices[:3]
    #
    #         for i in similar_indices:
    #             similar.append(index_word_mapping[i])
    #         del similar[index_word_mapping[index]]
    #
    #         dissimilar_index = np.argsort(similarities[idx])[0]
    #         dissimilar = index_word_mapping[dissimilar_index]
    #
    #         triplet = [word, similar, dissimilar]
    #         triplets.append(triplet)
    #
    #     for triplet in triplets:
    #         print("Word:", triplet[0])
    #         print("Similar:", triplet[1])
    #         print("Dissimilar:", triplet[2])
    #         print()
    def get_triplets(self):
        embeddings = self.embedding.weight.data.cpu().numpy()
        similarities = cosine_similarity(embeddings)

        triplets = []

        for word, index in word_index_mapping.items():
            similar = []

            similar_indices = np.argsort(similarities[index])[::-1]
            similar_indices = [i for i in similar_indices if i != index][:3]

            for i in similar_indices:
                similar.append((index_word_mapping[i], similarities[index][i]))

            dissimilar_index = np.argsort(similarities[index])[0]
            dissimilar = (index_word_mapping[dissimilar_index], similarities[index][dissimilar_index])

            triplets.append([word, similar, dissimilar])

        for triplet in triplets:
            print("Word:", triplet[0])
            print("Similar:", [(w, round(sim, 4)) for w, sim in triplet[1]])
            print("Dissimilar:", (triplet[2][0], round(triplet[2][1], 4)))
            print()

        return triplets


def get_data(vocab_size, split=0.9):
    task1.make_vocab_and_tokenize(vocab_size)

    file_path = "tokenized_data.json"
    with open(file_path, "r", encoding="utf-8") as file:
        corpus = json.load(file)

    vocab_file_path = "vocabulary_86.txt"

    with open(vocab_file_path, "r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f]

    word_index_mapping = {word: idx for idx, word in enumerate(vocab)}
    index_word_mapping = {idx: word for word, idx in word_index_mapping.items()}
    dataset = Word2VecDataset(corpus)
    dataset.preprocess_data(word_index_mapping)

    n = len(dataset)
    train = int(n * split)
    val = n - train

    train_dataset, val_dataset = random_split(dataset, [train, val])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader, word_index_mapping, index_word_mapping


def plot(loss_list, val_loss):
    plt.plot(range(1, epochs + 1), loss_list, label="Training Loss")
    plt.plot(range(1, epochs + 1), val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig("task_2.png")
    plt.legend()
    plt.grid(visible=True)
    plt.show()

def get_triplet_for_word(self, word):
    if word not in word_index_mapping:
        print(f"Word '{word}' not found in vocabulary.")
        return None

    index = word_index_mapping[word]
    embeddings = self.embedding.weight.data.cpu().numpy()
    similarities = cosine_similarity(embeddings)

    # Get sorted indices of words based on similarity (descending order)
    similar_indices = np.argsort(similarities[index])[::-1]

    # Exclude the word itself (index = `index`)
    similar_indices = [i for i in similar_indices if i != index][:3]

    similar_words = [(index_word_mapping[i], similarities[index][i]) for i in similar_indices]

    # Find the most dissimilar word (smallest similarity score)
    dissimilar_index = np.argsort(similarities[index])[0]  # Least similar word
    dissimilar_word = index_word_mapping[dissimilar_index]
    dissimilar_similarity = similarities[index][dissimilar_index]

    # Print results
    print(f"Word: {word}")
    print("Similar Words (with Cosine Similarity):")
    for sim_word, sim_value in similar_words:
        print(f"  {sim_word}: {sim_value:.4f}")

    print(f"Dissimilar Word: {dissimilar_word} (Cosine Similarity: {dissimilar_similarity:.4f})\n")

    return word, similar_words, (dissimilar_word, dissimilar_similarity)

def run_Word2Vec(vocabulary_size_ = 14000, context_window_ = 2,embedding_dim_ = 400,batch_size_ = 1024,epochs_ = 50,lr_ = 0.001, dropout_rate_ = 0.3):

    global train_dataloader, val_dataloader, word_index_mapping, index_word_mapping, vocabulary_size, context_window, embedding_dim, batch_size, epochs, lr, dropout_rate

    vocabulary_size = vocabulary_size_
    context_window = context_window_
    embedding_dim = embedding_dim_
    batch_size = batch_size_
    epochs = epochs_
    lr = lr_
    dropout_rate = dropout_rate_
    train_dataloader, val_dataloader, word_index_mapping, index_word_mapping = get_data(vocabulary_size)

    print("Loaded Data")

    model = Word2VecModel(vocabulary_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    print("Training...")

    loss_list, val_loss = model.train_model(model, criterion, optimizer)
    plot(loss_list, val_loss)

    torch.save(model.state_dict(), "word2vec_checkpoint.pth")

    model.get_triplets()

    get_triplet_for_word(model, "happy")
    get_triplet_for_word(model, "sad")
    get_triplet_for_word(model, "punish")

if __name__ == "__main__":
    run_Word2Vec( vocabulary_size_= vocabulary_size, context_window_= context_window, embedding_dim_= embedding_dim, batch_size_= batch_size, epochs_= epochs, lr_= lr, dropout_rate_=dropout_rate)
