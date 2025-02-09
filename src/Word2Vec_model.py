import random
import torch
import torch.nn as nn
import torch.optim
from torch.ao.nn.quantized import Dropout
from torch.nn.functional import dropout
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import torch.nn.functional
import WordPieceTokeniser

################################# HYPERPARAMETERS #########################################
vocabulary_size = 14000
context_window = 2
embedding_dim = 300
batch_size = 1024
epochs = 50
lr = 0.001
dropout_rate = 0
word_index_mapping, index_word_mapping = {}, {}
train_dataloader = []
val_dataloader = []


################################# DATASET LOADING #########################################

class Word2VecDataset(Dataset):
    """"
    Is responsible for loading the dataset and provides functions to load, process and get items at a specific index
    """
    # Reference -> https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    # Initialize
    def __init__(self, corpus):
        self.preprocessed_data = []
        self.corpus = corpus

    # returns the number of items in data
    def __len__(self):
        return len(self.preprocessed_data)

    # gets the item at a particular index and returns the target and context tensor after performing padding
    # note that padding is done using the [PAD] token generated in task 1
    def __getitem__(self, index):
        target = self.preprocessed_data[index][0]
        context = torch.tensor(self.preprocessed_data[index][1], dtype=torch.long)

        total_padding = 2 * context_window - len(context)

        left_padding = total_padding // 2
        right_padding = total_padding - left_padding

        # Reference -> https://pytorch.org/docs/stable/nn.functional.html
        context = torch.nn.functional.pad(context, (left_padding, right_padding), value=0)
        return torch.tensor(target, dtype=torch.long), context

    # finds context words for all the words in the corpus
    def preprocess_data(self, word_index_mapping):
        data = []
        for key in self.corpus:
            tokens = self.corpus[key]

            n = len(tokens)

            for i in range(0, n):
                target = tokens[i]

                # was giving for some words randomly thats why I added
                if target not in word_index_mapping:
                    print(target,"not found in vocabulary")
                    continue

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

################################# MODEL #########################################

class Word2VecModel(nn.Module):
    # Reference -> https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
    # Make the layers and Initilaize them
    def __init__(self, vocab_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Embedding(vocab_size, embedding_dim),
            nn.Linear(embedding_dim, vocab_size)
        )

        nn.init.xavier_uniform_(self.network[0].weight)
        nn.init.xavier_uniform_(self.network[1].weight)
        if self.network[1].bias is not None:
            nn.init.zeros_(self.network[1].bias)

    # pushes the data forward to make predictions
    def forward(self, context):
        embedded = self.network[0](context).mean(dim = 1)
        out = self.network[1](embedded)
        return out

    # Main training loop
    def train_model(self, model, criterion, optimizer):
        loss_list, val_loss = [], []
        # Reference -> https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html
        for _ in tqdm(range(epochs)):
            total_loss = 0
            model.train()  # enable this if we r able to implement some dropout thingy, this enables the training capability of the model

            for target, context in train_dataloader:

                # target = target.long()
                # context = context.long()

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
            model.eval() # disable sthe dropout

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


    def get_triplets(self):

        """"
        Will generate 5 random triplets and show the similarities as well
        remember to reound them off
        """
        embeddings = self.network[0].weight.data.cpu().numpy()

        #Reference -> https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
        cos_similarity_mat = cosine_similarity(embeddings)
        indexes = []
        for i in range(5):
            indexes.append(random.randint(1, len(word_index_mapping)))
        triplets = []
        for word, index in word_index_mapping.items():
            if index in indexes:
                similar, similar_indices_list = [], []

                similar_indices = np.argsort(cos_similarity_mat[index])[::-1]

                for i in similar_indices:
                    if i != index:
                        similar_indices_list.append(i)

                similar_indices = similar_indices_list[:3]

                for i in similar_indices:
                    similar.append([index_word_mapping[i], cos_similarity_mat[index][i]])

                dissimilar_index = np.argsort(cos_similarity_mat[index])[0]
                dissimilar = (index_word_mapping[dissimilar_index], cos_similarity_mat[index][dissimilar_index])

                triplet = [word, similar, dissimilar]
                triplets.append(triplet)

        for triplet in triplets:
            print(triplet[0], "\n")
            print("similar words:")
            for i in range(len(triplet[1])):
                print("word: " ,triplet[1][i][0]," ", "with similarity: ", triplet[1][i][1])
            print("Dissimilar:", triplet[2][0], triplet[2][1] , "\n")

# main data function, loads data from files and invokes the dataloader
def get_data(vocab_size, split=0.9):
    task1.make_vocab_and_tokenize(vocab_size)

    file_path = "../Data/tokenized_data.json"
    with open(file_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    vocab_file_path = "../Data/vocabulary_86.txt"

    with open(vocab_file_path, "r", encoding="utf-8") as f:
        vocab = []
        for line in f:
            vocab.append(line.strip())
    word_index_mapping,index_word_mapping = {},{}

    for i, j in enumerate(vocab):
        word_index_mapping[j] = i
    for i, j in word_index_mapping.items():
        index_word_mapping[j] = i

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
    """
    Helper function to just plot the graphs
    """
    plt.plot(range(1, epochs + 1), loss_list, label="Training Loss")
    plt.plot(range(1, epochs + 1), val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig("task_2.png")
    plt.legend()
    plt.grid(visible=True)
    plt.show()

def get_triplet_for_word(model,word):

    """"
    to find triplets for a specific word
    """


    index_of_word = word_index_mapping[word]
    embeddings = model.network[0].weight.data.cpu().numpy()

    #Reference -> https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html

    cos_similarity_mat = cosine_similarity(embeddings)

    triplets = []

    similar_indices = np.argsort(cos_similarity_mat[index_of_word])[::-1]
    similar_indices_list, similar = [], []

    for i in similar_indices:
        if i != index_of_word:
            similar_indices_list.append(i)

    similar_indices = similar_indices_list[:4]

    for i in similar_indices:
        similar.append([index_word_mapping[i], cos_similarity_mat[index_of_word][i]])

    dissimilar_index = np.argsort(cos_similarity_mat[index_of_word])[0]
    dissimilar = (index_word_mapping[dissimilar_index], cos_similarity_mat[index_of_word][dissimilar_index])

    triplet = [word, similar, dissimilar]
    triplets.append(triplet)
    for triplet in triplets:
        print(triplet[0], "\n")
        print("similar words:")
        for i in range(len(triplet[1])):
            print("word: ", triplet[1][i][0], " ", "with similarity: ", triplet[1][i][1])
        print("Dissimilar:", triplet[2][0], triplet[2][1], "\n")


def run_Word2Vec(vocabulary_size_ = 14000, context_window_ = 2,embedding_dim_ = 400,batch_size_ = 1024,epochs_ = 50,lr_ = 0.001, dropout_rate_ = 0.3):

    """"
    Pipelining functions
    1. makes the vocabulary and tokenized dataset from task1
    2. makes the dataset
    3. trains and saves and evaluates the model
    4. gives triplets for similarity and dissimilarities
    """
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

    #Reference -> https://pytorch.org/docs/stable/optim.html
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("Training...")

    loss_list, val_loss = model.train_model(model, criterion, optimizer)
    plot(loss_list, val_loss)

    torch.save(model.state_dict(), "word2vec_checkpoint.pth")

    model.get_triplets()
    #
    get_triplet_for_word(model, "happy")
    get_triplet_for_word(model, "sad")
    get_triplet_for_word(model, "punish")
    get_triplet_for_word(model, "politics")
    get_triplet_for_word(model, "him")

if __name__ == "__main__":
    run_Word2Vec( vocabulary_size_= vocabulary_size, context_window_= context_window, embedding_dim_= embedding_dim, batch_size_= batch_size, epochs_= epochs, lr_= lr, dropout_rate_=dropout_rate)
