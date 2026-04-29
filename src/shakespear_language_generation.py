#%%
## Add imports here
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt

# Global constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 128


#####################################
# Low-level attention functions
#####################################

def initialise_projections(in_dim, out_dim):
    """
    Create linear projections for Q, K, or V.
    """
    return nn.Linear(in_dim, out_dim)


def pairwise_similarities(Q, K):
    """
    Compute dot product attention scores.
    Q: (batch, heads, seq_len, head_dim)
    K: (batch, heads, seq_len, head_dim)
    Returns: (batch, heads, seq_len, seq_len)
    """
    return torch.matmul(Q, K.transpose(-2, -1))


def attention_scaled(scores, head_dim):
    """
    Scale the raw attention scores.
    """
    return scores / math.sqrt(head_dim)


def attention_softmax(scores):
    """
    Normalize the scaled raw attention scores with softmax.
    """
    return F.softmax(scores, dim=-1)


def compute_outputs(attn_probs, V):
    """
    Get outputs as a weighted sum of values by attention scores.
    attn_probs: (batch, heads, seq_len, seq_len)
    V: (batch, heads, seq_len, head_dim)
    Returns: (batch, heads, seq_len, head_dim)
    """
    return torch.matmul(attn_probs, V)


def make_causal_mask(seq_len, device):
    """
    Create a mask matrix that masks future context for the attention.
    Returns a boolean mask of shape (seq_len, seq_len) where True indicates positions to mask.
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask


def apply_causal_mask(scores, mask):
    """
    Apply mask to attention scores.
    """
    scores = scores.masked_fill(mask, float('-inf'))
    return scores


def split_heads(x, num_heads):
    """
    Splitting the input across multiple heads.
    x: (batch, seq_len, embed_dim)
    Returns: (batch, num_heads, seq_len, head_dim)
    """
    batch, seq_len, embed_dim = x.size()
    head_dim = embed_dim // num_heads
    return x.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)


def merge_heads(x):
    """
    Reverses the splitting action.
    x: (batch, num_heads, seq_len, head_dim)
    Returns: (batch, seq_len, embed_dim)
    """
    batch, num_heads, seq_len, head_dim = x.size()
    embed_dim = num_heads * head_dim
    return x.transpose(1, 2).contiguous().view(batch, seq_len, embed_dim)


def self_attention(x, q_proj, k_proj, v_proj, out_proj, num_heads, dropout):
    """
    Self-attention block.
    """
    batch, seq_len, embed_dim = x.size()
    head_dim = embed_dim // num_heads

    # Create Q, K, V
    Q = q_proj(x)  # (batch, seq_len, embed_dim)
    K = k_proj(x)
    V = v_proj(x)

    # Split heads
    Q = split_heads(Q, num_heads)  # (batch, heads, seq_len, head_dim)
    K = split_heads(K, num_heads)
    V = split_heads(V, num_heads)

    # Compute attention scores
    scores = pairwise_similarities(Q, K)  # (batch, heads, seq_len, seq_len)
    scores = attention_scaled(scores, head_dim)

    # Apply causal mask (same for all examples/heads)
    mask = make_causal_mask(seq_len, x.device)
    scores = apply_causal_mask(scores, mask)

    # Softmax and dropout
    attn_probs = attention_softmax(scores)
    attn_probs = F.dropout(attn_probs, p=dropout, training=True)

    # Compute outputs
    attn_out = compute_outputs(attn_probs, V)

    # Merge heads and project
    out = merge_heads(attn_out)
    out = out_proj(out)
    return out, attn_probs


#####################################
# Data Preprocessing functions
#####################################

def pad_to_length(tokens, max_len, tokenizer):
    """
    Pad tokens to a fixed length using the <PAD> token.
    """
    pad_id = tokenizer["<PAD>"]
    if len(tokens) < max_len:
        tokens = tokens + [pad_id] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    return tokens


def tokenize(sentence, pad_to_len=None, tokenizer=None, include_stop=True):
    """
    Tokenize a sentence.
    """
    tokens = sentence.strip().split()
    if not include_stop and "<STOP>" in tokens:
        tokens = tokens[:tokens.index("<STOP>")]
    token_ids = [tokenizer.get(tok, tokenizer["<UNK>"]) for tok in tokens]
    if pad_to_len:
        token_ids = pad_to_length(token_ids, pad_to_len, tokenizer)
    return token_ids


def decode(tokens, tokenizer_inv, end_at_stop=True, omit_pad=True):
    """
    Decode tokens to text.
    """
    words = []
    for tok in tokens:
        word = tokenizer_inv.get(tok, "<UNK>")
        if omit_pad and word == "<PAD>":
            continue
        words.append(word)
        if end_at_stop and word == "<STOP>":
            break
    return " ".join(words)


from torch.utils.data import Dataset


class ShakespeareDataset(Dataset):
    def __init__(self, lines, tokenizer, max_len=MAX_LEN):
        self.data = ["<START> " + line.strip() + " <STOP>" for line in lines]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        token_ids = tokenize(self.data[idx], pad_to_len=self.max_len, tokenizer=self.tokenizer)
        return torch.tensor(token_ids, dtype=torch.long)


def load_and_preprocess_data():
    with open("../Data/shakespear_train.txt", "r") as f:
        lines_train = f.readlines()
    with open("../Data/shakespear_dev.txt", "r") as f:
        lines_dev = f.readlines()
    with open("./Data/shakespear_test.txt", "r") as f:
        lines_test = f.readlines()

    # Tokenize training lines to build vocabulary
    tokens_train = [line.strip().split() for line in lines_train]

    def flat(tokens):
        return [t for sublist in tokens for t in sublist]

    token_counts = Counter(flat(tokens_train))

    # Create vocabulary including special tokens
    vocab = {"<PAD>": 0, "<START>": 1, "<STOP>": 2, "<UNK>": 3}
    for token in token_counts:
        if token not in vocab:
            vocab[token] = len(vocab)
    tokenizer = vocab
    tokenizer_inv = {v: k for k, v in tokenizer.items()}

    # Prepare datasets using our custom Dataset class
    train_dataset = ShakespeareDataset(lines_train, tokenizer, max_len=MAX_LEN)
    val_dataset = ShakespeareDataset(lines_dev, tokenizer, max_len=MAX_LEN)

    return train_dataset, val_dataset, tokenizer, tokenizer_inv


#####################################
# Text Generation
#####################################

@torch.no_grad()
def evaluate_losses(data, model, tokenizer, bs=32, progress=True, pad_to_len=MAX_LEN):
    it = range(0, len(data), bs)
    if progress:
        it = tqdm(it)
    out = []
    for b_start in it:
        batch = [data[i] for i in range(b_start, min(b_start + bs, len(data)))]
        tokens = torch.stack([item for item in batch]).to(DEVICE)
        X_tokens, y_tokens = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
        model.eval()
        logits, _ = model(X_tokens)
        log_probs = F.log_softmax(logits, dim=-1)
        y_log_probs = torch.gather(log_probs, 2, y_tokens[..., None])[..., 0]
        for i in range(y_tokens.shape[0]):
            not_pad = y_tokens[i] != tokenizer["<PAD>"]
            loss = -y_log_probs[i, not_pad].mean()
            out.append(loss.item())
    return out


def generate_text(model, tokenizer, tokenizer_inv, context="<START>", gen_tokens=10, temperature=0.6):
    """
    Generate a fixed number of tokens using the trained model.
    """
    model.eval()
    # Tokenize the context (do not pad here)
    tokens = tokenize(context, pad_to_len=None, tokenizer=tokenizer, include_stop=False)
    tokens_tensor = torch.tensor(tokens, dtype=torch.long, device=DEVICE).unsqueeze(0)
    for _ in range(gen_tokens):
        logits, _ = model(tokens_tensor)
        # Focus on the last token's predictions
        logits = logits[0, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        # Sample from the distribution
        next_token = torch.multinomial(probs, num_samples=1).item()
        tokens_tensor = torch.cat([tokens_tensor, torch.tensor([[next_token]], device=DEVICE)], dim=1)
        # Stop if we generated a STOP token
        if next_token == tokenizer["<STOP>"]:
            break
    return decode(tokens_tensor.squeeze(0).tolist(), tokenizer_inv)
#%%
#####################################
# Model Definitions
#####################################

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout

    def forward(self, x):
        b, t, embed_dim = x.size()
        # Compute Q, K, V projections
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        # Split heads
        Q = split_heads(Q, self.num_heads)  # (b, heads, t, head_dim)
        K = split_heads(K, self.num_heads)
        V = split_heads(V, self.num_heads)
        # Compute scaled dot-product attention with causal mask
        scores = pairwise_similarities(Q, K)
        scores = attention_scaled(scores, self.head_dim)
        mask = make_causal_mask(t, x.device)
        scores = apply_causal_mask(scores, mask)
        attn_probs = attention_softmax(scores)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
        attn_out = compute_outputs(attn_probs, V)
        # Merge heads and project output
        out = merge_heads(attn_out)
        out = self.out_proj(out)
        return out, attn_probs


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        attn_out, attn_weights = self.mha(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x, attn_weights


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, MAX_LEN, embed_dim))
        self.layers = nn.ModuleList([TransformerBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, mean=0, std=0.02)
        nn.init.normal_(self.fc_out.weight, mean=0, std=0.02)
        if self.fc_out.bias is not None:
            nn.init.zeros_(self.fc_out.bias)
        for layer in self.layers:
            layer._init_weights()

    def forward(self, x):
        b, t = x.shape
        token_embeddings = self.embed(x)  # (b, t, embed_dim)
        pos_embeddings = self.pos_embed[:, :t, :]
        x = token_embeddings + pos_embeddings
        x = self.dropout(x)
        attn_weights_all = []
        for layer in self.layers:
            x, attn_weights = layer(x)
            attn_weights_all.append(attn_weights)
        logits = self.fc_out(x)
        return logits, attn_weights_all


#####################################
# Training function
#####################################

def train_model(model, train_dataset, val_dataset, tokenizer, tokenizer_inv, epochs=10, bs=32, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    val_losses = []

    # Convert datasets to lists for simple batching
    train_data = [train_dataset[i] for i in range(len(train_dataset))]
    val_data = [val_dataset[i] for i in range(len(val_dataset))]

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        # Shuffle training content
        random.shuffle(train_data)
        for i in range(0, len(train_data), bs):
            batch = train_data[i:i + bs]
            tokens = torch.stack(batch).to(DEVICE)  # shape: (bs, MAX_LEN)
            X = tokens[:, :-1]
            y = tokens[:, 1:]
            optimizer.zero_grad()
            logits, _ = model(X)
            loss = F.cross_entropy(
    logits.reshape(-1, logits.size(-1)),
    y.reshape(-1),
    ignore_index=tokenizer["<PAD>"]
)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_train_loss = epoch_loss / (len(train_data) / bs)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i in range(0, len(val_data), bs):
                batch = val_data[i:i + bs]
                tokens = torch.stack(batch).to(DEVICE)
                X = tokens[:, :-1]
                y = tokens[:, 1:]
                logits, _ = model(X)
                loss = F.cross_entropy(
    logits.reshape(-1, logits.size(-1)),
    y.reshape(-1),
    ignore_index=tokenizer["<PAD>"]
)

                val_loss += loss.item()
        avg_val_loss = val_loss / (len(val_data) / bs)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        # Generate a sample text
        sample_text = generate_text(model, tokenizer, tokenizer_inv, context="<START>", gen_tokens=20)
        print(f"Sample text: {sample_text}")

    return train_losses, val_losses


#####################################
# Main training and evaluation pipeline
#####################################

def main():
    # Load and preprocess content
    train_dataset, val_dataset, tokenizer, tokenizer_inv = load_and_preprocess_data()
    vocab_size = len(tokenizer)

    # Model hyperparameters
    embed_dim = 128
    num_heads = 4
    num_layers = 2
    dropout = 0.1
    model = TransformerLM(vocab_size, embed_dim, num_heads, num_layers, dropout).to(DEVICE)

    # Print model summary
    print(model)

    # Train the model
    epochs = 10
    train_losses, val_losses = train_model(model, train_dataset, val_dataset, tokenizer, tokenizer_inv, epochs=epochs,
                                           bs=32, lr=1e-3)

    # Plot training and validation losses
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Save the model
    torch.save(model.state_dict(), "transformer_lm.pth")

    # Evaluate on test content
    with open("./content/shakespear_test.txt", "r") as f:
        lines_test = f.readlines()
    # Create a dataset for test content (using the same preprocessing as training)
    test_dataset = ShakespeareDataset(lines_test, tokenizer, max_len=MAX_LEN)
    test_losses = evaluate_losses([test_dataset[i] for i in range(len(test_dataset))], model, tokenizer, bs=32)
    test_ppl = math.exp(sum(test_losses) / len(test_losses))
    print(f"\nTest perplexity: {test_ppl:.4f}")


if __name__ == "__main__":
    main()

#%%

def inference(model_path, test_file, tokenizer, tokenizer_inv, gen_tokens=10, temperature=0.6):
    """
    Load the saved model, generate text for each line in test_file, and calculate perplexity.
    """
    # Hyperparameters must match training
    vocab_size = len(tokenizer)
    embed_dim = 128
    num_heads = 4
    num_layers = 2
    dropout = 0.1
    model = TransformerLM(vocab_size, embed_dim, num_heads, num_layers, dropout).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    with open(test_file, "r") as f:
        lines = f.readlines()

    generated_texts = []
    total_loss = 0
    count = 0
    for line in lines:
        line = line.strip()
        gen_text = generate_text(model, tokenizer, tokenizer_inv, context=line, gen_tokens=gen_tokens,
                                 temperature=temperature)
        generated_texts.append(gen_text)

        # Compute loss for the given line
        token_ids = tokenize("<START> " + line + " <STOP>", pad_to_len=MAX_LEN, tokenizer=tokenizer)
        tokens = torch.tensor(token_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
        X = tokens[:, :-1]
        y = tokens[:, 1:]
        with torch.no_grad():
            logits, _ = model(X)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=tokenizer["<PAD>"])
        total_loss += loss.item()
        count += 1

    ppl = math.exp(total_loss / count)
    return generated_texts, ppl


# Example usage of inference
model_path = "transformer_lm.pth"
test_file = "../Data/shakespear_train.txt"
# You must load these from your training pipeline; here we reload them for inference.
_, _, tokenizer, tokenizer_inv = load_and_preprocess_data()
generated_texts, ppl = inference(model_path, test_file, tokenizer, tokenizer_inv)
print("Generated texts:")
# for text in generated_texts:
#     print(text)
print(f"\nTest Perplexity: {ppl:.4f}")
