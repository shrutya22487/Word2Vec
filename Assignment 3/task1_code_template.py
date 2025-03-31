#%%
## Add imports here

DEVICE = 
MAX_LEN = 

def initialise_projections():
    """
    create projections for Q, K, V.
    """

def pairwise_similarities():
    """
    Compute dot product attention.
    """
    
def attention_scaled():
    """
    Scale the raw attention scores.
    """
    
def attention_softmax():
    """
    Normalize the scaled raw attention scores with softmax.
    """

def compute_outputs():
    """
    Get outputs as a weighted sum of values by attention scores.
    """

def make_causal_mask():
    """
    Create a mask matrix that masks future context for the attention.
    """

def apply_causal_mask():
    """
    Apply mask to attention.
    """

def split_heads():
    """
    Splitting the input across multiple heads.
    """

def merge_heads():
    """
    Reversing splitting action of function split_heads().
    """

def self_attention():
    """
    Self-attention block.
    """

def split_heads_qkv():
    """
    Split Q, K, V across multiple heads.
    """

def load_and_preprocess_data():
    with open("/content/shakespear_train.txt", "r") as f:
        lines_train = f.readlines()
    with open("/content/shakespear_dev.txt", "r") as f:
        lines_dev = f.readlines()
    with open("/content/shakespear_test.txt", "r") as f:
        lines_test = f.readlines()

    tokens_train = [line.split() for line in lines_train]

    # Utility function to flatten tokens
    def flat(tokens):
        ## Your code here

    token_counts = Counter(flat(tokens_train))

    ## Create tokenizer
    tokenizer =

    ## Create inverse tokenizer for decoding
    tokenizer_inv = 

    ## Prepare datasets
    data_train = 
    data_val = 

    ## Create input-output pairs
    train_dataset = 
    val_dataset = 

    return train_dataset, val_dataset, tokenizer, tokenizer_inv

def pad_to_length(tokens, max_len, tokenizer):
    """
    Pad tokens to a fixed length.
    """

def tokenize(sentence, pad_to_len=None, tokenizer=None, include_stop=True):
    """
    Tokenize a sentence.
    """
    

def decode(tokens, tokenizer_inv, end_at_stop=True, omit_pad=True):
    """
    Decode tokens to text.
    """

@torch.no_grad()
def evaluate_losses(data, model, tokenizer, bs=32, progress=True, pad_to_len=MAX_LEN):
    it = range(0, len(data), bs)
    if progress:
        it = tqdm(it)

    out = []
    for b_start in it:
        batch = slice(b_start, b_start + bs)
        tokens = torch.tensor(
            [tokenize(t, pad_to_len=pad_to_len, tokenizer=tokenizer) for t in data[batch]], dtype=torch.long
        ).to(DEVICE)
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
    ## Tokenize the context


    model.eval()
    with torch.no_grad():
        for _ in range():
            ## Get predictions

            ## Focus on the last token's predictions

            ## Apply softmax to get probabilities

            ## Sample from the distribution

            ## Append to the context

            ## Stop if we generated a STOP token
         

    ## Convert back to text
#%%
## Define the Transformer model
class TransformerLM():
    def __init__():        

    def _init_weights():

    def forward():


class TransformerBlock():
    def __init__():

    def forward():


class MultiHeadAttention():
    def __init__():


    def forward():


## Training function
def train_model():

    train_losses = []
    val_losses = []

    for epoch in range():
        model.train()

        
        model.eval()
        with torch.no_grad():

        print(f"Epoch {epoch+1}: Train Loss = {}, Val Loss = {}")

        ## Generate a sample text
        
        print(f"Sample text: {}")


def main():
    ## Load and preprocess data

    ## Create data loaders

    ## Model hyperparameters

    ## Initialize model

    ## Print model summary

    ## Train the model

    ## Plot training and validation losses

    ## Save the model

    ## Evaluate on test data
    with open("/content/shakespear_test.txt", "r") as f:
        lines_test = f.readlines()

    print(f"\nTest perplexity: {}")

if __name__ == "__main__":
    main()
#%%
def inference(model_path, test_file, tokenizer, tokenizer_inv, gen_tokens=10, temperature=0.6):
    ## Load the saved model
    
    ## Read and process the input from test.txt

    ## Generate text and calculate perplexity


model_path = 
test_file = 
generated_texts, ppl = inference(model_path, test_file, tokenizer, tokenizer_inv)
## Print the generated text and perplexity