import numpy as np
import pandas as pd
import re
import json

class WordPieceTokenizer:
    """
    Initialisation:
    1. Create a dictionary to store vocab and another to store freq
    2. Define vocab size
    """

    def __init__(self, vocab_size: int):

        self.vocab = []
        self.word_freq = {}
        self.vocab_size = vocab_size

    """
    Preprocessing: 
    1. Convert to lower case
    2. Split into words & remove spaces
    3. Remove numbers & non-alphabetic chars like ",", ".", "!" etc. 
    4. Returns an array of strings
    """

    def preprocess_data(self, text: str):

        text = text.lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text)  # replacing characters using RegEx
        return text.split()

    """
    Vocab Construction: 
    1. Preprocess the data and store in dict the frequency of every word/token
    2. Create alphabets from the  by splitting each word into letters
    3. Create splits for every word i.e. storing letters of each word in a dictionary
    4. 
    """

    def construct_vocabulary(self, corpus: list):

        # store word frequencies
        for sentence in corpus:
            tokens = self.preprocess_data(sentence)

            for token in tokens:
                self.word_freq[token] = self.word_freq.get(token, 0) + 1

        # create alphabet and splits
        alphabet = []  # unique morphemes stored
        splits = {}
        for word in self.word_freq.keys():
            if word[0] not in alphabet:
                alphabet.append(word[0])

            for remaining_letter in word[1:]:
                if f"##{remaining_letter}" not in alphabet:
                    alphabet.append(f"##{remaining_letter}")

            splits[word] = [c if i == 0 else f"##{c}" for i, c in enumerate(word)]

        alphabet.sort()

        self.vocab = ["[PAD]", "[UNK]"] + alphabet.copy()  # adding two special tokens to vocab

        # compute scores for pairs
        # merge pair
        """Deplag needed"""
        while len(self.vocab) < self.vocab_size:
            scores = self.compute_pair_scores(splits)
            best_pair, max_score = "", None
            for pair, score in scores.items():
                if max_score is None or max_score < score:
                    best_pair = pair
                    max_score = score
            splits = self.merge_pair(*best_pair, splits)
            new_token = (
                best_pair[0] + best_pair[1][2:]
                if best_pair[1].startswith("##")
                else best_pair[0] + best_pair[1]
            )
            self.vocab.append(new_token)

        self.save_vocabulary()

    """Deplag needed"""

    def tokenize(self, text: str):
        pre_tokenized_text = self.preprocess_data(text)
        encoded_words = [self.encode_word(word) for word in pre_tokenized_text]
        return sum(encoded_words, [])

    # ____________Helper Methods____________

    def save_vocabulary(self):
        vocab_file = f"vocabulary_86.txt"
        with open(vocab_file, "w") as f:
            for token in self.vocab:
                f.write(f"{token}\n")

    """Deplag needed"""

    def compute_pair_scores(self, splits):
        letter_freqs = {}
        pair_freqs = {}
        for word, freq in self.word_freq.items():
            split = splits[word]
            if len(split) == 1:
                letter_freqs[split[0]] = letter_freqs.get(split[0], 0) + freq
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                letter_freqs[split[i]] = letter_freqs.get(split[i], 0) + freq
                pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
            letter_freqs[split[-1]] = letter_freqs.get(split[-1], 0) + freq

        scores = {
            pair: freq / (letter_freqs[pair[0]] * letter_freqs[pair[1]])
            for pair, freq in pair_freqs.items()
        }
        return scores

    """Deplag needed"""

    def merge_pair(self, a, b, splits):
        for word in self.word_freq:
            split = splits[word]
            if len(split) == 1:
                continue
            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    merge = a + b[2:] if b.startswith("##") else a + b
                    split = split[:i] + [merge] + split[i + 2:]
                else:
                    i += 1
            splits[word] = split
        return splits

    """Deplag needed"""

    def encode_word(self, word):
        tokens = []
        while len(word) > 0:
            i = len(word)
            while i > 0 and word[:i] not in self.vocab:
                i -= 1
            if i == 0:
                return ["[UNK]"]
            tokens.append(word[:i])
            word = word[i:]
            if len(word) > 0:
                word = f"##{word}"
        return tokens


def make_vocab_and_tokenize(vocab_size):
    corpus = []
    with open("../Data/corpus.txt", 'r') as file:
        text = file.read()
        sentences = text.split()
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                corpus.append(sentence)

    tokenizer = WordPieceTokenizer(vocab_size=vocab_size)

    tokenizer.construct_vocabulary(corpus)
    vocab = tokenizer.vocab

    output_json_path = '../Data/tokenized_data.json'  # Path to save tokenized output

    # Prepare a dictionary to store the tokenized sentences
    tokenized_data = {}
    with open("../Data/corpus.txt", 'r') as file:
        text = file.read()
        sentences = text.split('\n')
        sentence_id = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                tokens = tokenizer.tokenize(sentence)
                tokenized_data[sentence_id] = tokens
                sentence_id += 1

    # Write the tokenized data to the output JSON file
    with open(output_json_path, 'w') as f:
        json.dump(tokenized_data, f, indent=4)

if __name__ == "__main__":
    corpus = []
    with open("../Data/corpus.txt", 'r') as file:
        text = file.read()
        sentences = text.split()
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                corpus.append(sentence)

    tokenizer = WordPieceTokenizer(vocab_size=1000)

    tokenizer.construct_vocabulary(corpus)
    vocab = tokenizer.vocab

    output_json_path = '../Data/tokenized_data.json'  # Path to save tokenized output

    # Prepare a dictionary to store the tokenized sentences
    tokenized_data = {}
    with open("../Data/corpus.txt", 'r') as file:
        text = file.read()
        sentences = text.split('\n')
        sentence_id = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                tokens = tokenizer.tokenize(sentence)
                tokenized_data[sentence_id] = tokens
                sentence_id += 1

    # Write the tokenized data to the output JSON file
    with open(output_json_path, 'w') as f:
        json.dump(tokenized_data, f, indent=4)