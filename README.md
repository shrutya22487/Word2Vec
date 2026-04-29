# 🚀 NLP Tokenization & Language Modeling from Scratch

This project implements core **Natural Language Processing (NLP)** components from scratch **without using external NLP libraries** like NLTK or Hugging Face.  

## 📌 Overview  

This repository covers three major NLP tasks:  

- 🏷 **Task 1**: Implementing a **WordPiece Tokenizer**.  
- 🧠 **Task 2**: Building a **Word2Vec (CBOW) Model** using PyTorch.  
- 🔥 **Task 3**: Training a **Neural Language Model (MLP-based)** with three architectural variations.  

---

## 🎯 Features  

✅ **Fully custom implementation** of a WordPiece Tokenizer.  
✅ **Word2Vec (CBOW) model** using PyTorch.  
✅ **Neural Language Model** trained for next-word prediction.  
✅ **PyTorch-based training pipeline** with loss visualization.  
✅ **Evaluation metrics** including cosine similarity, accuracy, and perplexity.  

---

## 📜 Task 1: WordPiece Tokenizer  

### 🛠 Implementation Details  

- **Preprocessing**: Cleans and processes raw text data.  
- **Vocabulary Construction**: Extracts subword tokens and saves them in `vocabulary_{GroupNo}.txt`.  
- **Tokenization**: Converts sentences into subword tokens.  

### 📂 Deliverables  

📄 `task1.py` - Contains the **WordPieceTokenizer** class.  
📄 `vocabulary_{GroupNo}.txt` - Stores the generated vocabulary.  
📄 `tokenized_{GroupNo}.json` - Output JSON file with tokenized sentences.  

📸 **Sample Output:**  
*(Insert Tokenized Output Image Here)*  

---

## 🧠 Task 2: Word2Vec Model (CBOW)  

### 🛠 Implementation Details  

- **Dataset Preparation**: Implements `Word2VecDataset` to create training data.  
- **Word2Vec Model**: Implements a CBOW-based neural network using PyTorch.  
- **Training Function**: Manages the training pipeline.  
- **Similarity Calculation**: Computes cosine similarity for token triplets.  

### 📂 Deliverables  

📄 `task2.py` - Contains `Word2VecDataset` and `Word2VecModel` classes.  
📦 **Model checkpoint** after training.  
📊 **Loss curve visualization**.  
🔗 **Identified token triplets** based on cosine similarity.  

📸 **Loss Graph Output:**  
*(Insert Graph Image Here)*  

📸 **Token Similarity Example:**  
*(Insert Token Similarity Output Image Here)*  

---

## 🔥 Task 3: Neural Language Model (MLP)  

### 🛠 Implementation Details  

- **Dataset Preparation**: Implements `NeuralLMDataset` for next-word prediction.  
- **Three Neural Network Variations**:  
  - 🏗 **NeuralLM1**: Baseline model.  
  - 🚀 **NeuralLM2**: Modified activation functions and layers.  
  - 🔬 **NeuralLM3**: Increased input token size.  
- **Training Function**: Handles training across all models.  
- **Evaluation Metrics**: Computes **accuracy and perplexity**.  
- **Next Token Prediction**: Predicts the next three tokens for test sentences.  

### 📂 Deliverables  

📄 `task3.py` - Contains dataset class and three model architectures.  
📊 **Training and validation loss curves**.  
📈 **Accuracy and perplexity scores**.  
📜 **Token predictions** for `test.txt`.  

📸 **Loss Curves for Models:**  
*(Insert Loss Graphs Here)*  

📸 **Accuracy and Perplexity Results:**  
*(Insert Results Here)*  

---

## ⚡ Setup and Execution  

### 🔧 Prerequisites  

Ensure you have the following installed:  

- 🖥 **Python 3.x**  
- 📦 **PyTorch**  
- 📦 **NumPy**  
- 📦 **Pandas**  

#### 📥 Installation
```bash
pip install torch numpy pandas
``` 

<!-- ```markdown -->
### 🚀 Running the Scripts  

Run the following commands to execute each task:  

**Task 1: WordPiece Tokenizer**  
```bash
python WordPieceTokeniser.py
```

**Task 2: Word2Vec Training**  
```bash
python Word2Vec_model.py
```

**Task 3: Neural Language Model**  
```bash
python task3.py
```

---

## 📊 Results and Observations  

✅ The **WordPiece Tokenizer** effectively segments words into subwords.  
✅ The **CBOW Word2Vec model** captures meaningful word relationships.  
✅ The **Neural Language Models** exhibit varying performance based on architecture choices.  
✅ **Higher token context** improves next-word prediction accuracy.  

---

## 🔮 Future Improvements  

🔹 **Implement positional encoding** for better embeddings.  
🔹 **Experiment with Transformer-based models** for improved performance.  
🔹 **Extend vocabulary using larger datasets**.  

---


