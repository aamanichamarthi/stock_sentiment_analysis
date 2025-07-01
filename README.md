# stock_sentiment_analysis
Stock Sentiment Analysis
This project aims to perform sentiment analysis on stock market-related text data. By analyzing the sentiment of financial news or social media posts, we can potentially gain insights into market trends and make more informed decisions.

Project Overview
The project involves several key steps:

Data Loading and Initial Exploration: Loading the dataset and understanding its basic structure, checking for missing values, and exploring the distribution of sentiment.

Data Cleaning: Preprocessing the text data by removing punctuations and stopwords to prepare it for analysis.

Word Cloud Visualization: Generating a word cloud to visualize the most frequent words in the pre-processed text, providing a quick overview of the prominent themes.

Data Preparation for Deep Learning: Tokenizing the text and padding the sequences to ensure uniform input length for the neural network model. The sentiment labels are also converted into a categorical 2D representation.

Model Building: Constructing a deep neural network, specifically using an LSTM (Long Short-Term Memory) layer, for sentiment classification.

Model Training: Training the built model on the prepared training data.

Model Performance Assessment: Evaluating the model's performance on unseen test data using various metrics like accuracy, F1-score, precision, recall, and AUC score, along with a confusion matrix visualization.

Setup Instructions
To set up and run this project, follow these steps:

1. Clone the Repository
If this project is part of a larger GitHub repository, you would clone it using:

Bash

git clone <repository_url>
cd <project_directory>
2. Install Required Libraries
The project relies on several Python libraries. You can install them using pip:

Bash

pip install wordcloud
pip install gensim
pip install nltk
pip install numpy
pip install pandas
pip install seaborn
pip install tensorflow
pip install jupyterthemes
pip install sklearn
3. Download NLTK Data
The nltk library requires specific datasets for its functionalities (like stopwords and tokenization). Download them by running the following commands within your Python environment or a Jupyter Notebook cell:

Python

import nltk
nltk.download("stopwords")
nltk.download('punkt')
4. Data
The dataset used in this project is named stock_sentiment.csv. Ensure this file is placed in a data directory within your project's root folder (e.g., data/stock_sentiment.csv).

5. Running the Jupyter Notebook
The entire project workflow is encapsulated within a Jupyter Notebook file: 005_Stock_Sentiment_Analysis.ipynb.

To run the notebook:

Start Jupyter Lab or Jupyter Notebook:

Bash

jupyter lab
# or
jupyter notebook
Navigate to the project directory in your Jupyter interface.

Open 005_Stock_Sentiment_Analysis.ipynb.

Run all cells sequentially to execute the data loading, cleaning, model building, training, and evaluation steps.

Project Structure
.
├── data/
│   └── stock_sentiment.csv  # The dataset used in the project
└── 005_Stock_Sentiment_Analysis.ipynb # Main Jupyter Notebook with all code
Data Description
The stock_sentiment.csv dataset contains two columns:

Text: The raw text content (likely financial news headlines or social media posts).

Sentiment: The associated sentiment, represented as a numerical label (e.g., 1 for positive, 0 for negative).

Key Steps and Code Highlights
Data Cleaning
Removing Punctuations: A custom function remove_func is used to strip punctuation from the text.

Removing Stopwords: NLTK's stopwords are utilized, with additional custom stopwords identified from the dataset (e.g., 'from', 'subject', 'https', 're', 'edu', 'use','will','aap','co','day','user','stock','today','week','year'). A preprocess function handles both stopword removal and filtering out short words (less than 3 characters).

Data Preparation
Tokenization: The nltk.word_tokenize function is used to break down sentences into words, and tensorflow.keras.preprocessing.text.Tokenizer converts words into integer sequences.

Padding: tensorflow.keras.preprocessing.sequence.pad_sequences ensures all input sequences to the neural network have a uniform length, which is set to 15 based on the analysis of tweet lengths.

Categorical Conversion: tensorflow.keras.utils.to_categorical transforms the sentiment labels (0 or 1) into a one-hot encoded format suitable for the softmax activation in the output layer.

Deep Learning Model
The model is a Sequential Keras model with the following layers:

Embedding Layer: model.add(Embedding(total_words, output_dim = 512)) maps each word (integer token) to a dense vector of fixed size (512 dimensions).

LSTM Layer: model.add(LSTM(256)) processes the sequence data, capable of capturing long-term dependencies in the text.

Dense Layers:

model.add(Dense(128, activation = 'relu')): A hidden layer with 128 neurons and ReLU activation.

model.add(Dropout(0.3)): A dropout layer for regularization to prevent overfitting.

model.add(Dense(2, activation = 'softmax')): The output layer with 2 neurons (for the two sentiment classes) and softmax activation for probability distribution.

Compilation: The model is compiled with the adam optimizer and binary_crossentropy loss function, suitable for binary classification. Accuracy (acc) is used as a metric.

Results
The model achieved the following performance metrics on the test set:

Training Accuracy: ~94.47%

Validation Accuracy: ~76.38% (from the training phase)

Test Accuracy: ~77.83%

F1 Score: 0.83

Precision: 0.79

Recall: 0.88

AUC Score: 0.74

The confusion matrix provides a visual breakdown of true positives, true negatives, false positives, and false negatives, allowing for a deeper understanding of the model's classification performance.
