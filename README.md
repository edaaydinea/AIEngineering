# AIEngineering

This repository contains materials, notes, and code for the Udemy course: [**The AI Engineer Course - Complete AI Engineer Bootcamp**](https://www.udemy.com/course/the-ai-engineer-course-complete-ai-engineer-bootcamp). Additionally, it includes personal notes, insights, and projects created throughout the learning process.  

---

## 📑 Table of Contents  

- [AIEngineering](#aiengineering)
  - [📑 Table of Contents](#-table-of-contents)
  - [📦 Installed Packages](#-installed-packages)
  - [🏗️ Setting Up the Conda Environment](#️-setting-up-the-conda-environment)
    - [1️⃣ Create and Activate the Environment](#1️⃣-create-and-activate-the-environment)
    - [2️⃣ Install Required Packages](#2️⃣-install-required-packages)
    - [3️⃣ Download Language Model for spaCy](#3️⃣-download-language-model-for-spacy)
    - [4️⃣ Install Jupyter \& Configure Kernel](#4️⃣-install-jupyter--configure-kernel)
  - [📆 Daily Progress](#-daily-progress)
    - [Day 1 - Day 19: Python Programming](#day-1---day-19-python-programming)
    - [Day 20: NLP Module: Introduction](#day-20-nlp-module-introduction)
    - [Day 21: NLP Module: Text Preprocessing](#day-21-nlp-module-text-preprocessing)
    - [Day 22: NLP Module: Identifying Parts of Speech and Named Entities](#day-22-nlp-module-identifying-parts-of-speech-and-named-entities)
    - [Day 23: NLP Module: Sentiment Analysis](#day-23-nlp-module-sentiment-analysis)
    - [Day 24: NLP Module: Vectorizing Text](#day-24-nlp-module-vectorizing-text)
    - [Day 25: NLP Module: Topic Modeling](#day-25-nlp-module-topic-modeling)
    - [Day 26: NLP Module: Building Your Own Text Classifier](#day-26-nlp-module-building-your-own-text-classifier)
    - [Day 27: NLP Module: Categorizing Fake News (Case Study)](#day-27-nlp-module-categorizing-fake-news-case-study)
    - [🔍 Highlights](#-highlights)
    - [📌 Notes \& Next Steps](#-notes--next-steps)
    - [Day 28: NLP Module: The Future of NLP](#day-28-nlp-module-the-future-of-nlp)
    - [Day 29: LLMs Module: Introduction to Large Language Models](#day-29-llms-module-introduction-to-large-language-models)
  - [Day 30: LLMs Module: The Transformer Architecture](#day-30-llms-module-the-transformer-architecture)
  - [Day 31: LLMs Module: Getting Started with GPT models](#day-31-llms-module-getting-started-with-gpt-models)
  - [Day 32: LLMs Module: Hugging Face Transformers](#day-32-llms-module-hugging-face-transformers)

---

## 📦 Installed Packages  

Below is the list of installed packages and their respective versions used throughout the course. These dependencies were managed within a Conda environment to ensure reproducibility and consistency.  

**Python Version: 3.11**  

```plaintext
nltk==3.9.1  
pandas==2.2.3  
matplotlib==3.10.0  
spacy==3.8.3  
textblob==0.18.0.post0  
vaderSentiment==3.3.2  
transformers==4.47.1  
scikit-learn==1.6.0  
gensim==4.3.3  
seaborn==0.13.2  
torch==2.5.1  
ipywidgets==8.1.5  
```

---

## 🏗️ Setting Up the Conda Environment  

To maintain an isolated and structured development environment, we use Conda to create and manage dependencies. Follow the steps below to set up the required environment:  

### 1️⃣ Create and Activate the Environment  

```bash
conda create --name nlp_course_env python=3.11
conda activate nlp_course_env
```

### 2️⃣ Install Required Packages  

```bash
pip install nltk==3.9.1 pandas==2.2.3 matplotlib==3.10.0 spacy==3.8.3 \
    textblob==0.18.0.post0 vaderSentiment==3.3.2 transformers==4.47.1 \
    scikit-learn==1.6.0 gensim==4.3.3 seaborn==0.13.2 torch==2.5.1 \
    ipywidgets==8.1.5
```

### 3️⃣ Download Language Model for spaCy  

```bash
python -m spacy download en_core_web_sm
```

### 4️⃣ Install Jupyter & Configure Kernel  

```bash
pip install ipykernel jupyterlab notebook
python -m ipykernel install --user --name=nlp_course_env
```

---

## 📆 Daily Progress  

A daily log to track learning progress, document challenges, and reflect on new concepts.

### Day 1 - Day 19: Python Programming

📌 Note: Days 1-19 cover foundational Python programming and general concepts, which will not be shared here. From Day 20 onward, the focus shifts to NLP, LLMs, and Speech Recognition, and relevant notes will be documented. 🚀

### Day 20: NLP Module: Introduction

**What I did today:**

- Gained an overview of Natural Language Processing (NLP) applications in daily life, including search engines, spam detection, and chatbots.
- Understood the fundamentals of text pre-processing, parts of speech tagging, and named entity recognition.
- Explored sentiment analysis techniques for extracting emotional context from text data.
- Learned about text vectorization methods to prepare text data for machine learning models.
- Acquired knowledge of advanced NLP topics such as topic modeling and custom classifier development.
- Examined a real-world case study to apply theoretical knowledge to practical scenarios.
- Distinguished between supervised and unsupervised learning within the context of NLP, including the use of labeled and unlabeled data.
- Understood the evolution of NLP from rule-based systems to advanced models like ChatGPT, driven by large datasets and technological advancements.
- Conceptualized code examples for keyword extraction, spam detection, chatbot responses, supervised learning, and unsupervised learning in NLP.

**Resources:**

- [notes.ipynb](./Section20/notes.ipynb)

### Day 21: NLP Module: Text Preprocessing

**What I did today:**

- Gained a comprehensive understanding of the importance of data preprocessing in Natural Language Processing (NLP), emphasizing the "garbage in, garbage out" principle.
- Developed practical skills in text cleaning, including noise removal and formatting, to prepare data for machine learning algorithms.
- Implemented Python's `lower()` function to ensure consistent word recognition by converting text to lowercase, while acknowledging potential drawbacks with certain abbreviations.
- Utilized the NLTK library to remove stopwords, customizing the stopword list to improve model performance and reduce data complexity.
- Applied regular expressions (regex) for pattern matching, text replacement, and punctuation removal, using `re.search` and `re.sub` functions.
- Practiced tokenization using NLTK, distinguishing between word and sentence tokenization to break text into manageable units.
- Explored text standardization techniques, including stemming with the Porter stemmer and lemmatization with WordNet Lemmatizer, comparing their effects on word reduction and meaning preservation.
- Analyzed n-grams (unigrams, bigrams, and trigrams) using NLTK, pandas, and matplotlib, visualizing frequent word sequences for feature creation and pattern identification.

**Resources:**

- [notes.ipynb](./Section21/notes.ipynb)
- [codes.ipynb](./Section21/codes.ipynb)
- [practical.ipynb](./Section21/practical.ipynb)

### Day 22: NLP Module: Identifying Parts of Speech and Named Entities

**What I did today:*

- Implemented Parts of Speech (POS) tagging using Spacy and Pandas to analyze textual data, specifically processing Jane Austen's "Emma."
- Developed a Python script to extract tokens and their POS tags, organizing the data into a Pandas DataFrame for frequency analysis.
- Analyzed the frequency of tokens and POS tags, filtering for specific grammatical roles like nouns and adjectives to gain insights into text composition.
- Applied Named Entity Recognition (NER) with Spacy to identify and label entities such as dates, people, and organizations within text from the Google Wikipedia page.
- Utilized Displacy for visualizing identified named entities, enhancing text comprehension.
- Demonstrated the impact of text cleaning (punctuation removal and lowercasing) on NER results, highlighting the importance of preprocessing timing.

**Resources:**

- [notes.ipynb](./Section22/notes.ipynb)
- [codes.ipynb](./Section22/codes.ipynb)
- [practical.ipynb](./Section22/practical.ipynb)

### Day 23: NLP Module: Sentiment Analysis

**What I did today:**

- Defined sentiment analysis as a Natural Language Processing (NLP) technique used to determine the emotional tone of text, categorizing it as positive, negative, or neutral.
- Explained the applicability of sentiment analysis in gauging public opinion regarding brands, products, and individuals.
- Demonstrated the use of rule-based sentiment analysis with Textblob and Vader, comparing their methods and outputs on example sentences.
- Analyzed the differences in how Textblob and Vader handle nuanced language, noting Vader's higher sensitivity to contrasting clauses.
- Introduced sentiment analysis using pre-trained transformer models from the Transformers library, highlighting their advanced capability in understanding word relationships and context.
- Showed how to implement sentiment analysis pipelines with the Transformers library, including the selection of specific pre-trained models for optimized results.
- Emphasized the importance of matching the pre-trained model to the characteristics of the data for improved accuracy.

**Resources:**

- [notes.ipynb](./Section23/notes.ipynb)
- [codes.ipynb](./Section23/codes.ipynb)
- [codes2.ipynb](./Section23/codes2.ipynb)
- [practical.ipynb](./Section23/practical.ipynb)

### Day 24: NLP Module: Vectorizing Text

**What I did today:**

- Gained a foundational understanding of text vectorization, including the Bag of Words model and TF-IDF, and their respective applications in machine learning.
- Implemented the Bag of Words model using `CountVectorizer` from scikit-learn, converting text data into a numerical matrix representing word occurrences.
- Utilized pandas DataFrames to effectively visualize and interpret the output of the Bag of Words model, noting the loss of contextual information.
- Explored TF-IDF as an enhanced text vectorization technique, which considers both term frequency and inverse document frequency to capture word importance.
- Applied `TfidfVectorizer` from scikit-learn to transform text data into a TF-IDF matrix, demonstrating the retention of more contextual information compared to the Bag of Words model.
- Analyzed the numerical output of TF-IDF, understanding how higher scores indicate greater word significance across documents.
- Prepared for the next lesson by recognizing the importance of TF-IDF calculations for improving machine learning model performance in natural language processing tasks.

**Resources:**

- [notes.ipynb](./Section24/notes.ipynb)
- [codes.ipynb](./Section24/codes.ipynb)

### Day 25: NLP Module: Topic Modeling

**What I did today:**

- Understood the concept of topic modeling as an unsupervised machine learning technique for identifying patterns and grouping similar documents.
- Learned that topic modeling algorithms analyze word patterns to uncover latent themes within text data, simplifying large datasets.
- Explored real-world applications of topic modeling, including organizing news articles, analyzing customer feedback, and monitoring social media discussions.
- Studied Latent Dirichlet Allocation (LDA), a probabilistic model that identifies latent topics by analyzing word frequencies and distributions.
- Implemented LDA in Python using the Gensim library, preprocessing text data, training the model, and interpreting the resulting topics.
- Examined Latent Semantic Analysis (LSA), a topic modeling technique based on Singular Value Decomposition (SVD) for dimensionality reduction.
- Implemented LSA in Python using Gensim's LsiModel, comparing its output with LDA and interpreting the generated topics.
- Learned how to determine the optimal number of topics for LSA using coherence scores and visualized the results to identify the best number of topics.
- Applied mathematical metrics and business context to select the most meaningful and coherent topics.
- Prepared for future work by understanding the importance of optimizing the number of topics for improved topic modeling results.

**Resources:**

- [notes.ipynb](./Section25/notes.ipynb)
- [codes1.ipynb](./Section25/codes1.ipynb)
- [codes2.ipynb](./Section25/codes2.ipynb)

### Day 26: NLP Module: Building Your Own Text Classifier

**What I did today:**

- Developed a foundational understanding of custom text classification using supervised machine learning algorithms.
- Explored the application of Logistic Regression, Naive Bayes, and Linear Support Vector Machine for text data classification.
- Implemented a logistic regression model for sentiment analysis, establishing a baseline for model performance.
- Trained a Naive Bayes model, observing a slight improvement in accuracy compared to the logistic regression baseline.
- Applied a Linear Support Vector Machine (SVM) model, achieving further accuracy improvement, while acknowledging the need for data refinement.
- Gained practical experience in data preprocessing, feature extraction using CountVectorizer, and model evaluation using accuracy scores and classification reports.
- Identified the iterative nature of machine learning projects and the importance of hyperparameter tuning and data augmentation for model optimization.
- Recognized the value of further learning in algorithm details and hyperparameter tuning through recommended 365 Data Science courses.

**Resources:**

- [notes.ipynb](./Section26/notes.ipynb)
- [codes.ipynb](./Section26/codes.ipynb)

### Day 27: NLP Module: Categorizing Fake News (Case Study)

### 🔍 Highlights

- Imported and utilized the `pandas` and `matplotlib.pyplot` libraries for data manipulation and plotting.
- Loaded and explored a dataset related to fake and factual news articles.
- Preprocessed the dataset, potentially cleaning and formatting it for analysis.
- Implemented a Support Vector Machine (SVM) model using `sklearn.svm` to classify news as either "Factual News" or "Fake News".
- Evaluated the SVM model's performance using `accuracy_score` and `classification_report` from `sklearn.metrics`.
- Achieved an accuracy of approximately 83% on the test set with the SVM model.
- Generated a classification report showing precision, recall, and F1-score for both "Factual News" and "Fake News" categories.

### 📌 Notes & Next Steps

- Further analysis may involve exploring other classification models or refining the current SVM model.
- Visualization of the classification results or data characteristics could enhance understanding.
- Communication strategies for presenting findings to stakeholders should be developed.
- Investigate methods to address the class imbalance, as indicated by the differences in support for each class.

**Resources:**

- [codes.ipynb](./Section27/codes.ipynb)

### Day 28: NLP Module: The Future of NLP

**What I did today:**

- Gained a foundational understanding of deep learning principles, including neural network architectures and training methodologies.
- Explored the application of deep learning in Natural Language Processing, with a focus on large language models like ChatGPT and the Transformer architecture.
- Learned about the challenges and strategies involved in extending NLP techniques to languages beyond English, including data availability and pre-processing considerations.
- Investigated the importance of language support in NLP packages and the existence of language-specific libraries such as AI-NLTK for Indic languages.
- Considered the future trajectory of NLP, including advancements in contextual understanding, multimodal integration, model optimization, and ethical considerations.
- Reviewed example code demonstrating the loading of spaCy language models and the concept of adapting English NLP code for other languages by utilizing different language models.

**Resources:**

- [notes.ipynb](./Section28/notes.ipynb)

### Day 29: LLMs Module: Introduction to Large Language Models

**What I did today:**

- Gained an understanding of the diverse capabilities of large language models (LLMs) and their increasing prominence, exemplified by models like ChatGPT.
- Learned about the fundamental Transformer architecture that underpins the functionality of modern LLMs.
- Recognized the significance of the size of LLMs, measured by the number of parameters, in their ability to process and generate language effectively.
- Understood that LLMs are general-purpose models, pre-trained on vast datasets to acquire broad language understanding, which can then be fine-tuned for specific tasks.
- Identified pre-training and fine-tuning as crucial stages in LLM development, enabling them to learn general language patterns and then specialize in particular applications.
- Explored the wide array of potential applications for LLMs, including content creation, language translation, question answering, chatbot development, and more.

***Resources:**

- [notes.ipynb](./Section29/notes.ipynb)

## Day 30: LLMs Module: The Transformer Architecture

**What I did today:**

- Reviewed the evolution of machine learning from rule-based systems to modern deep learning and transformer architectures capable of complex data analysis and generation.
- Identified the limitations of Recurrent Neural Networks (RNNs) in processing long sequences and understanding context due to sequential processing and memory decay.
- Explored the "attention" mechanism, a core component of transformer models introduced in the "Attention is All You Need" paper, which enables weighting input token importance for improved contextual understanding and handling long-range dependencies.
- Gained an understanding of the overall Transformer architecture, recognizing its encoder-decoder structure and its ability to process input words simultaneously, contrasting it with RNNs.
- Detailed the initial step of creating input embeddings in the Transformer architecture, encompassing tokenization, vocabulary mapping, word embeddings, positional encoding, and handling sequence length variations.
- Analyzed the multi-head attention mechanism within the encoder block, understanding its function in weighing token importance through query, key, and value vectors and enabling the model to capture diverse data patterns.
Examined the role of the feedforward layer in the encoder block, recognizing its importance in learning complex, non-linear relationships between tokens through linear transformations and activation functions.
- Investigated the masked multi-head attention mechanism in the decoder block, understanding how it enforces sequential output generation by preventing the model from accessing future tokens during training.
- Deciphered the process of predicting final outputs in the decoder block, detailing how it combines information from the encoder and its own masked attention to generate a context vector for probabilistic prediction of the next token.

**Resources:**

- [notes.ipynb](./Section30/notes.ipynb)

## Day 31: LLMs Module: Getting Started with GPT models

**What I did today:**

- Gained a foundational understanding of Large Language Models (LLMs), specifically ChatGPT, by exploring its "Generative Pre-trained Transformer" architecture and its core capabilities in text generation and completion.
- Traced the evolutionary trajectory of GPT models, noting the significant increase in scale and performance from GPT-1 to GPT-4 and understanding ChatGPT's specific fine-tuning for conversational interactions.
- Acquired practical knowledge on accessing and utilizing the OpenAI API, including the essential steps for creating and securely managing API keys and understanding the token-based pricing structure.
- Developed hands-on experience with generating text using the OpenAI API through Python functions, learning how to structure API calls with prompts and interpret responses.
- Mastered techniques for customizing GPT model outputs by experimenting with the `max_tokens` and `temperature` parameters to control response length and creativity.
- Implemented a key word text summarization function using the OpenAI API's chat completion endpoint, demonstrating the use of system, user, and assistant roles to guide the model's output.
- Constructed a simple chatbot with a defined persona (poetic) by strategically employing system messages and providing illustrative user-assistant message pairs to shape the model's response style.
- Recognized the inherent knowledge limitations of standard LLMs due to training data cut-off dates and identified LangChain as a solution for integrating external, custom data.
- Introduced the LangChain framework as an open-source solution for connecting LLMs with external data and computation, highlighting its role in building more powerful and data-aware applications.
- Implemented a multi-step process using LangChain to add custom web data to a language model, involving data loading, text splitting, embedding creation, vector store utilization, and initializing a conversational retrieval chain for answering questions based on the custom data.

**Resources:**

- [notes.ipynb](./Section31/notes.ipynb)
- [codes.ipynb](./Section31/codes.ipynb)

## Day 32: LLMs Module: Hugging Face Transformers

**What I did today:**

- Gained a foundational understanding of Hugging Face as an open-source hub and the capabilities of its Transformers library for accessing, utilizing, and fine-tuning diverse Large Language Models (LLMs).
- Acquired hands-on experience with the Hugging Face `pipeline` function, successfully implementing sentiment analysis, named entity recognition (NER), and zero-shot classification tasks.
- Explored the intricacies of pre-trained tokenizers, including the use of `AutoTokenizer` for model-specific tokenization, and analyzed the structure of tokenizer outputs such as `input_ids`, `token_type_ids`, and `attention_mask` for models like BERT and XLNet.
- Developed a clear understanding of various special tokens (e.g., [CLS], [SEP], [MASK], [PAD]) and their critical roles in structuring input, guiding model behavior for tasks like classification and masked language modeling, and ensuring correct input formatting across different LLMs.
- Integrated Hugging Face Transformers with PyTorch, performing manual inference by tokenizing input to PyTorch tensors, loading pre-trained models such as `distilbert-base-uncased-finetuned-sst-2-english` with `AutoModelForSequenceClassification`, and interpreting model logits to retrieve human-readable labels.
- Mastered the procedures for managing model lifecycles by saving trained models and tokenizers using `save_pretrained()` and efficiently reloading them for future use or deployment with `from_pretrained()`.
- Enhanced conceptual knowledge regarding the importance of framework integration (PyTorch/TensorFlow), model-specific tokenization, the utility of the Hugging Face Model Hub, and the underlying mechanisms of high-level abstractions like the `pipeline` function.

**Resources:**

- [notes.ipynb](./Section32/notes.ipynb)
