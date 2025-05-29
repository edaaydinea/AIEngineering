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
    - [Day 28: NLP Module: The Future of NLP](#day-28-nlp-module-the-future-of-nlp)
    - [Day 29: LLMs Module: Introduction to Large Language Models](#day-29-llms-module-introduction-to-large-language-models)
    - [Day 30: LLMs Module: The Transformer Architecture](#day-30-llms-module-the-transformer-architecture)
    - [Day 31: LLMs Module: Getting Started with GPT models](#day-31-llms-module-getting-started-with-gpt-models)
    - [Day 32: LLMs Module: Hugging Face Transformers](#day-32-llms-module-hugging-face-transformers)
    - [Day 33: LLMs Module: Question and Answer Models with BERT](#day-33-llms-module-question-and-answer-models-with-bert)
  - [Day 34: LLms Module: Text Classification with XLNet](#day-34-llms-module-text-classification-with-xlnet)
  - [Day 35: LangChain Module: Introduction](#day-35-langchain-module-introduction)
  - [Day 36: LangChain Module: Tokens, Models, and Prices](#day-36-langchain-module-tokens-models-and-prices)
  - [Day 37: LangChain Module: Setting Up the Environment](#day-37-langchain-module-setting-up-the-environment)
  - [Day 38: LangChain Module: The OpenAI API](#day-38-langchain-module-the-openai-api)
  - [Day 39: LangChain Module: Model Inputs](#day-39-langchain-module-model-inputs)
  - [Day 40: LangChain Module: Message History and Chatbot Memory](#day-40-langchain-module-message-history-and-chatbot-memory)
  - [Day 41: LangChain Module: Output Parsers](#day-41-langchain-module-output-parsers)
  - [Day 42: LangChain Module: LangChain Expression Language (LCEL)](#day-42-langchain-module-langchain-expression-language-lcel)
  - [Day 43: LangChain Module: Retrieval Augmented Generation (RAG)](#day-43-langchain-module-retrieval-augmented-generation-rag)
  - [Day 44: LangChain Module: Tools and Agents](#day-44-langchain-module-tools-and-agents)
  - [Day 45: Vector Databases Module: Introduction](#day-45-vector-databases-module-introduction)
  - [Day 46: Vector Databases Module: Basics of Vector Space and High-Dimensional Data](#day-46-vector-databases-module-basics-of-vector-space-and-high-dimensional-data)
  - [Day 47: Vector Databases Module: Introduction to The Pinecone Vector Database](#day-47-vector-databases-module-introduction-to-the-pinecone-vector-database)
  - [Day 48: Vector Databases Module: Semantic Search with Pinecone and Custom (Case Study)](#day-48-vector-databases-module-semantic-search-with-pinecone-and-custom-case-study)

---

## 📦 Installed Packages

Below is the list of installed packages and their respective versions used throughout the course. These dependencies were managed within a Conda environment to ensure reproducibility and consistency.

**Python Version: 3.11**

```
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

```
conda create --name nlp_course_env python=3.11
conda activate nlp_course_env
```

### 2️⃣ Install Required Packages

```
pip install nltk==3.9.1 pandas==2.2.3 matplotlib==3.10.0 spacy==3.8.3 \
    textblob==0.18.0.post0 vaderSentiment==3.3.2 transformers==4.47.1 \
    scikit-learn==1.6.0 gensim==4.3.3 seaborn==0.13.2 torch==2.5.1 \
    ipywidgets==8.1.5
```

### 3️⃣ Download Language Model for spaCy

```
python -m spacy download en_core_web_sm
```

### 4️⃣ Install Jupyter & Configure Kernel

```
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

\*_What I did today:_

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

**What I did today:**

- Imported and utilized the `pandas` and `matplotlib.pyplot` libraries for data manipulation and plotting.
- Loaded and explored a dataset related to fake and factual news articles.
- Preprocessed the dataset, potentially cleaning and formatting it for analysis.
- Implemented a Support Vector Machine (SVM) model using `sklearn.svm` to classify news as either "Factual News" or "Fake News".
- Evaluated the SVM model's performance using `accuracy_score` and `classification_report` from `sklearn.metrics`.
- Achieved an accuracy of approximately 83% on the test set with the SVM model.
- Generated a classification report showing precision, recall, and F1-score for both "Factual News" and "Fake News" categories.

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

\***Resources:**

- [notes.ipynb](./Section29/notes.ipynb)

### Day 30: LLMs Module: The Transformer Architecture

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

### Day 31: LLMs Module: Getting Started with GPT models

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

### Day 32: LLMs Module: Hugging Face Transformers

**What I did today:**

- Gained a foundational understanding of Hugging Face as an open-source hub and the capabilities of its Transformers library for accessing, utilizing, and fine-tuning diverse Large Language Models (LLMs).
- Acquired hands-on experience with the Hugging Face `pipeline` function, successfully implementing sentiment analysis, named entity recognition (NER), and zero-shot classification tasks.
- Explored the intricacies of pre-trained tokenizers, including the use of `AutoTokenizer` for model-specific tokenization, and analyzed the structure of tokenizer outputs such as `input_ids`, `token_type_ids`, and `attention_mask` for models like BERT and XLNet.
- Developed a clear understanding of various special tokens (e.g., \[CLS\], \[SEP\], \[MASK\], \[PAD\]) and their critical roles in structuring input, guiding model behavior for tasks like classification and masked language modeling, and ensuring correct input formatting across different LLMs.
- Integrated Hugging Face Transformers with PyTorch, performing manual inference by tokenizing input to PyTorch tensors, loading pre-trained models such as `distilbert-base-uncased-finetuned-sst-2-english` with `AutoModelForSequenceClassification`, and interpreting model logits to retrieve human-readable labels.
- Mastered the procedures for managing model lifecycles by saving trained models and tokenizers using `save_pretrained()` and efficiently reloading them for future use or deployment with `from_pretrained()`.
- Enhanced conceptual knowledge regarding the importance of framework integration (PyTorch/TensorFlow), model-specific tokenization, the utility of the Hugging Face Model Hub, and the underlying mechanisms of high-level abstractions like the `pipeline` function.

**Resources:**

- [notes.ipynb](./Section32/notes.ipynb)

### Day 33: LLMs Module: Question and Answer Models with BERT

**What I did today:**

- Gained a foundational understanding of BERT's bidirectional nature and its advantages over autoregressive models like GPT for tasks requiring deep contextual understanding.
- Explored the Transformer architecture as the backbone of BERT, recognizing how its attention mechanisms contribute to effective natural language processing.
- Learned about BERT's pre-training on large corpora and its impact on developing a generalized understanding of language patterns.
- Identified BERT's suitability for specific NLP tasks such as sentiment analysis, question answering, and named entity recognition, distinguishing its strengths from GPT's generative capabilities.
- Understood the implications of different BERT model sizes (Base and Large) on performance and computational requirements.
- Delved into BERT's encoder-only architecture, differentiating it from encoder-decoder models and understanding its optimization for text representation.
- Mastered the concept of BERT's three-part input embeddings (token, segment, and positional) and their role in providing comprehensive input information.
- Comprehended the Masked Language Modeling (MLM) pre-training objective as a key enabler of BERT's bidirectional context learning.
- Understood the Next Sentence Prediction (NSP) pre-training objective and its function in teaching BERT inter-sentence relationships.
- Recognized BERT's adaptability through fine-tuning, allowing its powerful pre-trained representations to be specialized for various downstream NLP tasks.
- Acquired practical skills in loading BERT models (specifically `bert-large-uncased-whole-word-masking-finetuned-squad`) and their corresponding tokenizers using the Hugging Face Transformers library.
- Practiced preparing inputs for BERT by tokenizing questions and contexts with `tokenizer.encode_plus`, understanding the role of special tokens (\[CLS\], \[SEP\]) and token type IDs.
- Successfully converted input encodings into PyTorch tensors for model consumption.
- Learned to interpret BERT's output for question answering, specifically how start and end logits are used to predict answer spans.
- Gained hands-on experience in extracting answer indices using `torch.argmax` and reconstructing human-readable answers from tokens.
- Developed skills in visualizing token probabilities for start and end positions using `matplotlib` and `seaborn` to understand model decision-making.
- Designed and implemented a prototype FAQ chatbot for "Sunset Motors" by defining a context, creating a core processing function, manually handling segment IDs, and performing model inference.
- Incorporated answer span validation and text cleanup techniques to refine the chatbot's output.
- Successfully tested the FAQ chatbot with various questions, demonstrating its ability to extract relevant information from the provided context.
- Explored RoBERTa as a robustly optimized BERT variant, noting its improved pre-training strategies and performance benefits.
- Investigated DistilBERT as a smaller, faster, and lightweight distilled version of BERT, suitable for resource-constrained environments.
- Understood how to load and utilize RoBERTa and DistilBERT models and tokenizers via the Hugging Face Transformers library, recognizing their specific use cases.

**Resources:**

- [notes.ipynb](./Section33/notes.ipynb)
- [QAbot.ipynb](./Section33/QAbot.ipynb)

## Day 34: LLms Module: Text Classification with XLNet

**What I did today:**

- Gained a comprehensive understanding of XLNet's architecture, including its decoder-only nature and Permutation Language Modeling, which enables superior bidirectional context capture compared to models like BERT.
- Explored XLNet's ability to achieve state-of-the-art performance on various NLP benchmarks by overcoming limitations of previous autoregressive and masked language models.
- Mastered data preprocessing techniques for an emotion classification task using XLNet, including importing necessary libraries (pandas, transformers, sklearn), cleaning text by removing emojis and punctuation, and handling class imbalance through undersampling.
- Successfully encoded categorical labels to integers, split the data into training, validation, and test sets, and formatted it into a Hugging Face `DatasetDict` for efficient model training.
- Implemented the tokenization process for XLNet, loading the `XLNetTokenizer` from "xlnet-base-cased" and defining a function to tokenize text with padding to a `max_length` of 128 and truncation.
- Applied tokenization to the dataset using the `.map()` method and developed an understanding of `input_ids`, `token_type_ids`, and `attention_mask` as generated by the XLNet tokenizer.
- Initialized `XLNetForSequenceClassification` from "xlnet-base-cased" for an emotion detection task with 4 labels, providing an `id2label` map.
- Configured the training process by defining an accuracy metric, creating a `compute_metrics` function, and setting up `TrainingArguments` for 3 epochs with epoch-based evaluation.
- Utilized the Hugging Face `Trainer` API to fine-tune the XLNet model on a subset of the data (100 samples for training and evaluation) and initiated the training.
- Evaluated the performance of the fine-tuned XLNet model using `trainer.evaluate()` and successfully saved the trained model artifacts using `model.save_pretrained("fine_tuned_model")`.
- Loaded the fine-tuned XLNet model from the saved directory and created a `text-classification` pipeline for inference.
- Demonstrated practical application of the fine-tuned model by making predictions on new text samples and retrieving scores for all emotion labels.

**Resources:**

- [notes.ipynb](./Section34/notes.ipynb)
- [codes.ipynb](./Section34/codes.ipynb)

## Day 35: LangChain Module: Introduction

**What I did today:**

- Gained an initial understanding of building sophisticated chat applications leveraging OpenAI's Large Language Models, specifically GPT-4, and the LangChain Python library.
- Acquired foundational knowledge of LLMs, including their definition, massive scale, and the specific characteristics of chat models.
- Explored LangChain as a comprehensive framework designed to simplify the development of stateful, context-aware, and reasoning-capable LLM-powered applications.
- Reviewed several real-world business implementations of LangChain, such as Ally Financial's secure call summarization with PII masking, Adyen's intelligent customer support ticket routing using Retrieval Augmented Generation (RAG), and RoboCop's AI-assisted code generation.
- Investigated the core capabilities that make LangChain powerful, including its seamless integration with various LLM providers, mechanisms for managing conversational state, versatile document loaders for ingesting diverse data sources, and database integration for extensive knowledge bases.
- Learned how LangChain enables LLMs to utilize external tools (e.g., web search, APIs), allowing them to reason, access real-time information, and perform actions beyond their training data.
- Understood the importance of the broader LangChain ecosystem, including LangSmith for application monitoring and evaluation, and LangServe for deployment.
- Surveyed the key topics covered in a comprehensive LangChain course, encompassing OpenAI API fundamentals (tokens, models, pricing), essential environment setup, core LangChain components (Model I/O, Memory, Document Retrieval, Agents), and the pivotal LangChain Expression Language (LCEL).
- Conceptualized the significance of Retrieval Augmented Generation (RAG) for building chatbots that can answer questions using custom, proprietary, or up-to-date information.
- Recognized the power of combining tools and agents within LangChain to create LLM systems that can autonomously choose and execute actions to solve complex tasks.

**Resources:**

- [notes.ipynb](./Section35/notes.ipynb)

## Day 36: LangChain Module: Tokens, Models, and Prices

**What I did today:**

- Gained a comprehensive understanding of OpenAI's token-based economy, including token definition (approx. 3/4 of a word), model-specific pricing (e.g., GPT-4o: ~$2.50/M input, ~$10.00/M output as of late 2024/early 2025), and the impact of context window limits (e.g., 128,000 for GPT-4o) on project scope and budget.
- Mastered the fundamentals of setting up a development environment for OpenAI and LangChain projects, including Anaconda configuration and API key management.
- Acquired foundational knowledge of the OpenAI API, including chat prompting terminology crucial for effective LLM interaction and integration with LangChain.
- Explored core components of the LangChain framework, such as model I/O, memory management, document retrieval mechanisms, agent capabilities, and the LangChain Expression Language (LCEL), to build advanced LLM applications.
- Developed skills in crafting effective chat messages, utilizing prompt templates, and applying few-shot prompting techniques to guide LLM outputs accurately.
- Understood the principles of creating stateful chatbots that maintain conversational context for more coherent and engaging user experiences.
- Learned techniques for output parsing, enabling the conversion of LLM responses into various data formats (string, list, DateTime) for seamless integration with other applications and tools.
- Grasped the significance of the LangChain Expression Language (LCEL) as the cornerstone for constructing complex chains and workflows within the LangChain ecosystem.
- Investigated Retrieval Augmented Generation (RAG) to empower LLMs with custom, up-to-date information, enabling more accurate and contextually relevant responses.
- Explored the use of external tools and agents within LangChain, allowing LLMs to perform complex, multi-step tasks and exhibit enhanced reasoning capabilities.
- Reviewed essential Python prerequisites and the importance of foundational generative AI knowledge for effectively leveraging LangChain and LLMs.
- Identified key criteria for model selection, including cost, training data recency (e.g., GPT-4o cut-off October 2023), and context window size, to optimize for specific project requirements.
- Became familiar with specific OpenAI models, including the flagship GPT-4o and embedding models like `text-embedding-3-small`, and their respective roles in NLP tasks.

**Resources:**

- [notes.ipynb](./Section36/notes.ipynb)

## Day 37: LangChain Module: Setting Up the Environment

**What I did today:**

- Successfully established a custom Anaconda environment (e.g., `langchain_env` with Python 3.10) to ensure an isolated and reproducible workspace for the LangChain project, preventing potential dependency conflicts.
- Installed essential Python packages, including `openai`, `python-dotenv`, `ipykernel`, `jupyterlab`, and `notebook`, within the new environment to facilitate interaction with the OpenAI API and enable Jupyter integration.
- Integrated the custom Anaconda environment as a new kernel in Jupyter Notebook, allowing for seamless selection and use of the project-specific dependencies and Python interpreter.
- Obtained an OpenAI API key by navigating the platform, configuring billing information, and securely generating and storing the new secret key.
- Mastered the use of environment variables for secure API key management, understanding the critical risks of hardcoding sensitive credentials.
- Implemented the recommended method of setting the OpenAI API key using a `.env` file in conjunction with IPython magic commands (`%load_ext dotenv`, `%dotenv`) in a Jupyter Notebook for persistent and secure access.
- Explored temporary methods for setting environment variables using the `os` module and learned to verify their successful configuration.

**Resources:**

- [notes.ipynb](./Section37/notes.ipynb)
- [codes.ipynb](./Section37/codes.ipynb)

## Day 38: LangChain Module: The OpenAI API

**What I did today:**

- Mastered the fundamentals of creating basic chatbots using Python and the OpenAI API, focusing on direct API interaction without the LangChain framework initially.
- Established best practices for Python project development by utilizing separate virtual environments to manage dependencies and prevent package conflicts.
- Implemented secure API key management by configuring the OpenAI API key as an environment variable and loading it into Python scripts for authorized API calls.
- Gained hands-on experience with initializing the `OpenAIClient` to serve as the primary interface for making requests to OpenAI models.
- Practiced using the `client.chat.completions.create()` method to send requests to models like GPT-4 and retrieve AI-generated responses.
- Developed a strong understanding of structuring messages for the API using a list of dictionaries, each containing `role` and `content` key-value pairs to guide model behavior.
- Learned to differentiate and effectively use message roles (`system`, `user`, `assistant`) to define AI persona, provide user input, and offer example interactions for few-shot prompting.
- Successfully implemented a sarcastic chatbot ("Marv") by crafting specific `system` and `user` messages and programmatically extracting its response from the `ChatCompletion` object.
- Explored advanced API parameters, including `n` to request multiple response variations and `usage` data (completion\_tokens, prompt\_tokens, total\_tokens) for monitoring API costs.
- Acquired skills in fine-tuning model outputs using `max_tokens` to control response length, `temperature` to adjust creativity versus predictability, and `seed` for achieving more reproducible results.
- Implemented real-time response delivery by setting `stream=True` in API calls and learned to process `ChatCompletionChunk` objects to assemble the complete message, enhancing user experience.
- Reinforced conceptual understanding of message structuring, role definition, and response object navigation through practical code examples and reflective analysis.

**Resources:**

- [notes.ipynb](./Section38/notes.ipynb)
- [codes1.ipynb](./Section38/codes1.ipynb)
- [codes2.ipynb](./Section38/codes2.ipynb)
- [codes3.ipynb](./Section38/codes3.ipynb)

## Day 39: LangChain Module: Model Inputs

**What I did today:**

- Acquired a foundational understanding of the LangChain framework, covering its core components (Model I/O, Retrieval, Agents) and its utility in developing sophisticated, stateful, and context-aware LLM applications.
- Successfully configured the LangChain development environment, including essential library installations (`langchain`, `langchain-openai`) and secure OpenAI API key setup.
- Mastered direct interaction with OpenAI chat models (e.g., GPT-4) via the `ChatOpenAI` class, effectively utilizing parameters like `model_name`, `seed`, `temperature`, and `max_tokens` to control and refine responses.
- Developed practical experience in structuring LLM conversations using `SystemMessage` to set AI personas and `HumanMessage` for user inputs, and invoking models with ordered message lists for nuanced contextual control.
- Applied few-shot prompting techniques by constructing explicit `HumanMessage` and `AIMessage` pairs to guide model output style and behavior through illustrative examples.
- Enhanced prompt engineering skills by creating dynamic and reusable prompts with string `PromptTemplate`, generating populated `StringPromptValue` objects for LLM interaction.
- Utilized `ChatPromptTemplate` to systematically manage and format multi-message sequences for chat models, combining `SystemMessagePromptTemplate` and `HumanMessagePromptTemplate` to produce `ChatPromptValue` objects.
- Implemented an advanced and scalable strategy for few-shot prompting in chat applications using `FewShotChatMessagePromptTemplate`, successfully guiding model tone and behavior with structured example sets.
- Explored the `LLMChain` class, understanding its role in directly connecting language models with prompt templates and processing their dictionary-based input/output.
- Grasped the significance of sequential component invocation (e.g., `template.invoke()` output feeding `model.invoke()`) as a core LangChain pattern and a precursor to the LangChain Expression Language (LCEL).
- Conceptualized the role of Output Parsers for structuring LLM outputs and the principles of Retrieval Augmented Generation (RAG) using LangChain's Retrieval module for building context-aware systems.

**Resources:**

- [notes.ipynb](./Section39/notes.ipynb)
- [codes1.ipynb](./Section39/codes1.ipynb)
- [codes2.ipynb](./Section39/codes2.ipynb)
- [codes3.ipynb](./Section39/codes3.ipynb)
- [codes4.ipynb](./Section39/codes4.ipynb)
- [codes5.ipynb](./Section39/codes5.ipynb)
- [codes6.ipynb](./Section39/codes6.ipynb)
- [codes7.ipynb](./Section39/codes7.ipynb)

## Day 40: LangChain Module: Message History and Chatbot Memory

**What I did today:**

- Mastered the use of `ChatMessageHistory` for manual storage and management of conversational dialogue, forming a foundational understanding of context handling in LangChain.
- Configured the setup for persistent chatbot memory by effectively utilizing `MessagesPlaceholder` within `ChatPromptTemplate` for dynamic history injection and enabling `set_verbose(True)` for enhanced debugging capabilities.
- Successfully implemented `ConversationBufferMemory` in an `LLMChain`, enabling chatbots to retain and recall full conversation histories for coherent, multi-turn interactions.
- Explored and implemented `ConversationBufferWindowMemory`, managing conversational context by retaining a fixed window (`k`) of recent interactions to balance context length with token efficiency.
- Implemented `ConversationSummaryMemory`, leveraging a dedicated LLM to progressively summarize dialogues, allowing for efficient long-term context retention by focusing on key information.
- Advanced memory management techniques by utilizing `CombinedMemory` to integrate multiple memory types (e.g., `ConversationBufferMemory` and `ConversationSummaryMemory`) simultaneously, providing LLMs with a richer, multi-faceted conversational context.
- Practiced designing versatile prompt templates that accommodate various memory strategies, including explicit history injection, dynamic placeholders for buffered or summarized memory, and inputs from combined memory sources.

**Resources:**

- [notes.ipynb](./Section40/notes.ipynb)
- [codes1.ipynb](./Section40/codes.ipynb)
- [codes2.ipynb](./Section40/codes.ipynb)
- [codes3.ipynb](./Section40/codes.ipynb)
- [codes4.ipynb](./Section40/codes.ipynb)
- [codes5.ipynb](./Section40/codes.ipynb)
- [codes6.ipynb](./Section40/codes.ipynb)

## Day 41: LangChain Module: Output Parsers

**What I did today:**

- Mastered the use of `StringOutputParser` to convert raw language model outputs into simple, usable string formats, essential for various data science workflows and application integrations.
- Learned to implement the `CommaSeparatedListOutputParser`, including the critical technique of embedding `get_format_instructions()` in prompts to guide the LLM to produce comma-separated output, enabling direct conversion to Python lists.
- Gained proficiency in utilizing the `DateTimeOutputParser` by modifying prompts with format instructions to accurately extract and structure date information from LLM responses into Python `datetime` objects.
- Understood the fundamental principle that output parsers require specific input formats, and effective prompt engineering with format instructions is key to ensuring LLM outputs are compatible for reliable parsing.
- Recognized and addressed potential `OutputParserException` errors by ensuring LLM responses strictly adhere to the expected formats, particularly for specialized parsers like `DateTimeOutputParser`.
- Explored the conceptual workflow and code structure for initializing chat models, creating messages, invoking models, and then applying various output parsers (`StringOutputParser`, `CommaSeparatedListOutputParser`, `DateTimeOutputParser`) to process the responses.
- Identified crucial related areas for further study, including advanced prompt engineering, data serialization/deserialization, Pydantic for schema validation, LangChain Expression Language (LCEL), and robust error handling in LLM chains.

**Resources:**

- [notes.ipynb](./Section41/notes.ipynb)
- [codes1.ipynb](./Section41/codes1.ipynb)
- [codes2.ipynb](./Section41/codes2.ipynb)
- [codes3.ipynb](./Section41/codes3.ipynb)

## Day 42: LangChain Module: LangChain Expression Language (LCEL)

**What I did today:**

- Gained a foundational understanding of LangChain Expression Language (LCEL), including its core Runnable protocol, the pipe operator (`|`) for intuitive chain composition, and the critical role of structured output parsing for downstream tasks.
- Developed hands-on experience with essential LCEL methods such as `invoke` for single executions, `batch` for efficient parallel processing of multiple inputs, and `stream` for generating real-time, token-by-token responses using Python generators.
- Explored the underlying object model of LCEL, recognizing chains as `RunnableSequence` instances and understanding how components like prompt templates, models, and parsers inherit `Runnable` capabilities for consistent interaction.
- Mastered techniques for piping multiple LCEL chains and components together, utilizing `RunnablePassthrough` (especially with its `.assign()` method and dictionary structuring) to effectively manage data flow, reshape inputs, and ensure compatibility between chained elements.
- Implemented concurrent execution of distinct processing paths on a single input using `RunnableParallel`, including its convenient implicit dictionary-based syntax, thereby optimizing complex workflows.
- Practiced integrating custom Python functions and complex logic into LCEL chains by leveraging `RunnableLambda` and the more Pythonic `@chain` decorator, allowing for bespoke data transformations and computations within sequences.
- Successfully engineered a complete, stateful conversational AI by integrating `ConversationSummaryMemory` into an LCEL chain, which involved dynamically preparing inputs with `RunnablePassthrough.assign()`, extracting nested data using `operator.itemgetter` wrapped in `RunnableLambda`, generating responses, and explicitly saving conversation context.
- Designed and implemented a reusable, memory-enabled runnable function by encapsulating the entire conversational logic, including memory interaction and state updates, using the `@chain` decorator to simplify the development of stateful applications.
- Utilized the "Gandalf" library to generate ASCII visualizations of LCEL chains, enhancing comprehension of their structure, data flow, and the execution plan of both sequential and parallel operations.

**Resources:**

- [notes.ipynb](./Section42/notes.ipynb)
- [codes1.ipynb](./Section42/codes1.ipynb)
- [codes2.ipynb](./Section42/codes2.ipynb)
- [codes3.ipynb](./Section42/codes3.ipynb)
- [codes4.ipynb](./Section42/codes4.ipynb)
- [codes5.ipynb](./Section42/codes5.ipynb)
- [codes6.ipynb](./Section42/codes6.ipynb)
- [codes7.ipynb](./Section42/codes7.ipynb)
- [codes8.ipynb](./Section42/codes8.ipynb)
- [codes9.ipynb](./Section42/codes9.ipynb)
- [codes10.ipynb](./Section42/codes10.ipynb)
- [codes11.ipynb](./Section42/codes11.ipynb)
- [codes12.ipynb](./Section42/codes12.ipynb)
- [codes13.ipynb](./Section42/codes13.ipynb)
- [codes14.ipynb](./Section42/codes14.ipynb)

## Day 43: LangChain Module: Retrieval Augmented Generation (RAG)

**What I did today:**

- Gained a comprehensive understanding of methods for integrating custom data with LLMs, focusing on Retrieval Augmented Generation (RAG) and its core components: Indexing (loading, splitting, embedding, storing), Retrieval (similarity search, diverse retrieval methods), and Generation (context-aware response formulation).
- Mastered document loading for various formats (PDFs with `PyPDFLoader`, DOCX with `Docx2txtLoader`) into LangChain `Document` objects, including practical text preprocessing techniques such as newline removal for token optimization and cost efficiency.
- Explored and implemented multiple document splitting strategies in LangChain, including character-based splitting with `CharacterTextSplitter` (configuring chunk size, overlap, and separators) and semantic splitting using markdown headers with `MarkdownHeaderTextSplitter` for improved topical coherence and metadata enrichment.
- Successfully generated high-dimensional text embeddings from processed document chunks using OpenAI models via LangChain, and quantitatively measured semantic similarity between them using dot products with NumPy, confirming vector normalization and its implications for cosine similarity (where for normalized vectors, $\cos \theta = \vec{a} \cdot \vec{b}$).
- Developed hands-on proficiency in creating, persisting, and loading Chroma vector stores, including batch embedding of documents and performing CRUD operations (add, get, update, delete documents) to effectively manage the vector store's content lifecycle.
- Implemented and compared different document retrieval strategies, including basic `similarity_search` and advanced `Maximal Marginal Relevance (MMR)` search, leveraging metadata filtering to enhance result diversity and relevance for improved contextual input to LLMs.
- Constructed an end-to-end Retrieval Augmented Generation (RAG) pipeline using LangChain Expression Language (LCEL), by creating a runnable retriever object (`as_retriever()`) and chaining it with dynamic prompt templates (utilizing `RunnableParallel` and `RunnablePassthrough`), an LLM (`ChatOpenAI`), and an output parser (`StrOutputParser`).
- Applied the "stuffing" method to inject retrieved context into LLM prompts for response generation, critically evaluated its advantages and limitations (such as context window constraints and the "lost in the middle" problem), and was introduced to alternative strategies like "document refinement" for handling extensive contextual information.

**Resources:**

- [notes.ipynb](./Section43/notes.ipynb)
- [codes1.ipynb](./Section43/codes1.ipynb)
- [codes2.ipynb](./Section43/codes2.ipynb)
- [codes3.ipynb](./Section43/codes3.ipynb)
- [codes4.ipynb](./Section43/codes4.ipynb)
- [codes5.ipynb](./Section43/codes5.ipynb)
- [codes6.ipynb](./Section43/codes6.ipynb)
- [codes7.ipynb](./Section43/codes7.ipynb)
- [codes8.ipynb](./Section43/codes8.ipynb)
- [codes9.ipynb](./Section43/codes9.ipynb)
- [codes10.ipynb](./Section43/codes10.ipynb)
- [codes11.ipynb](./Section43/codes11.ipynb)
- [codes12.ipynb](./Section43/codes12.ipynb)

## Day 44: LangChain Module: Tools and Agents

**What I did today:**

- Developed a strong conceptual understanding of LangChain agents, including their ability to reason, dynamically select tools, and orchestrate multi-step task execution, moving beyond simple context awareness to enable LLM-driven actions.
- Learned the core architecture of LangChain agents, encompassing tools (defined by clear names, descriptions, and input schemas), toolkits, the agent chain structure (prompt with `agent_scratchpad`, LLM, output parser producing `AgentAction` or `AgentFinish`), and the role of the `AgentExecutor` in managing the iterative reasoning loop.
- Gained practical experience in creating and integrating various tools for LangChain agents, including using the `WikipediaTool`, creating specialized data retrieval tools from vector stores with `create_retriever_tool`, and converting custom Python functions into tools using the `@tool` decorator.
- Utilized LangChain Hub to fetch and inspect pre-defined agent prompt templates (e.g., `"hwchase17/openai-tools-agent"`), understanding the critical role of placeholders like `input`, optional `chat_history`, and particularly `agent_scratchpad` for enabling iterative reasoning.
- Successfully constructed and ran a tool-calling agent using `create_tool_calling_agent` and `AgentExecutor`, configuring it with `verbose=True` and `return_intermediate_steps=True` to observe its decision-making process (tool selection, inputs, observations) transparently.
- Witnessed and analyzed the agent's capability to handle complex, multi-tool queries by observing its problem decomposition and sequential invocation of different tools (e.g., a custom data retriever followed by multiple Wikipedia searches) to synthesize comprehensive answers.

**Resources:**

- [notes.ipynb](./Section44/notes.ipynb)
- [codes1.ipynb](./Section44/codes1.ipynb)
- [codes2.ipynb](./Section44/codes2.ipynb)
- [codes3.ipynb](./Section44/codes3.ipynb)
- [codes4.ipynb](./Section44/codes4.ipynb)
- [codes5.ipynb](./Section44/codes5.ipynb)

## Day 45: Vector Databases Module: Introduction

**What I did today:**

- Reviewed the increasing importance of vector databases in business and data science, covering their theoretical foundations (vector space, search algorithms) and practical implementation aspects.
- Understood the power of semantic search enabled by vector databases, which identifies similarities based on contextual meaning rather than exact keyword matches, and its versatility across data types like text, images, and audio.
- Compared and contrasted SQL, NoSQL, and vector databases:
  - SQL databases: Ideal for structured, relational data requiring transactional integrity.
  - NoSQL databases: Offer flexible, scalable solutions for varied and large-scale data.
  - Vector databases: Specialized in managing high-dimensional vector data crucial for AI/ML applications like semantic search and recommendation systems.
- Learned about high-dimensional vector spaces in vector databases and their importance in representing complex data (text, images, audio) for AI tasks.
- Explored the core capability of vector databases in performing similarity searches by measuring distances between numerical vectors, which are transformations of complex data.
- Identified real-world applications of vector databases, such as reverse image search, personalized recommendations (music, fashion), advanced query matching (healthcare, customer service), and fraud detection.
- Recognized the driving forces behind the growing adoption of vector databases, including escalating data volumes and the demand for sophisticated AI-driven insights.

**Resources:**

- [notes.ipynb](./Section45/notes.ipynb)


## Day 46: Vector Databases Module: Basics of Vector Space and High-Dimensional Data

**What I did today:**

- Understood vector spaces as foundational mathematical structures for representing data as multi-dimensional vectors in vector databases, enabling powerful similarity searches.
- Learned that data (images, text, products) is transformed into vectors in vector databases, with each dimension quantifying a specific attribute for computational analysis.
- Recognized that similarity in vector databases is determined by the proximity of vectors, crucial for recommendation engines, anomaly detection, and semantic search.
- Explored how data is encoded into numerical vectors for compatibility with vector databases, ranging from simple assignments to sophisticated feature extraction.
- Reviewed various distance metrics (Euclidean, Manhattan, Dot Product, Cosine Similarity) used to quantify similarity between vectors, each suited for different data types and application goals.
- Understood that Euclidean distance measures the straight-line distance, Manhattan distance sums absolute differences along Cartesian coordinates, Dot Product reflects alignment and magnitude, and Cosine Similarity measures the angle between vectors, focusing on orientation.
- Learned about the embedding process in NLP and ML, where raw data is transformed into dense vector representations (embeddings) that capture semantic meaning and context.
- Recognized that embeddings are essential for machines to process complex data by converting it into structured numerical vectors.
- Understood that modern embeddings capture contextual meaning, representing the same word differently based on its surrounding context.
- Noted that practical embeddings use high dimensionality (hundreds or thousands of dimensions) to capture subtle and complex data relationships.

**Resources:**

- [notes.ipynb](./Section46/notes.ipynb)

## Day 47: Vector Databases Module: Introduction to The Pinecone Vector Database

**What I did today:**

- Reviewed popular vector databases (Pinecone, Milvus, Weaviate, Qdrant), comparing their strengths, weaknesses, and ideal use cases, and understood the rationale for using Pinecone in the course.
- Learned about the trade-offs between managed services (like Pinecone) and self-hosted open-source solutions, impacting scalability, cost, and control.
- Explored Pinecone's features, including its passwordless registration, workspace organization, API key management, and the limitations of the free "starter" plan.
- Mastered the creation of a Pinecone index, defining critical parameters like name, vector dimensions (which must match the embedding model), and similarity metric.
- Established a secure connection to Pinecone using Python, emphasizing the use of `.env` files for storing API keys and environment identifiers.
- Verified Pinecone connection by listing existing indexes programmatically.
- Learned to programmatically create and delete Pinecone indexes in Python, including checking for an index's existence before creation to prevent errors.
- Understood and implemented the "upsert" operation to add or update data in a Pinecone index, formatting data as a list of tuples (ID, vector_values, metadata).
- Recognized the critical impact of data representation (feature choice and dimensionality) on similarity search quality.
- Practiced loading large text datasets (e.g., Hugging Face "FineWeb") using `IterableDataset` for memory-efficient processing.
- Converted text data to vector embeddings using a sentence transformer model, ensuring the Pinecone index dimension matched the embedding model's output (e.g., 384 dimensions).
- Implemented batch upserting to efficiently upload vectors and associated metadata to Pinecone, handling a subset of a large dataset for practical demonstration.

**Resources:**

- [notes.ipynb](./Section47/notes.ipynb)

## Day 48: Vector Databases Module: Semantic Search with Pinecone and Custom (Case Study)

**What I did today:**

- Understood semantic search as a method to find information based on meaning, contrasting it with traditional keyword-based search, and outlined a case study using Pinecone with existing tabular data.
- Analyzed the deficiencies of exact-match search on educational platforms and proposed semantic search at course and section levels to improve content discoverability, focusing on retrieval accuracy over generative AI summarization.
- Learned to prepare tabular data for semantic search by merging relevant text columns (e.g., course name, slug, technology, topic) into a single descriptive string per record using Python and pandas to create richer vector embeddings.
- Securely managed API credentials using `.env` files and established connections to Pinecone, covering library imports, `python-dotenv` usage (including IPython magic commands), and Pinecone client initialization.
- Reviewed the evolution of text embedding algorithms from traditional methods (BoW, TF-IDF) to context-aware models (Word2Vec, BERT, ELMo), emphasizing the role of contextual understanding and introducing the Sentence Transformers library.
- Generated vector embeddings from course descriptions using Sentence Transformers, ensuring the embedding model's output dimension matched the Pinecone index configuration, and upserted these into Pinecone.
- Implemented semantic search in Python by vectorizing text queries, using Pinecone's `query()` function with `top_k` and `include_metadata=True`, and refined results using score thresholding and robust metadata handling with `.get()`.
- Recognized that data quality and granularity are often more critical than the choice of embedding algorithm for search effectiveness and explored strategies to incorporate section-level details by creating composite IDs and rich metadata for each course-section pair.
- Upserted section-level data (IDs, embeddings, metadata) into a new Pinecone index and verified the process and metadata utility on the Pinecone platform, noting its basic filtering capabilities.
- Improved search relevance significantly by switching to a BERT-based embedding model (768 dimensions) optimized for semantic search, in conjunction with the granular section-level data, which successfully surfaced previously missed relevant content.
- Explored the broader applications of vector databases, including item-based and user-based recommendation systems, semantic image search (leveraging CNNs or Siamese Networks for embeddings), and their use in biomedical research for tasks like drug discovery and gene expression analysis.

**Resources:**

- [notes.ipynb](./Section48/notes.ipynb)

