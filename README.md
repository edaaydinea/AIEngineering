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