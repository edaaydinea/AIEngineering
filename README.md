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

- [notes.ipynb](./Section23/notes.ipynb)
- [codes.ipynb](./Section23/codes.ipynb)
- [codes2.ipynb](./Section23/codes2.ipynb)
- [practical.ipynb](./Section23/practical.ipynb)