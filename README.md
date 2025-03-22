### AIEngineering Repository  

This repository contains materials, notes, and code for the Udemy course: [**The AI Engineer Course - Complete AI Engineer Bootcamp**](https://www.udemy.com/course/the-ai-engineer-course-complete-ai-engineer-bootcamp). Additionally, it includes personal notes, insights, and projects created throughout the learning process.  

---

## 📑 Table of Contents  

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

