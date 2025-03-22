### AIEngineering Repository  

This repository contains materials, notes, and code for the Udemy course: [**The AI Engineer Course - Complete AI Engineer Bootcamp**](https://www.udemy.com/course/the-ai-engineer-course-complete-ai-engineer-bootcamp). Additionally, it includes personal notes, insights, and projects created throughout the learning process.  

---

## 📑 Table of Contents  

1. [📦 Installed Packages](#-installed-packages)  
2. [🏗️ Setting Up the Conda Environment](#️-setting-up-the-conda-environment)  
   - [1️⃣ Create and Activate the Environment](#1️⃣-create-and-activate-the-environment)  
   - [2️⃣ Install Required Packages](#2️⃣-install-required-packages)  
   - [3️⃣ Download Language Model for spaCy](#3️⃣-download-language-model-for-spacy)  
   - [4️⃣ Install Jupyter & Configure Kernel](#4️⃣-install-jupyter--configure-kernel)  
3. [📖 Course Notes](#-course-notes)  
4. [📝 Personal Notes & Projects](#-personal-notes--projects)  
5. [📆 Daily Progress](#-daily-progress)  

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

## 📖 Course Notes  

This section contains structured notes from *The AI Engineer Course - Complete AI Engineer Bootcamp*, including key concepts, summaries, and code snippets.  

---

## 📝 Personal Notes & Projects  

This section includes additional insights, self-created notes, and projects beyond the course material. Topics may include further research, advanced implementations, and real-world applications of AI concepts.  

---

## 📆 Daily Progress  

A daily log to track learning progress, document challenges, and reflect on new concepts.  

- **Day 20: NLP Module: Introduction** 
 

📌 Note: Days 1-19 cover foundational Python programming and general concepts, which will not be shared here. From Day 20 onward, the focus shifts to NLP, LLMs, and Speech Recognition, and relevant notes will be documented. 🚀