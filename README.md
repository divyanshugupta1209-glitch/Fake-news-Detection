# 📰 NEO-REFUTE

## Multimodal Fake News Detection & Analysis System

---

## 📌 Project Overview

NEO-REFUTE is a **Final Year Team Project** developed to analyze news claims using Artificial Intelligence, Natural Language Processing (NLP), and multimodal analysis.

The system allows users to submit a news claim and optionally upload an image. It analyzes multiple signals to classify the claim as:

- 🟢 **REAL**
- 🔴 **FAKE**
- 🟡 **UNCERTAIN**

The project combines AI-powered text analysis, BERT-based classification, source credibility analysis, image-text consistency checking, OCR, AI-generated image detection, and supporting analysis modules.

---

## 🚀 Live Demo

🔗 **[Open NEO-REFUTE Live Demo](https://fake-news-detection-jr6dmne67mmyxn25ufe7xk.streamlit.app/)**

---

## ✨ Key Features

- 📰 News claim analysis
- 🤖 AI-powered text analysis
- 🧠 REAL / FAKE / UNCERTAIN classification
- 📊 Confidence score display
- 🎚️ Adjustable confidence threshold
- 📰 Configurable related article fetching
- 🖼️ Optional image analysis
- 📄 OCR-based text extraction
- 🤖 AI-generated image detection
- 🌐 Source credibility analysis
- 🧩 Multimodal text and image analysis
- ⚠️ Uncertainty handling
- ⚡ Cached results for faster repeated analysis
- 📝 Prediction explanations and analysis details
- 📄 PDF report generation
- 🎨 Interactive Streamlit web interface

---

## 🧠 How NEO-REFUTE Works

```text
User News Claim
       │
       ▼
AI Text Analysis
       │
       ├── Text Classification
       ├── Source Credibility Analysis
       ├── Supporting Analysis Modules
       └── Uncertainty Detection
       │
       ▼
Optional Image Analysis
       │
       ├── Image-Text Consistency
       ├── OCR Analysis
       └── AI-Generated Image Detection
       │
       ▼
Multimodal Fusion
       │
       ▼
REAL / FAKE / UNCERTAIN
       │
       ▼
Confidence Score + Explanation
       │
       ▼
PDF Analysis Report
```

---

## 🛠️ Technologies Used

### Programming Language

- Python

### AI / Machine Learning

- BERT
- PyTorch
- Hugging Face Transformers
- OpenRouter AI

### Image and Text Analysis

- Image-Text Consistency Analysis
- OCR
- AI-Generated Image Detection

### Web Application

- Streamlit

### Supporting Libraries

- Pandas
- NumPy
- Scikit-learn
- NLTK

---

## 📂 Project Structure

```text
NEO-REFUTE/
│
├── app.py
├── requirements.txt
├── README.md
│
├── utils/
│   ├── detection.py
│   ├── db.py
│   ├── image_expert.py
│   ├── image_ocr.py
│   ├── ai_image_detector.py
│   └── explanation_generator.py
│
├── data/
└── models/
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/divyanshugupta1209-glitch/Fake-news-Detection.git
```

### 2. Move to the Project Directory

```bash
cd Fake-news-Detection
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file and add the required API key:

```env
OPENROUTER_API_KEY=your_api_key_here
```

### 5. Run the Application

```bash
streamlit run app.py
```

---

## 🎯 Project Objective

The objective of NEO-REFUTE is to develop an AI-assisted news analysis system that uses multiple signals instead of relying only on a single text-classification model.

The system aims to provide a more comprehensive analysis by considering:

- Textual information
- Optional visual information
- Source-related analysis
- Confidence levels
- Uncertainty

---

## ⚠️ Important Disclaimer

NEO-REFUTE is an **AI-assisted news analysis system** and is not a replacement for professional fact-checking organizations.

Predictions may not always be correct. Important information should always be verified using reliable and authoritative sources.

Future events, predictions, rumors, opinions, and claims without sufficient verifiable evidence may be classified as **UNCERTAIN**.

---

## 👨‍💻 My Contribution

This project was completed as a **Final Year Team Project**.

My contribution included:

- Contributing to project implementation and module integration
- Testing and validation of REAL, FAKE, and UNCERTAIN predictions
- Project documentation
- Streamlit application deployment
- Deployment-related testing and maintenance

---

## 👥 Team Members

- Jayavardhana Kumar K
- Divyanshu Gupta
- Azeem Saeed Mohamed M
- Daynish Manane S

---

## 📜 License

This project was developed for **academic and educational purposes**.
