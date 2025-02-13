# Amazon Chat Insights - Sentiment Analysis & Summarization

## Overview
This project aims to build a scalable and modular machine learning system for sentiment analysis and summarization of Amazon chat conversations. The system leverages Hugging Face transformers, PyTorch, and FastAPI for efficient model training, inference, and deployment.

---

## Work Completed
Due to time constraints and work commitments, I was unable to complete the full assignment. However, I made significant progress in the following areas:

### 1. Modular Coding Approach
- Ensured a clean and structured codebase for better maintainability and reusability.
- Followed best practices for handling data, model loading, and inference.

### 2. Scalable ML System Design
- Integrated FastAPI for serving the models with proper endpoint structuring.
- Used MongoDB for data ingestion, ensuring smooth data handling.
- Designed a flexible training pipeline with `Trainer` from Hugging Face.

### 3. Model Training & Deployment
- Trained a `RoBERTa`-based sentiment classification model and saved it.
- Used a `BART` model for text summarization.
- Implemented endpoints for prediction using trained models.

## Features
- **Sentiment Analysis:** Classifies the sentiment of chat conversations using a fine-tuned `RoBERTa` model.
- **Text Summarization:** Generates concise summaries of chat conversations using a `BART` model.
- **Scalable API:** Built with FastAPI for high-performance and easy integration.
- **Modular Design:** Clean and structured codebase for maintainability and extensibility.

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
