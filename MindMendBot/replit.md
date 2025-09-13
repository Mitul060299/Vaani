# Overview

This is a full-stack mental health support chatbot web application built with Flask and Python. The application provides empathetic AI-powered conversations with multi-modal input capabilities including text, voice, and facial emotion analysis. It features crisis detection and intervention, session-based chat history, and text-to-speech responses. The chatbot is designed to be a supportive companion while encouraging professional help when needed.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Single Page Application**: HTML template with embedded CSS and JavaScript
- **Multi-modal Input Interface**: Three interaction modes via buttons
  - Text input with standard form submission
  - Voice recording (5-second audio capture saved as WAV)
  - Combined facial and voice capture (simultaneous photo + audio recording)
- **Real-time Chat Interface**: Dynamic message appending with chat history display
- **Audio Playback**: Automatic TTS response playback using HTML5 audio element
- **Responsive Design**: Mobile-friendly interface with gradient styling

## Backend Architecture
- **Flask Web Framework**: Lightweight Python web server
- **Session Management**: Flask sessions with secret key for per-user chat history (3 message limit)
- **Multi-modal Processing Pipeline**: 
  - Speech recognition using Google's speech-to-text API
  - Image processing with OpenCV and PIL
  - Audio file handling with temporary storage and cleanup
- **AI/ML Model Integration**: Multiple transformer models for various analyses
  - Text generation models (Phi-3 Mini or Gemma-2b)
  - Sentiment analysis (RoBERTa or DistilBERT)
  - Emotion detection for text, voice, and facial expressions
  - Toxicity detection and safety filtering
- **RAG System**: Knowledge retrieval using sentence embeddings and FAISS vector search
- **Crisis Detection**: Keyword-based and embedding similarity safety checks

## Data Storage Solutions
- **Session-based Storage**: Flask sessions for temporary chat history
- **Knowledge Base**: Excel file (RAG_Knowledge_Base_WithID.xlsx) with fallback to hardcoded responses
- **Temporary File Management**: Short-lived audio/image files with automatic cleanup
- **Vector Storage**: FAISS index for document embeddings and similarity search

## Safety and Content Filtering
- **Crisis Intervention**: Keyword detection with immediate crisis helpline information
- **Toxicity Filtering**: ML-based unsafe content detection with confidence thresholds
- **Duplicate Response Prevention**: Embedding-based similarity checking
- **Content Moderation**: Multi-layered safety checks before response generation

## Response Generation Pipeline
- **Context Analysis**: Emotion, sentiment, and aspect-based sentiment analysis
- **Intent Detection**: RAG retrieval for advice-seeking queries
- **Prompt Engineering**: Dynamic prompt construction based on user emotion and context
- **Safety Integration**: Pre and post-generation safety validation
- **Audio Synthesis**: Google TTS with base64 encoding for seamless playback

# External Dependencies

## AI/ML Models and Services
- **Hugging Face Transformers**: Multiple pre-trained models for NLP tasks
  - Text generation (Microsoft Phi-3, Google Gemma)
  - Sentiment analysis (Cardiff RoBERTa, DistilBERT)
  - Emotion detection (various specialized models)
  - Toxicity detection (Unitary Toxic-BERT)
- **Sentence Transformers**: all-MiniLM-L6-v2 for embeddings
- **Google Speech Recognition**: Web Speech API for voice transcription
- **Google Text-to-Speech (gTTS)**: Audio response generation

## Computer Vision and Audio Processing
- **OpenCV**: Image processing and facial analysis
- **PIL (Python Imaging Library)**: Image manipulation and format handling
- **SpeechRecognition Library**: Audio transcription interface
- **NumPy/Pandas**: Data processing and manipulation

## Vector Search and Retrieval
- **FAISS**: Facebook AI Similarity Search for efficient vector operations
- **Custom RAG Implementation**: Document retrieval and relevance scoring

## Python Web Stack
- **Flask**: Core web framework with session management
- **Standard Libraries**: os, time, base64, tempfile, secrets, datetime
- **File Processing**: Temporary file creation and cleanup utilities

## Deployment Platform
- **Replit Environment**: Cloud-based development and hosting platform
- **Network Configuration**: Host on 0.0.0.0:8080 for Replit compatibility
- **Environment Variables**: SESSION_SECRET for security configuration