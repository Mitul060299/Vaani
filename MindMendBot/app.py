import os
import time
import base64
import tempfile
import random
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, session, send_file, render_template
import speech_recognition as sr
from gtts import gTTS
import cv2
from PIL import Image
import io
import secrets
import re
from datetime import datetime

app = Flask(__name__)

# Set session secret key
app.secret_key = os.environ.get('SESSION_SECRET', secrets.token_hex(16))

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Crisis keywords for safety detection
CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end my life", "want to die", "hurt myself", 
    "self harm", "cutting", "overdose", "jump off", "hang myself",
    "worthless", "hopeless", "can't go on", "end it all", "better off dead"
]

# Mental health knowledge base (simplified)
KNOWLEDGE_BASE = {
    "anxiety": "Anxiety is normal, but persistent worry can be managed with breathing exercises, mindfulness, and professional help. Consider deep breathing: inhale for 4, hold for 4, exhale for 4.",
    "depression": "Depression affects many people. Small daily activities, social connection, and professional support can help. You're not alone in this journey.",
    "stress": "Stress can be managed through regular exercise, good sleep habits, time management, and relaxation techniques. Break big problems into smaller, manageable steps.",
    "sleep": "Good sleep hygiene includes consistent bedtime, avoiding screens before bed, and creating a calm environment. Most adults need 7-9 hours of sleep.",
    "relationships": "Healthy relationships involve communication, respect, and boundaries. It's okay to seek help if relationships become difficult or harmful.",
    "self_care": "Self-care isn't selfish. Regular exercise, healthy eating, social connection, and activities you enjoy are essential for mental health."
}

# Crisis intervention message
CRISIS_MESSAGE = """I'm concerned about you. Please reach out for immediate help:
- National Suicide Prevention Lifeline: 988 or 1-800-273-8255
- Crisis Text Line: Text HOME to 741741
- International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/
- Or go to your nearest emergency room
You matter, and help is available."""

def is_safe_response(text):
    """Check if response contains unsafe content"""
    unsafe_words = ['kill', 'hurt', 'harm', 'die', 'suicide']
    text_lower = text.lower()
    return not any(word in text_lower for word in unsafe_words)

def deduplicate_response(new_response, history):
    """Simple deduplication to avoid repeating responses"""
    if not history:
        return new_response
    
    recent_responses = [msg['response'] for msg in history[-3:] if 'response' in msg]
    if new_response in recent_responses:
        alternatives = [
            "I understand you're going through a difficult time. How are you feeling right now?",
            "Thank you for sharing with me. What's been on your mind today?",
            "I'm here to listen. Can you tell me more about what you're experiencing?"
        ]
        return random.choice(alternatives)
    return new_response

def detect_sentiment(text):
    """Simple rule-based sentiment analysis"""
    positive_words = ['happy', 'good', 'great', 'awesome', 'wonderful', 'excited', 'love', 'amazing', 'fantastic', 'excellent']
    negative_words = ['sad', 'bad', 'terrible', 'awful', 'horrible', 'hate', 'angry', 'frustrated', 'depressed', 'anxious', 'worried']
    
    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count > negative_count:
        return 'positive'
    elif negative_count > positive_count:
        return 'negative'
    else:
        return 'neutral'

def detect_text_emotion(text):
    """Simple rule-based emotion detection"""
    emotions = {
        'sad': ['sad', 'cry', 'tears', 'depressed', 'down', 'blue', 'melancholy'],
        'angry': ['angry', 'mad', 'furious', 'rage', 'upset', 'irritated', 'frustrated'],
        'anxious': ['anxious', 'worried', 'nervous', 'scared', 'afraid', 'panic', 'stress'],
        'happy': ['happy', 'joy', 'excited', 'cheerful', 'glad', 'delighted', 'elated'],
        'neutral': []
    }
    
    text_lower = text.lower()
    emotion_scores = {}
    
    for emotion, words in emotions.items():
        score = sum(1 for word in words if word in text_lower)
        emotion_scores[emotion] = score
    
    if max(emotion_scores.values()) == 0:
        return 'neutral'
    
    return max(emotion_scores.keys(), key=lambda k: emotion_scores[k])

def is_unsafe_message(text):
    """Check if message contains crisis keywords"""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in CRISIS_KEYWORDS)

def retrieve_docs(query):
    """Simple knowledge base retrieval"""
    query_lower = query.lower()
    relevant_docs = []
    
    for topic, content in KNOWLEDGE_BASE.items():
        if topic in query_lower or any(word in query_lower for word in topic.split('_')):
            relevant_docs.append(content)
    
    return relevant_docs[:3]  # Return top 3 matches

def build_prompt(user_input, history, sentiment=None, emotion=None, voice_emotion=None, facial_emotion=None, retrieved_docs=None):
    """Build prompt for response generation"""
    
    # Get recent history
    recent_history = ""
    if history:
        for msg in history[-3:]:
            if 'user_input' in msg and 'response' in msg:
                recent_history += f"User: {msg['user_input']}\nAssistant: {msg['response']}\n"
    
    # Build context
    context = "You are a compassionate mental health support chatbot. Provide empathetic, supportive responses. Keep responses to 1-3 sentences."
    
    if emotion and emotion != 'neutral':
        context += f" The user seems to be feeling {emotion}."
    
    if voice_emotion and voice_emotion != 'neutral':
        context += f" Their voice suggests they're feeling {voice_emotion}."
    
    if facial_emotion and facial_emotion != 'neutral':
        context += f" Their facial expression shows {facial_emotion}."
    
    if retrieved_docs:
        context += f" Relevant information: {' '.join(retrieved_docs)}"
    
    prompt = f"{context}\n\nRecent conversation:\n{recent_history}\nUser: {user_input}\nAssistant:"
    return prompt

def generate_simple_response(prompt):
    """Generate a simple empathetic response"""
    
    user_input = prompt.split("User: ")[-1].split("\nAssistant:")[0].lower()
    
    # Crisis response
    if is_unsafe_message(user_input):
        return CRISIS_MESSAGE
    
    # Check if knowledge base information is in the prompt
    knowledge_info = ""
    if "Relevant information:" in prompt:
        knowledge_section = prompt.split("Relevant information: ")[1].split("\n")[0]
        knowledge_info = knowledge_section.strip()
    
    # Specific responses based on keywords
    if any(word in user_input for word in ['hello', 'hi', 'hey']):
        responses = [
            "Hello! I'm here to listen and support you. How are you feeling today?",
            "Hi there! I'm glad you reached out. What's on your mind?",
            "Hello! I'm here to help. How can I support you today?"
        ]
        return random.choice(responses)
    
    if any(word in user_input for word in ['help', 'support', 'advice']):
        if knowledge_info:
            return f"Here's some guidance that might help: {knowledge_info} Would you like to talk more about this?"
        else:
            responses = [
                "I'm here to help you through this. Can you tell me more about what you're experiencing?",
                "Thank you for reaching out for support. That takes courage. What's been difficult for you?",
                "I want to help you. Can you share what's been weighing on your mind?"
            ]
            return random.choice(responses)
    
    if any(word in user_input for word in ['anxious', 'anxiety', 'worried', 'nervous']):
        if knowledge_info:
            return f"I understand you're feeling anxious. {knowledge_info} What's been causing you to feel this way?"
        else:
            responses = [
                "Anxiety can feel overwhelming. Try taking slow, deep breaths. What's causing you to feel anxious?",
                "I understand anxiety is difficult. Grounding techniques like naming 5 things you can see can help. What's troubling you?",
                "Anxiety is challenging but manageable. Would you like to talk about what's making you feel this way?"
            ]
            return random.choice(responses)
    
    if any(word in user_input for word in ['sad', 'depressed', 'down', 'blue']):
        if knowledge_info:
            return f"I'm sorry you're feeling this way. {knowledge_info} What's been bothering you lately?"
        else:
            responses = [
                "I'm sorry you're feeling sad. Your feelings are valid, and I'm here to listen. What's been bothering you?",
                "Sadness is a natural emotion, but you don't have to face it alone. Can you tell me more?",
                "I hear that you're feeling down. Thank you for sharing that with me. What's been on your mind?"
            ]
            return random.choice(responses)
    
    # General supportive responses - incorporate knowledge base if available
    if knowledge_info:
        general_responses = [
            f"Thank you for sharing that with me. {knowledge_info} How are you feeling about this?",
            f"I'm here to listen and support you. {knowledge_info} Can you tell me more about what you're experiencing?",
            f"I appreciate you opening up to me. {knowledge_info} What would be most helpful for you right now?"
        ]
    else:
        general_responses = [
            "Thank you for sharing that with me. How are you feeling about this situation?",
            "I'm here to listen and support you. Can you tell me more about what you're experiencing?",
            "It sounds like you're going through something difficult. I'm here for you.",
            "I appreciate you opening up to me. What would be most helpful for you right now?",
            "Your feelings are important and valid. How can I best support you today?"
        ]
    
    return random.choice(general_responses)

def generate_response_pipeline(user_input, history, voice_emotion=None, facial_emotion=None):
    """Main response generation pipeline"""
    
    # Safety check
    if is_unsafe_message(user_input):
        return CRISIS_MESSAGE, None, None, None, None
    
    # Detect sentiment and emotion
    sentiment = detect_sentiment(user_input)
    emotion = detect_text_emotion(user_input)
    
    # Check if advice is needed OR if user mentions mental health topics
    advice_keywords = ['what should i do', 'advice', 'help me', 'suggestion', 'recommend']
    mental_health_keywords = ['anxiety', 'anxious', 'depression', 'depressed', 'stress', 'stressed', 
                             'sleep', 'insomnia', 'relationship', 'self care', 'worried', 'nervous',
                             'sad', 'down', 'blue', 'panic', 'overwhelmed']
    
    needs_advice = any(keyword in user_input.lower() for keyword in advice_keywords)
    mentions_mh_topic = any(keyword in user_input.lower() for keyword in mental_health_keywords)
    
    retrieved_docs = None
    if needs_advice or mentions_mh_topic:
        retrieved_docs = retrieve_docs(user_input)
    
    # Build prompt with voice and facial emotions
    prompt = build_prompt(user_input, history, sentiment, emotion, voice_emotion, facial_emotion, retrieved_docs)
    
    # Generate response
    response = generate_simple_response(prompt)
    
    # Deduplicate
    response = deduplicate_response(response, history)
    
    return response, sentiment, emotion, voice_emotion, facial_emotion

def generate_audio_base64(text):
    """Generate audio from text and return base64 encoded"""
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            tts.save(temp_file.name)
            
            # Read and encode to base64
            with open(temp_file.name, 'rb') as audio_file:
                audio_data = audio_file.read()
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            # Clean up
            os.unlink(temp_file.name)
            
            return audio_base64
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

def detect_voice_emotion(audio_file_path):
    """Simple voice emotion detection (placeholder)"""
    # This is a simplified version - in the full implementation this would use actual ML models
    emotions = ['neutral', 'sad', 'happy', 'angry', 'anxious']
    return random.choice(emotions)

def detect_facial_emotion(image_file_path):
    """Simple facial emotion detection using OpenCV"""
    try:
        # Load the image
        image = cv2.imread(image_file_path)
        if image is None:
            return 'neutral'
        
        # Load face cascade with proper path
        try:
            # Try to get the data path, fallback to direct filename
            try:
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            except:
                cascade_path = 'haarcascade_frontalface_default.xml'
            
            face_cascade = cv2.CascadeClassifier(cascade_path)
            if face_cascade.empty():
                return 'neutral'
        except Exception as e:
            print(f"Could not load face cascade: {e}")
            return 'neutral'
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # For simplicity, return random emotion if face detected
            emotions = ['neutral', 'happy', 'sad', 'angry', 'surprised']
            return random.choice(emotions)
        else:
            return 'neutral'
            
    except Exception as e:
        print(f"Facial emotion detection error: {e}")
        return 'neutral'

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/process_text', methods=['POST'])
def process_text():
    """Process text input"""
    try:
        data = request.get_json()
        user_input = data.get('message', '').strip()
        
        if not user_input:
            return jsonify({'error': 'No message provided'}), 400
        
        # Get chat history
        history = session.get('chat_history', [])
        
        # Generate response
        response, sentiment, emotion, voice_emotion, facial_emotion = generate_response_pipeline(user_input, history)
        
        # Generate audio
        audio_base64 = generate_audio_base64(response)
        
        # Add to history
        history.append({
            'user_input': user_input,
            'response': response,
            'sentiment': sentiment,
            'emotion': emotion,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 3 conversations
        session['chat_history'] = history[-3:]
        
        return jsonify({
            'reply': response,
            'audio': audio_base64,
            'emotion': emotion,
            'sentiment': sentiment
        })
        
    except Exception as e:
        print(f"Text processing error: {e}")
        return jsonify({'error': 'Processing failed'}), 500

@app.route('/process_voice', methods=['POST'])
def process_voice():
    """Process voice input"""
    try:
        # Get the audio file
        audio_file = request.files.get('audio')
        if not audio_file:
            return jsonify({'error': 'No audio file provided'}), 400
        
        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            audio_file.save(temp_audio.name)
            
            # Transcribe audio - simplified version
            try:
                with sr.AudioFile(temp_audio.name) as source:
                    audio_data = recognizer.record(source)
                    # Simplified transcription - in real implementation would use Google API
                    text = "I heard your voice message"  # Placeholder
            except:
                text = "[Voice not transcribed]"
            
            # Detect voice emotion
            voice_emotion = detect_voice_emotion(temp_audio.name)
            
            # Clean up
            os.unlink(temp_audio.name)
        
        # Get chat history
        history = session.get('chat_history', [])
        
        # Generate response
        if text == "[Voice not transcribed]":
            response = "I couldn't understand your voice clearly, but I'm here to listen. Can you try typing your message or speaking again?"
            audio_base64 = generate_audio_base64(response)
        else:
            response, sentiment, emotion, _, _ = generate_response_pipeline(text, history)
            # Add voice emotion context to response
            if voice_emotion != 'neutral':
                response = f"I can hear the {voice_emotion} in your voice. " + response
            audio_base64 = generate_audio_base64(response)
        
        # Add to history
        history.append({
            'user_input': text,
            'response': response,
            'voice_emotion': voice_emotion,
            'timestamp': datetime.now().isoformat()
        })
        
        session['chat_history'] = history[-3:]
        
        return jsonify({
            'transcription': text,
            'reply': response,
            'audio': audio_base64,
            'voice_emotion': voice_emotion
        })
        
    except Exception as e:
        print(f"Voice processing error: {e}")
        return jsonify({'error': 'Voice processing failed'}), 500

@app.route('/process_face_voice', methods=['POST'])
def process_face_voice():
    """Process combined facial and voice input"""
    try:
        # Get files
        audio_file = request.files.get('audio')
        image_file = request.files.get('image')
        
        if not audio_file or not image_file:
            return jsonify({'error': 'Both audio and image files required'}), 400
        
        # Process audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            audio_file.save(temp_audio.name)
            
            try:
                with sr.AudioFile(temp_audio.name) as source:
                    audio_data = recognizer.record(source)
                    # Simplified transcription - in real implementation would use Google API  
                    text = "I heard your voice message"  # Placeholder
            except:
                text = "[Voice not transcribed]"
            
            voice_emotion = detect_voice_emotion(temp_audio.name)
            os.unlink(temp_audio.name)
        
        # Process image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_image:
            image_file.save(temp_image.name)
            facial_emotion = detect_facial_emotion(temp_image.name)
            os.unlink(temp_image.name)
        
        # Get chat history
        history = session.get('chat_history', [])
        
        # Generate response
        if text == "[Voice not transcribed]":
            response = "I couldn't understand your voice clearly, but I can see your expression. I'm here to support you."
        else:
            response, sentiment, emotion, _, _ = generate_response_pipeline(text, history)
        
        # Add emotional context
        emotions_detected = []
        if voice_emotion != 'neutral':
            emotions_detected.append(f"voice sounds {voice_emotion}")
        if facial_emotion != 'neutral':
            emotions_detected.append(f"expression looks {facial_emotion}")
        
        if emotions_detected:
            emotion_context = f"I can see your {' and '.join(emotions_detected)}. "
            response = emotion_context + response
        
        audio_base64 = generate_audio_base64(response)
        
        # Add to history
        history.append({
            'user_input': text,
            'response': response,
            'voice_emotion': voice_emotion,
            'facial_emotion': facial_emotion,
            'timestamp': datetime.now().isoformat()
        })
        
        session['chat_history'] = history[-3:]
        
        return jsonify({
            'transcription': text,
            'reply': response,
            'audio': audio_base64,
            'voice_emotion': voice_emotion,
            'facial_emotion': facial_emotion
        })
        
    except Exception as e:
        print(f"Face+Voice processing error: {e}")
        return jsonify({'error': 'Processing failed'}), 500

@app.route('/process', methods=['POST'])
def process_unified():
    """Unified processing endpoint that handles optional audio and image files along with text"""
    try:
        # Get text message (allow empty if audio is provided)
        message = request.form.get('message', '').strip()
        audio_file = request.files.get('audio')
        
        if not message and not audio_file:
            return jsonify({'error': 'No message or audio provided'}), 400
        
        # Get image file
        image_file = request.files.get('image')
        
        # Initialize variables with defaults
        voice_emotion = 'neutral'
        facial_emotion = 'neutral'
        transcribed_text = None
        
        # Process audio if provided
        if audio_file:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
                    audio_file.save(temp_audio.name)
                    
                    # Transcribe audio - simplified version
                    try:
                        with sr.AudioFile(temp_audio.name) as source:
                            audio_data = recognizer.record(source)
                            # Simplified transcription - in real implementation would use Google API
                            if message.lower().strip() in ['', 'audio only', 'voice only']:
                                transcribed_text = "I heard your voice message"  # Placeholder
                            else:
                                transcribed_text = None  # Keep original text if provided
                    except:
                        transcribed_text = "[Voice not transcribed]"
                    
                    # Detect voice emotion
                    voice_emotion = detect_voice_emotion(temp_audio.name)
                    
                    # Clean up
                    os.unlink(temp_audio.name)
            except Exception as e:
                print(f"Audio processing error: {e}")
                voice_emotion = 'neutral'
        
        # Process image if provided
        if image_file:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_image:
                    image_file.save(temp_image.name)
                    facial_emotion = detect_facial_emotion(temp_image.name)
                    os.unlink(temp_image.name)
            except Exception as e:
                print(f"Image processing error: {e}")
                facial_emotion = 'neutral'
        
        # Determine the final user input text
        final_text = transcribed_text if transcribed_text and transcribed_text != "[Voice not transcribed]" else message
        
        # Get chat history
        history = session.get('chat_history', [])
        
        # Generate response using the pipeline
        if transcribed_text == "[Voice not transcribed]" and not message:
            response = "I couldn't understand your voice clearly, but I'm here to listen. Can you try typing your message?"
            sentiment = 'neutral'
            emotion = 'neutral'
        else:
            response, sentiment, emotion, _, _ = generate_response_pipeline(final_text, history, voice_emotion, facial_emotion)
        
        # Add emotional context to response
        emotions_detected = []
        if voice_emotion and voice_emotion != 'neutral':
            emotions_detected.append(f"voice sounds {voice_emotion}")
        if facial_emotion and facial_emotion != 'neutral':
            emotions_detected.append(f"expression looks {facial_emotion}")
        
        if emotions_detected:
            emotion_context = f"I can sense your {' and '.join(emotions_detected)}. "
            response = emotion_context + response
        
        # Generate audio response
        audio_base64 = generate_audio_base64(response)
        
        # Add to history
        history_entry = {
            'user_input': final_text,
            'response': response,
            'sentiment': sentiment,
            'emotion': emotion,
            'timestamp': datetime.now().isoformat()
        }
        
        if voice_emotion:
            history_entry['voice_emotion'] = voice_emotion
        if facial_emotion:
            history_entry['facial_emotion'] = facial_emotion
            
        history.append(history_entry)
        
        # Keep only last 3 conversations
        session['chat_history'] = history[-3:]
        
        # Prepare response data
        response_data = {
            'reply': response,
            'audio': audio_base64,
            'emotion': emotion,
            'sentiment': sentiment
        }
        
        # Add optional data if available
        if transcribed_text:
            response_data['transcription'] = transcribed_text
        if voice_emotion:
            response_data['voice_emotion'] = voice_emotion
        if facial_emotion:
            response_data['facial_emotion'] = facial_emotion
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Unified processing error: {e}")
        return jsonify({'error': 'Processing failed'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)