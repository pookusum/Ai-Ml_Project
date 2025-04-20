import streamlit as st
import os
import time
import threading
import re
import pygame
from gtts import gTTS
import speech_recognition as sr
from model import Model, load_or_train_model

# Initialize pygame for audio playback
pygame.mixer.init()

class TextToSpeech:
    """
    Simple text-to-speech class using gTTS
    """
    def __init__(self):
        # Make sure pygame mixer is initialized
        if not pygame.mixer.get_init():
            pygame.mixer.init()
    
    def detect_language(self, text):
        """
        Detect if text is primarily English or Hindi
        """
        # Hindi Unicode range (approximate)
        hindi_pattern = re.compile(r'[\u0900-\u097F]')
        hindi_chars = len(hindi_pattern.findall(text))
        
        # If more than 20% of characters are Hindi, use Hindi TTS
        if hindi_chars > len(text) * 0.2:
            return "hi"
        return "en"
    
    def speak(self, text):
        """
        Convert text to speech and play it
        """
        if not text:
            return
            
        try:
            # Detect language
            lang = self.detect_language(text)
            
            # Create temporary file for audio
            temp_file = "temp_speech.mp3"
            
            # Generate speech
            tts = gTTS(text=text, lang=lang, slow=False)
            tts.save(temp_file)
            
            # Play the speech
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            
            # Wait for the audio to finish playing
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
                
            # Clean up
            pygame.mixer.music.unload()
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
        except Exception as e:
            st.error(f"Error in text-to-speech: {e}")


class SpeechToText:
    """
    Simple speech recognition class using SpeechRecognition
    """
    def __init__(self):
        self.recognizer = sr.Recognizer()
        # Adjust for ambient noise and microphone sensitivity
        self.recognizer.energy_threshold = 4000
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        
    def listen(self):
        """
        Listen to the microphone and convert speech to text
        """
        with sr.Microphone() as source:
            st.session_state.listening = True
            try:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                # Listen for speech
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                st.session_state.listening = False
                
                # Try to recognize in English
                try:
                    text = self.recognizer.recognize_google(audio, language="en-IN")
                    return text
                except:
                    # If English recognition fails, try Hindi
                    try:
                        text = self.recognizer.recognize_google(audio, language="hi-IN")
                        return text
                    except sr.UnknownValueError:
                        return None
                    except sr.RequestError as e:
                        st.error(f"Could not request results; {e}")
                        return None
            except Exception as e:
                st.error(f"Error during listening: {e}")
                st.session_state.listening = False
                return None


class VoiceAssistant:
    """
    Main voice assistant class that integrates speech recognition,
    question answering, and text-to-speech components.
    """
    def __init__(self, model_path, csv_path):
        self.speech_to_text = SpeechToText()
        self.text_to_speech = TextToSpeech()
        self.model = self.load_or_train_model(model_path, csv_path)
        
    def load_or_train_model(self, model_path, csv_path):
        """Load existing model or train a new one if not available"""
        model = load_or_train_model(csv_path, model_path)
        if model is None:
            # Create a minimal model if loading/training fails
            model = Model()
            model.questions = ["Hello"]
            model.answers = {"Hello": "Hi there! I'm having trouble with my training data."}
            st.error("Failed to load or train model. Using minimal functionality.")
        return model
        
    def process_voice_query(self):
        """
        Process a single voice query:
        1. Listen for speech
        2. Convert speech to text
        3. Generate an answer
        4. Convert answer to speech
        """
        # Listen for speech and convert to text
        query = self.speech_to_text.listen()
        
        if query:
            # Generate answer
            answer = self.model.answer(query)
            
            # Convert answer to speech if speech is enabled
            if st.session_state.get('speech_enabled', True):
                threading.Thread(target=self.text_to_speech.speak, args=(answer,)).start()
            
            return query, answer
        else:
            error_message = "I didn't hear anything. Could you please try again?"
            if st.session_state.get('speech_enabled', True):
                self.text_to_speech.speak(error_message)
            return None, error_message
        
    def process_text_query(self, query):
        """Process a text query without speech recognition"""
        if query:
            # Generate answer
            answer = self.model.answer(query)
            
            # Convert answer to speech if speech is enabled
            if st.session_state.get('speech_enabled', True):
                threading.Thread(target=self.text_to_speech.speak, args=(answer,)).start()
            
            return query, answer
        return None, None
        
    def train_with_new_data(self, question, answer):
        """Train the model with a new question-answer pair"""
        if question and answer:
            self.model.train([question], [answer], epochs=20)
            self.model.save_model("voice_assistant_model.pkl")
            return True
        return False


def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "assistant" not in st.session_state:
        # Initialize with paths to model and training data
        st.session_state.assistant = VoiceAssistant(
            model_path="voice_assistant_model.pkl",
            csv_path="training_data.csv"
        )
    
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False
        
    if "speech_enabled" not in st.session_state:
        st.session_state.speech_enabled = True
        
    if "listening" not in st.session_state:
        st.session_state.listening = False


def display_conversation():
    """Display the conversation history in a chat-like interface"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "debug" in message and st.session_state.debug_mode:
                st.caption(message["debug"])


def main():
    # Set page config
    st.set_page_config(
        page_title="Voice AI Assistant",
        page_icon="🎤",
        layout="wide"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("Voice AI Assistant")
    st.subheader("Ask me anything! ")
    
    # Sidebar for settings and training
    with st.sidebar:
        st.header("Settings")
        
        # Speech toggle
        speech_enabled = st.toggle("Enable Speech Output", value=st.session_state.speech_enabled)
        if speech_enabled != st.session_state.speech_enabled:
            st.session_state.speech_enabled = speech_enabled
        
        # Debug mode toggle
        debug_mode = st.toggle("Debug Mode", value=st.session_state.debug_mode)
        if debug_mode != st.session_state.debug_mode:
            st.session_state.debug_mode = debug_mode
            
        # Training section
        st.header("Train the Assistant")
        with st.form("training_form"):
            st.write("Add new question-answer pairs")
            new_question = st.text_input("Question:")
            new_answer = st.text_area("Answer:")
            submitted = st.form_submit_button("Add to Training Data")
            
            if submitted:
                success = st.session_state.assistant.train_with_new_data(new_question, new_answer)
                if success:
                    st.success("New training data added successfully!")
                else:
                    st.error("Please provide both a question and an answer.")
    
    # Main content area - split into two columns
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Instructions
        with st.expander("How to use this assistant", expanded=False):
            st.markdown("""
            ### Voice Interaction
            1. Click the **Listen** button to activate the microphone
            2. Speak your question clearly (in English, Hindi, or a mix)
            3. Wait for the assistant to process and respond
            4. The assistant will speak the answer and display it in the chat
            
            ### Text Interaction
            1. Type your question in the text input at the bottom
            2. Press Enter or click the Send button
            3. The assistant will respond in the chat
            
            **Note:** Make sure your microphone is properly connected and browser permissions are granted for voice interaction.
            """)
        
        # Chat display
        st.subheader("Conversation")
        display_conversation()
        
        # Input area
        if st.session_state.listening:
            st.info("Listening... Please speak now.")
        
        # Text input
        user_input = st.chat_input("Type your message here...")
        if user_input:
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Get response
            _, response = st.session_state.assistant.process_text_query(user_input)
            
            # Get debug info if in debug mode
            debug_info = ""
            if st.session_state.debug_mode:
                closest_question, similarity = st.session_state.assistant.model.find_closest_question(user_input)
                debug_info = f"Matched with: '{closest_question}' (similarity: {similarity:.4f})"
            
            # Add assistant response to chat
            st.session_state.messages.append({"role": "assistant", "content": response, "debug": debug_info})
            
            # Rerun to update the UI
            st.experimental_rerun()
    
    with col2:
        # Voice interaction button
        st.subheader("Voice Interaction")
        listen_col, clear_col = st.columns(2)
        
        with listen_col:
            if st.button("🎙️ Listen", use_container_width=True):
                query, answer = st.session_state.assistant.process_voice_query()
                
                if query:
                    # Get debug info if in debug mode
                    debug_info = ""
                    if st.session_state.debug_mode:
                        closest_question, similarity = st.session_state.assistant.model.find_closest_question(query)
                        debug_info = f"Matched with: '{closest_question}' (similarity: {similarity:.4f})"
                    
                    # Add user message to chat history
                    st.session_state.messages.append({"role": "user", "content": query})
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": answer, "debug": debug_info})
                    
                    # Rerun to update the UI
                    st.experimental_rerun()
        
        with clear_col:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.experimental_rerun()
        
        # System status
        st.subheader("System Status")
        
        # Display model info
        model = st.session_state.assistant.model
        st.info(f"Model loaded with {len(model.questions)} Q&A pairs")
        
        # Error handling
        st.subheader("Troubleshooting")
        with st.expander("Common Issues"):
            st.markdown("""
            **If the assistant can't hear you:**
            - Check if your microphone is connected
            - Make sure you've granted browser permissions
            - Speak clearly and at a normal volume
            
            **If the assistant doesn't respond correctly:**
            - Try rephrasing your question
            - Add training data for specific questions
            - Check for background noise
            
            **If the model is not learning properly:**
            - Add more diverse training examples
            - Make sure questions and answers are clear
            - Try retraining with more epochs
            """)


if __name__ == "__main__":
    main()
