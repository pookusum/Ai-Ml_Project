import numpy as np
import re
import string
import pickle
import os
import csv
from collections import Counter
import random

class Model:
    """
    A simple language model for conversational interactions.
    Built from scratch without external NLP libraries.
    """
    def __init__(self):
        # Core model components
        self.questions = []
        self.answers = {}
        self.question_vectors = {}
        self.vocabulary = set()
        self.word_to_index = {}
        self.index_to_word = {}
        self.word_vectors = None
        self.vector_size = 50
        
        # Conversation management
        self.conversation_history = []
        self.max_history_length = 5
        
        # Response templates for different dialogue states
        self.templates = {
            "greeting": [
                "Hello! How can I help you today?",
                "Hi there! What can I do for you?",
                "Hey! How can I assist you?",
                "नमस्ते! आज मैं आपकी कैसे मदद कर सकता हूं?",
                "हैलो! मैं आपकी क्या सहायता कर सकता हूं?"
            ],
            "farewell": [
                "Goodbye! Have a great day!",
                "Bye! Feel free to call again if you need anything.",
                "See you later! Take care!",
                "अलविदा! आपका दिन शुभ हो!",
                "फिर मिलेंगे! अपना ख्याल रखना!"
            ],
            "thanks": [
                "You're welcome! Is there anything else I can help with?",
                "Happy to help! Do you need anything else?",
                "No problem at all! What else can I do for you?",
                "आपका स्वागत है! क्या मैं और कुछ मदद कर सकता हूं?",
                "खुशी हुई मदद करके! क्या आपको कुछ और चाहिए?"
            ],
            "help": [
                "I can answer questions, provide information, or just chat. What would you like to know?",
                "I'm here to help! You can ask me questions in English or Hindi.",
                "मैं आपके सवालों का जवाब दे सकता हूं, जानकारी प्रदान कर सकता हूं, या बस बात कर सकता हूं। आप क्या जानना चाहेंगे?",
                "मैं मदद के लिए यहां हूं! आप मुझसे अंग्रेजी या हिंदी में सवाल पूछ सकते हैं।"
            ],
            "fallback": [
                "I'm not sure I understand. Could you rephrase that?",
                "I didn't quite catch that. Can you say it differently?",
                "मुझे समझ नहीं आया। क्या आप इसे दोबारा कह सकते हैं?",
                "मुझे वह ठीक से समझ नहीं आया। क्या आप इसे अलग तरीके से कह सकते हैं?"
            ]
        }
        
        # Keywords for dialogue state detection
        self.keywords = {
            "greeting": ["hello", "hi", "hey", "नमस्ते", "हैलो", "हाय"],
            "farewell": ["goodbye", "bye", "see you", "अलविदा", "फिर मिलेंगे"],
            "thanks": ["thank", "thanks", "धन्यवाद", "शुक्रिया"],
            "help": ["help", "मदद", "सहायता"],
            "yes": ["yes", "yeah", "sure", "हां", "जी हां"],
            "no": ["no", "nope", "नहीं", "ना"]
        }
        
    def preprocess_text(self, text):
        """
        Simple text preprocessing: lowercase, remove punctuation, tokenize.
        """
        if not text:
            return []
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Simple tokenization (split by whitespace)
        words = text.split()
        
        return words
    
    def build_vocabulary(self, texts):
        """Build vocabulary from a list of texts"""
        all_words = []
        for text in texts:
            all_words.extend(self.preprocess_text(text))
        
        # Get unique words and sort by frequency
        word_counts = Counter(all_words)
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Create vocabulary
        self.vocabulary = set(word for word, _ in sorted_words)
        
        # Create word to index mapping
        self.word_to_index = {word: i+1 for i, (word, _) in enumerate(sorted_words)}
        self.word_to_index['<UNK>'] = 0  # Unknown token
        
        # Create index to word mapping
        self.index_to_word = {i: word for word, i in self.word_to_index.items()}
        
        # Initialize word vectors with random values
        vocab_size = len(self.word_to_index)
        self.word_vectors = np.random.randn(vocab_size, self.vector_size) * 0.1
        
        print(f"Vocabulary built with {len(self.vocabulary)} words")
        return self.vocabulary
    
    def text_to_vector(self, text):
        """
        Convert text to a simple bag-of-words vector representation
        """
        if not text or not self.word_vectors is not None:
            # Return zero vector if text is empty or word vectors not initialized
            return np.zeros(self.vector_size)
            
        words = self.preprocess_text(text)
        
        if not words:
            return np.zeros(self.vector_size)
        
        # Get word vectors for each word in the text
        vectors = []
        for word in words:
            # Get word index (use 0 for unknown words)
            idx = self.word_to_index.get(word, 0)
            vectors.append(self.word_vectors[idx])
        
        # Average the word vectors
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.vector_size)
    
    def cosine_similarity(self, vec1, vec2):
        """
        Calculate cosine similarity between two vectors
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
            
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def train(self, questions, answers, epochs=50):
        """
        Train the model on question-answer pairs
        """
        print(f"Training model with {len(questions)} question-answer pairs...")
        
        # Store questions and answers
        self.questions = questions
        for q, a in zip(questions, answers):
            self.answers[q] = a
        
        # Build vocabulary from all text
        all_text = questions + answers
        self.build_vocabulary(all_text)
        
        # Create vector representations for questions
        for question in questions:
            self.question_vectors[question] = self.text_to_vector(question)
        
        # Simple training loop to adjust vectors for better matching
        for epoch in range(epochs):
            correct = 0
            
            # For each question, find the closest match and adjust if wrong
            for i, question in enumerate(questions):
                closest, similarity = self.find_closest_question(question)
                
                if closest == question:
                    correct += 1
                else:
                    # Adjust vectors to make the correct match stronger
                    q_vec = self.question_vectors[question]
                    closest_vec = self.question_vectors[closest]
                    
                    # Move question vector away from incorrect match
                    adjustment = 0.1 * (q_vec - closest_vec)
                    self.question_vectors[question] = q_vec + adjustment
                    
                    # Normalize
                    norm = np.linalg.norm(self.question_vectors[question])
                    if norm > 0:
                        self.question_vectors[question] = self.question_vectors[question] / norm
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
                accuracy = correct / len(questions)
                print(f"Epoch {epoch+1}/{epochs}: Accuracy = {accuracy:.2%}")
                
            # If perfect accuracy, stop training
            if correct == len(questions):
                print(f"Reached perfect accuracy at epoch {epoch+1}. Stopping training.")
                break
        
        print("Training complete!")
        return True
    
    def find_closest_question(self, query):
        """
        Find the most similar question in the training data
        """
        if not self.questions:
            return None, 0
            
        query_vector = self.text_to_vector(query)
        
        best_match = None
        best_similarity = -1
        
        # Find the question with the highest cosine similarity
        for question in self.questions:
            q_vector = self.question_vectors[question]
            similarity = self.cosine_similarity(query_vector, q_vector)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = question
        
        # If the best match is too weak, try keyword matching
        if best_similarity < 0.5:
            query_words = set(self.preprocess_text(query))
            
            for question in self.questions:
                question_words = set(self.preprocess_text(question))
                
                # Check for exact matches or subsets
                if query.lower() in question.lower() or question.lower() in query.lower():
                    return question, 0.9
                
                # Check for word overlap
                overlap = len(query_words.intersection(question_words))
                if overlap > 0 and overlap == len(question_words):
                    return question, 0.8
        
        return best_match, best_similarity
    
    def detect_dialogue_state(self, text):
        """
        Detect the dialogue state based on keywords in the input text
        """
        if not text:
            return None, 0
            
        text_lower = text.lower()
        words = set(self.preprocess_text(text_lower))
        
        best_state = None
        best_score = 0
        
        # Check each dialogue state
        for state, keywords in self.keywords.items():
            # Count matching keywords
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            score = matches / len(keywords) if matches > 0 else 0
            
            if score > best_score:
                best_score = score
                best_state = state
        
        return best_state, best_score
    
    def answer(self, query):
        """
        Generate a response to the user query
        """
        if not query:
            return "I didn't hear anything. Could you please repeat?"
        
        # Detect dialogue state
        dialogue_state, confidence = self.detect_dialogue_state(query)
        
        # If we detected a dialogue state with high confidence, use a template response
        if dialogue_state and confidence >= 0.3:
            if dialogue_state in self.templates:
                return random.choice(self.templates[dialogue_state])
        
        # Otherwise, find the closest question and return its answer
        closest_question, similarity = self.find_closest_question(query)
        
        # Debug information
        print(f"Query: '{query}'")
        print(f"Closest match: '{closest_question}' with similarity {similarity:.4f}")
        
        # If we found a good match, return the answer
        if similarity >= 0.5 and closest_question in self.answers:
            return self.answers[closest_question]
        
        # If no good match, try keyword matching for common topics
        query_words = set(self.preprocess_text(query.lower()))
        
        # Check for "name" related questions
        name_keywords = {"name", "naam", "नाम"}
        if query_words.intersection(name_keywords):
            for question in self.questions:
                if "name" in question.lower() or "naam" in question.lower() or "नाम" in question.lower():
                    return self.answers[question]
        
        # Check for "python" related questions
        if "python" in query_words or "पायथन" in query_words:
            for question in self.questions:
                if "python" in question.lower() or "पायथन" in question.lower():
                    return self.answers[question]
        
        # If all else fails, use a fallback response
        return random.choice(self.templates["fallback"])
    
    def save_model(self, filepath):
        """Save the model to a file"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'questions': self.questions,
                'answers': self.answers,
                'question_vectors': self.question_vectors,
                'vocabulary': self.vocabulary,
                'word_to_index': self.word_to_index,
                'index_to_word': self.index_to_word,
                'word_vectors': self.word_vectors,
                'vector_size': self.vector_size
            }, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load the model from a file.
        Returns False if the file doesn't exist or is empty, True otherwise.
        """
        try:
            # Check if file exists and is not empty
            if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
                print(f"Model file {filepath} doesn't exist or is empty.")
                return False
                
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.questions = data['questions']
                self.answers = data['answers']
                self.question_vectors = data['question_vectors']
                self.vocabulary = data['vocabulary']
                self.word_to_index = data['word_to_index']
                self.index_to_word = data['index_to_word']
                self.word_vectors = data['word_vectors']
                self.vector_size = data['vector_size']
                
            print(f"Model loaded from {filepath}")
            print(f"Model contains {len(self.questions)} question-answer pairs")
            print(f"Vocabulary size: {len(self.vocabulary)}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False


def load_training_data(csv_file):
    """
    Load training data from a CSV file
    """
    questions = []
    answers = []
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                questions.append(row['question'])
                answers.append(row['answer'])
                
        print(f"Loaded {len(questions)} question-answer pairs from {csv_file}")
        return questions, answers
        
    except Exception as e:
        print(f"Error loading training data: {e}")
        return [], []


def train_model(csv_file, model_file):
    """
    Train a model using data from a CSV file and save it
    """
    # Load training data
    questions, answers = load_training_data(csv_file)
    
    if not questions:
        print("No training data available. Cannot train model.")
        return False
    
    # Create and train model
    model = Model()
    model.train(questions, answers, epochs=50)
    
    # Save model
    model.save_model(model_file)
    
    return True


def load_or_train_model(csv_file, model_file):
    """
    Load an existing model or train a new one if it doesn't exist
    """
    model = Model()
    
    # Try to load the model
    if model.load_model(model_file):
        return model
    
    # If loading fails, train a new model
    print("Training a new model...")
    questions, answers = load_training_data(csv_file)
    
    if questions:
        model.train(questions, answers, epochs=50)
        model.save_model(model_file)
        return model
    else:
        print("No training data available. Cannot train model.")
        return None


# If this file is run directly, train a model
if __name__ == "__main__":
    csv_file = "training_data.csv"
    model_file = "voice_assistant_model.pkl"
    
    print("Training voice assistant model...")
    success = train_model(csv_file, model_file)
    
    if success:
        print("Model training complete!")
        
        # Test the model
        model = Model()
        model.load_model(model_file)
        
        test_questions = [
            "What is your name?",
            "Tell me a joke",
            "What is Python?",
            "आपका नाम क्या है?",
            "Hello",
            "Thank you"
        ]
        
        print("\nTesting model with sample questions:")
        for question in test_questions:
            answer = model.answer(question)
            print(f"Q: {question}")
            print(f"A: {answer}")
            print()
    else:
        print("Model training failed.")