---
sidebar_position: 7
title: "Conversational Robotics Introduction"
---

# Conversational Robotics Introduction

This lesson introduces the fundamentals of conversational robotics, focusing on integrating natural language processing and speech capabilities into humanoid robots.

## Learning Objectives

After completing this lesson, you will be able to:
- Understand the components of conversational robotics
- Implement speech recognition and natural language understanding
- Design dialogue management systems
- Integrate conversational AI with robot actions
- Evaluate conversational system performance
- Plan for multi-modal conversational interactions

## Introduction to Conversational Robotics

Conversational robotics combines natural language processing, speech recognition, and dialogue management to enable robots to engage in natural conversations with humans. This capability is essential for humanoid robots to serve as social companions, assistants, and collaborators.

### What Makes Conversational Robotics Unique?

Unlike traditional chatbots, conversational robots must:
- **Ground language in the physical world**: Connect spoken language to objects and actions
- **Handle multimodal input**: Process speech, gestures, and visual information together
- **Maintain spatial awareness**: Understand spatial relationships in conversation
- **Coordinate with actions**: Execute robot behaviors based on conversational input
- **Manage attention**: Focus on appropriate conversation partners
- **Express personality**: Show consistent character traits through conversation

### Key Components of Conversational Robotics

1. **Automatic Speech Recognition (ASR)**: Converting speech to text
2. **Natural Language Understanding (NLU)**: Interpreting user intent
3. **Dialogue Management**: Managing conversation flow
4. **Natural Language Generation (NLG)**: Creating appropriate responses
5. **Text-to-Speech (TTS)**: Converting text to speech
6. **Multimodal Integration**: Coordinating with gestures and expressions

## Speech Recognition and Natural Language Understanding

### Automatic Speech Recognition (ASR)

ASR systems convert spoken language to text. For conversational robots, special considerations include:

#### Real-time Processing
- **Low latency**: Fast response to maintain natural conversation flow
- **Incremental recognition**: Providing partial results during speech
- **Robustness**: Handling background noise and varying acoustic conditions

#### Robot-Specific Challenges
- **Self-noise**: Robot's own fans, motors, and speakers creating noise
- **Acoustic environment**: Room acoustics affecting speech quality
- **Speaker adaptation**: Adjusting to different speakers' voices and accents

### Implementation Example

```python
import speech_recognition as sr
import threading
import queue
import time

class RobotSpeechRecognizer:
    def __init__(self, robot_self_noise_threshold=0.3):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.is_listening = False
        self.self_noise_threshold = robot_self_noise_threshold
        
        # Energy threshold for speech detection
        self.recognizer.energy_threshold = 4000  # Adjust based on environment
        self.recognizer.dynamic_energy_threshold = True
        
        # Set up microphone
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
    
    def start_listening(self):
        """Start continuous listening for speech"""
        self.is_listening = True
        self.listen_thread = threading.Thread(target=self._continuous_listen)
        self.listen_thread.daemon = True
        self.listen_thread.start()
    
    def _continuous_listen(self):
        """Continuously listen for speech"""
        with self.microphone as source:
            while self.is_listening:
                try:
                    # Listen for audio with timeout
                    audio = self.recognizer.listen(
                        source, 
                        timeout=0.5,  # Short timeout to check if still listening
                        phrase_time_limit=5.0  # Max phrase length
                    )
                    
                    # Add to processing queue
                    self.audio_queue.put(audio)
                    
                except sr.WaitTimeoutError:
                    # Continue listening
                    continue
                except Exception as e:
                    print(f"Listening error: {e}")
                    continue
    
    def process_audio_async(self):
        """Process audio asynchronously"""
        try:
            audio = self.audio_queue.get_nowait()
            
            # Process audio in separate thread to avoid blocking
            process_thread = threading.Thread(
                target=self._process_audio, 
                args=(audio,)
            )
            process_thread.daemon = True
            process_thread.start()
            
        except queue.Empty:
            pass  # No audio to process
    
    def _process_audio(self, audio):
        """Process audio and perform recognition"""
        try:
            # Recognize speech using Google's service
            text = self.recognizer.recognize_google(audio)
            
            # Add to results queue
            self.result_queue.put({
                'text': text,
                'timestamp': time.time(),
                'confidence': 1.0  # Google API doesn't provide confidence
            })
            
            print(f"Recognized: {text}")
            
        except sr.UnknownValueError:
            print("Could not understand audio")
            self.result_queue.put({
                'text': '',
                'timestamp': time.time(),
                'error': 'unrecognized'
            })
        except sr.RequestError as e:
            print(f"Recognition service error: {e}")
            self.result_queue.put({
                'text': '',
                'timestamp': time.time(),
                'error': f'request_error: {str(e)}'
            })
    
    def get_recognition_result(self, timeout=0.1):
        """Get recognition result with optional timeout"""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def stop_listening(self):
        """Stop the listening process"""
        self.is_listening = False
        if hasattr(self, 'listen_thread'):
            self.listen_thread.join(timeout=1.0)
```

### Natural Language Understanding (NLU)

NLU interprets the meaning of recognized text, extracting intents and entities.

#### Intent Recognition
Intent recognition determines what the user wants to do:

```python
class IntentClassifier:
    def __init__(self):
        # Define possible intents
        self.intents = {
            'greeting': {
                'keywords': ['hello', 'hi', 'hey', 'good morning', 'good evening'],
                'responses': ['Hello!', 'Hi there!', 'Good to see you!']
            },
            'navigation': {
                'keywords': ['go to', 'move to', 'walk to', 'navigate to', 'go'],
                'responses': ['I can help with navigation.', 'Where would you like me to go?']
            },
            'manipulation': {
                'keywords': ['pick up', 'grasp', 'take', 'get', 'lift', 'hold'],
                'responses': ['I can help with that.', 'What would you like me to grasp?']
            },
            'information_request': {
                'keywords': ['what', 'how', 'where', 'when', 'who', 'why'],
                'responses': ['I can provide information.', 'What would you like to know?']
            },
            'farewell': {
                'keywords': ['bye', 'goodbye', 'see you', 'farewell', 'later'],
                'responses': ['Goodbye!', 'See you later!', 'Take care!']
            },
            'unknown': {
                'keywords': [],
                'responses': ['I\'m not sure I understand.', 'Could you rephrase that?']
            }
        }
    
    def classify_intent(self, text):
        """Classify intent based on keywords and patterns"""
        text_lower = text.lower()
        
        # Check each intent for keyword matches
        for intent_name, intent_data in self.intents.items():
            if intent_name == 'unknown':
                continue  # Skip unknown for now
            
            # Count keyword matches
            matches = sum(1 for keyword in intent_data['keywords'] 
                         if keyword in text_lower)
            
            if matches > 0:
                return {
                    'intent': intent_name,
                    'confidence': min(1.0, matches / len(intent_data['keywords']) * 2),
                    'matched_keywords': [kw for kw in intent_data['keywords'] 
                                       if kw in text_lower]
                }
        
        # If no specific intent found, return unknown
        return {
            'intent': 'unknown',
            'confidence': 1.0,
            'matched_keywords': []
        }
    
    def get_response(self, intent_name):
        """Get appropriate response for intent"""
        import random
        intent_data = self.intents.get(intent_name, self.intents['unknown'])
        return random.choice(intent_data['responses'])
```

#### Entity Extraction
Entity extraction identifies specific objects, locations, or values in user utterances:

```python
import re

class EntityExtractor:
    def __init__(self):
        # Define entity patterns
        self.patterns = {
            'object': [
                r'\b(?:the\s+)?(\w+)\b',  # Simple object names
                r'\b(red|blue|green|yellow|large|small)\s+(\w+)\b',  # Colored/size objects
            ],
            'location': [
                r'\b(?:to|at|in|on)\s+(?:the\s+)?(\w+(?:\s+\w+)*)\b',  # Locations after prepositions
                r'\b(kitchen|living room|bedroom|office|bathroom|hallway|door|window)\b',  # Known places
            ],
            'number': [
                r'\b(\d+)\b',  # Numbers
                r'\b(one|two|three|four|five|six|seven|eight|nine|ten)\b',  # Word numbers
            ],
            'person': [
                r'\b(me|you|him|her|them|someone|anyone)\b',  # Pronouns
                r'\b(\w+\s+\w+|\w+)\b',  # Potential names
            ]
        }
        
        # Known objects and locations in the environment
        self.known_objects = {
            'red cup', 'blue bottle', 'green plant', 'wooden table', 'metal chair'
        }
        self.known_locations = {
            'kitchen', 'living room', 'bedroom', 'office', 'entrance'
        }
    
    def extract_entities(self, text):
        """Extract entities from text"""
        entities = {}
        
        text_lower = text.lower()
        
        for entity_type, patterns in self.patterns.items():
            found_entities = []
            
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                for match in matches:
                    if isinstance(match, tuple):
                        match = ' '.join(match).strip()
                    else:
                        match = match.strip()
                    
                    if match and self.is_valid_entity(entity_type, match):
                        found_entities.append({
                            'text': match,
                            'type': entity_type,
                            'confidence': 0.8  # Simple confidence
                        })
            
            if found_entities:
                entities[entity_type] = found_entities
        
        return entities
    
    def is_valid_entity(self, entity_type, text):
        """Check if extracted text is a valid entity"""
        if entity_type == 'object':
            return text in self.known_objects or len(text.split()) <= 3
        elif entity_type == 'location':
            return text in self.known_locations or len(text.split()) <= 2
        elif entity_type == 'number':
            try:
                float(text) if text.replace('.', '').isdigit() else self.word_to_number(text)
                return True
            except:
                return False
        else:
            return True
    
    def word_to_number(self, word):
        """Convert number words to digits"""
        word_map = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
        }
        return word_map.get(word.lower(), 0)
```

## Dialogue Management

Dialogue management coordinates the flow of conversation, maintaining context and managing state.

### Dialogue State Tracking

```python
class DialogueStateTracker:
    def __init__(self):
        self.current_topic = None
        self.context_stack = []
        self.user_goals = []
        self.robot_goals = []
        self.conversation_history = []
        self.entity_memory = {}
        self.turn_count = 0
    
    def update_state(self, user_input, recognized_intent, entities):
        """Update dialogue state with user input"""
        self.turn_count += 1
        
        # Store user input
        user_turn = {
            'turn_id': self.turn_count,
            'speaker': 'user',
            'text': user_input,
            'intent': recognized_intent,
            'entities': entities,
            'timestamp': time.time()
        }
        
        self.conversation_history.append(user_turn)
        
        # Update context based on intent
        self.update_context(recognized_intent, entities)
        
        # Update entity memory
        self.update_entity_memory(entities)
        
        # Update goals
        self.update_goals(recognized_intent, entities)
    
    def update_context(self, intent, entities):
        """Update conversation context based on user input"""
        # Set current topic based on intent
        topic_map = {
            'navigation': 'navigation_task',
            'manipulation': 'manipulation_task',
            'information_request': 'information_query',
            'greeting': 'social_interaction',
            'farewell': 'conversation_end'
        }
        
        if intent in topic_map:
            self.current_topic = topic_map[intent]
        
        # Add to context stack if relevant
        if self.current_topic and (not self.context_stack or self.context_stack[-1] != self.current_topic):
            self.context_stack.append(self.current_topic)
    
    def update_entity_memory(self, entities):
        """Update memory of entities mentioned"""
        for entity_type, entity_list in entities.items():
            if entity_type not in self.entity_memory:
                self.entity_memory[entity_type] = []
            
            for entity in entity_list:
                # Add entity if not already remembered
                if not any(e['text'] == entity['text'] for e in self.entity_memory[entity_type]):
                    self.entity_memory[entity_type].append(entity)
    
    def update_goals(self, intent, entities):
        """Update user and robot goals based on interaction"""
        if intent == 'navigation':
            # User wants robot to navigate somewhere
            if 'location' in entities:
                location = entities['location'][0]['text']
                self.user_goals.append({
                    'type': 'navigation',
                    'target': location,
                    'status': 'pending'
                })
        
        elif intent == 'manipulation':
            # User wants robot to manipulate something
            if 'object' in entities:
                obj = entities['object'][0]['text']
                self.user_goals.append({
                    'type': 'manipulation',
                    'target': obj,
                    'status': 'pending'
                })
    
    def get_context(self):
        """Get current conversation context"""
        return {
            'current_topic': self.current_topic,
            'recent_entities': self.entity_memory,
            'user_goals': self.user_goals,
            'turn_count': self.turn_count,
            'context_stack': self.context_stack[:]
        }
    
    def reset_context(self):
        """Reset dialogue context"""
        self.current_topic = None
        self.context_stack = []
        self.user_goals = []
        self.conversation_history = []
        self.entity_memory = {}
        self.turn_count = 0
```

### Dialogue Policy

The dialogue policy determines the robot's response based on the current state:

```python
class DialoguePolicy:
    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.state_tracker = DialogueStateTracker()
    
    def generate_response(self, user_input):
        """Generate appropriate response to user input"""
        # Classify intent
        intent_result = self.intent_classifier.classify_intent(user_input)
        
        # Extract entities
        entities = self.entity_extractor.extract_entities(user_input)
        
        # Update dialogue state
        self.state_tracker.update_state(user_input, intent_result['intent'], entities)
        
        # Generate response based on state
        response = self.select_response(intent_result, entities)
        
        return {
            'response_text': response,
            'intent': intent_result['intent'],
            'entities': entities,
            'context': self.state_tracker.get_context()
        }
    
    def select_response(self, intent_result, entities):
        """Select appropriate response based on intent and context"""
        intent = intent_result['intent']
        context = self.state_tracker.get_context()
        
        # Check for specific response conditions
        if intent == 'navigation' and 'location' in entities:
            location = entities['location'][0]['text']
            return f"I can help you navigate to {location}. Is that correct?"
        
        elif intent == 'manipulation' and 'object' in entities:
            obj = entities['object'][0]['text']
            return f"I can help you with {obj}. What would you like me to do with it?"
        
        elif intent == 'information_request':
            return self.generate_information_response(entities)
        
        else:
            # Use intent-based response
            return self.intent_classifier.get_response(intent)
    
    def generate_information_response(self, entities):
        """Generate response for information requests"""
        if 'object' in entities:
            obj = entities['object'][0]['text']
            return f"Regarding {obj}, I can tell you about its properties or help you interact with it."
        elif 'location' in entities:
            location = entities['location'][0]['text']
            return f"About {location}, I can guide you there or tell you what's there."
        else:
            return "I can provide information about objects, locations, or robot capabilities."
```

## Conversational AI Integration

### Using Large Language Models

Integrating large language models (LLMs) can enhance conversational capabilities:

```python
import openai
import json

class LLMConversationalAgent:
    def __init__(self, api_key=None, model="gpt-3.5-turbo"):
        if api_key:
            openai.api_key = api_key
        self.model = model
        
        # Conversation history for context
        self.conversation_history = []
    
    def generate_response(self, user_input, robot_capabilities=None):
        """Generate response using LLM with robot context"""
        # Prepare system message with robot context
        system_message = {
            "role": "system",
            "content": f"""You are a helpful humanoid robot with the following capabilities: 
            {robot_capabilities or 'Basic navigation, manipulation, and conversation'}
            
            Respond naturally and helpfully to the user's requests. If asked to perform actions, 
            acknowledge the request and indicate you can help. Keep responses concise but informative.
            
            Always maintain a helpful and friendly tone appropriate for a social robot."""
        }
        
        # Add user message
        user_message = {
            "role": "user",
            "content": user_input
        }
        
        # Prepare messages for API
        messages = [system_message] + self.conversation_history[-10:] + [user_message]  # Last 10 exchanges + current
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=150
            )
            
            # Extract response
            bot_response = response.choices[0].message.content
            
            # Update conversation history
            self.conversation_history.append(user_message)
            self.conversation_history.append({"role": "assistant", "content": bot_response})
            
            # Keep history to reasonable length
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            return bot_response
            
        except Exception as e:
            print(f"LLM API error: {e}")
            return "I'm having trouble responding right now. Could you try rephrasing?"
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []
    
    def get_conversation_summary(self):
        """Get a summary of the current conversation"""
        if not self.conversation_history:
            return "No conversation history."
        
        # Extract key points from recent exchanges
        recent_exchanges = self.conversation_history[-6:]  # Last 3 exchanges (user+bot)
        
        summary = "Recent conversation:\n"
        for msg in recent_exchanges:
            speaker = "User" if msg["role"] == "user" else "Robot"
            summary += f"{speaker}: {msg['content'][:50]}...\n"
        
        return summary
```

### Robot Action Integration

Connecting conversational understanding to robot actions:

```python
class RobotActionIntegrator:
    def __init__(self, robot_controller):
        self.robot_controller = robot_controller
        self.action_map = {
            'navigation': self.execute_navigation,
            'manipulation': self.execute_manipulation,
            'greeting': self.execute_greeting,
            'expression': self.execute_expression
        }
    
    def execute_intent(self, intent, entities, context):
        """Execute robot action based on intent and entities"""
        if intent in self.action_map:
            action_func = self.action_map[intent]
            return action_func(entities, context)
        else:
            print(f"No action defined for intent: {intent}")
            return False
    
    def execute_navigation(self, entities, context):
        """Execute navigation action"""
        if 'location' in entities:
            target_location = entities['location'][0]['text']
            
            print(f"Navigating to {target_location}")
            
            # In a real robot, this would call navigation system
            success = self.robot_controller.navigate_to(target_location)
            
            if success:
                print(f"Successfully navigated to {target_location}")
                return True
            else:
                print(f"Failed to navigate to {target_location}")
                return False
        else:
            print("No location specified for navigation")
            return False
    
    def execute_manipulation(self, entities, context):
        """Execute manipulation action"""
        if 'object' in entities:
            target_object = entities['object'][0]['text']
            
            print(f"Attempting to manipulate {target_object}")
            
            # In a real robot, this would call manipulation system
            success = self.robot_controller.manipulate_object(target_object)
            
            if success:
                print(f"Successfully manipulated {target_object}")
                return True
            else:
                print(f"Failed to manipulate {target_object}")
                return False
        else:
            print("No object specified for manipulation")
            return False
    
    def execute_greeting(self, entities, context):
        """Execute greeting action"""
        print("Executing greeting behavior")
        
        # Perform greeting actions
        success1 = self.robot_controller.wave()
        success2 = self.robot_controller.nod()
        
        return success1 and success2
    
    def execute_expression(self, entities, context):
        """Execute facial expression"""
        if 'emotion' in entities:
            emotion = entities['emotion'][0]['text']
            print(f"Showing {emotion} expression")
            
            # In a real robot, this would control facial expressions
            success = self.robot_controller.show_expression(emotion)
            return success
        else:
            print("No emotion specified for expression")
            return False
```

## Multi-Modal Conversational Systems

### Integrating Speech, Vision, and Gestures

```python
class MultiModalConversationalSystem:
    def __init__(self, robot_controller):
        # Initialize components
        self.speech_recognizer = RobotSpeechRecognizer()
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.dialogue_policy = DialoguePolicy()
        self.llm_agent = LLMConversationalAgent()
        self.action_integrator = RobotActionIntegrator(robot_controller)
        
        # Text-to-speech
        self.tts = TextToSpeech()  # From previous lesson
        
        # State management
        self.is_active = False
        self.conversation_context = {}
        
        # Multi-modal queues
        self.vision_queue = queue.Queue()
        self.gesture_queue = queue.Queue()
    
    def start_conversation_system(self):
        """Start the conversational system"""
        self.is_active = True
        self.speech_recognizer.start_listening()
        
        print("Conversational system started. Listening for input...")
        
        # Main processing loop
        self.process_conversations()
    
    def process_conversations(self):
        """Main loop for processing multi-modal input"""
        import time
        
        while self.is_active:
            # Process speech input
            speech_result = self.speech_recognizer.get_recognition_result(timeout=0.01)
            if speech_result and speech_result.get('text'):
                self.handle_speech_input(speech_result['text'])
            
            # Process other modalities (vision, gestures)
            self.process_vision_input()
            self.process_gesture_input()
            
            # Small delay to prevent overwhelming CPU
            time.sleep(0.05)
    
    def handle_speech_input(self, text):
        """Handle speech input and generate response"""
        print(f"Processing speech input: {text}")
        
        # Generate response using dialogue policy
        response_data = self.dialogue_policy.generate_response(text)
        
        response_text = response_data['response_text']
        intent = response_data['intent']
        entities = response_data['entities']
        context = response_data['context']
        
        # If we have specific robot capabilities to execute
        if intent in ['navigation', 'manipulation', 'greeting', 'expression']:
            success = self.action_integrator.execute_intent(intent, entities, context)
            if success:
                response_text += " I've done that for you."
            else:
                response_text += " I'm having trouble executing that right now."
        
        # Speak the response
        self.tts.speak(response_text)
        
        # Log interaction
        self.log_interaction(text, response_text, intent, entities)
    
    def process_vision_input(self):
        """Process visual input (objects, people, etc.)"""
        try:
            vision_data = self.vision_queue.get_nowait()
            
            # Handle visual information
            self.handle_visual_input(vision_data)
            
        except queue.Empty:
            pass  # No visual input to process
    
    def process_gesture_input(self):
        """Process gesture input"""
        try:
            gesture_data = self.gesture_queue.get_nowait()
            
            # Handle gesture information
            self.handle_gesture_input(gesture_data)
            
        except queue.Empty:
            pass  # No gesture input to process
    
    def handle_visual_input(self, vision_data):
        """Handle input from vision system"""
        # This could include:
        # - Object recognition
        # - Person detection
        # - Scene understanding
        # - Attention management
        
        if 'person_detected' in vision_data:
            person_id = vision_data['person_detected']
            print(f"Person {person_id} detected. Updating attention.")
            # Update attention system
            # self.attention_system.focus_on_person(person_id)
        
        elif 'object_detected' in vision_data:
            obj_info = vision_data['object_detected']
            print(f"Object {obj_info['name']} detected at {obj_info['position']}")
            # Update entity memory
            # self.dialogue_policy.state_tracker.update_entity_memory({
            #     'object': [{'text': obj_info['name'], 'type': 'object', 'confidence': 0.9}]
            # })
    
    def handle_gesture_input(self, gesture_data):
        """Handle input from gesture recognition"""
        gesture_type = gesture_data['gesture_type']
        person_id = gesture_data.get('person_id')
        
        print(f"Gesture '{gesture_type}' detected from person {person_id}")
        
        # Respond to gestures appropriately
        if gesture_type == 'wave':
            self.respond_to_gesture('wave', person_id)
        elif gesture_type == 'point':
            self.respond_to_gesture('point', person_id)
        # Add other gesture responses
    
    def respond_to_gesture(self, gesture_type, person_id):
        """Respond to specific gestures"""
        if gesture_type == 'wave':
            response = "Hello! I see you waving. How can I help you?"
            self.tts.speak(response)
            
            # Perform acknowledging gesture
            # self.robot_controller.wave_back()
        
        elif gesture_type == 'point':
            response = "I see you're pointing. Are you directing my attention to something?"
            self.tts.speak(response)
            
            # Look in the direction pointed
            # self.robot_controller.look_at_direction(gesture_data['direction'])
    
    def log_interaction(self, user_input, response, intent, entities):
        """Log interaction for analysis and improvement"""
        log_entry = {
            'timestamp': time.time(),
            'user_input': user_input,
            'response': response,
            'intent': intent,
            'entities': entities,
            'session_id': getattr(self, 'session_id', 'unknown')
        }
        
        # In a real system, this would be saved to a database
        print(f"Interaction logged: {user_input[:30]}... -> {response[:30]}...")
    
    def stop_conversation_system(self):
        """Stop the conversational system"""
        self.is_active = False
        self.speech_recognizer.stop_listening()
        print("Conversational system stopped.")
```

## Evaluation and Improvement

### Conversational System Metrics

```python
class ConversationalEvaluator:
    def __init__(self):
        self.metrics = {
            'understanding_accuracy': 0.0,
            'response_appropriateness': 0.0,
            'task_success_rate': 0.0,
            'user_satisfaction': 0.0,
            'conversation_flow': 0.0
        }
        self.interaction_logs = []
    
    def evaluate_interaction(self, interaction_data):
        """Evaluate a single interaction"""
        evaluation = {}
        
        # Understanding accuracy
        if 'expected_intent' in interaction_data and 'recognized_intent' in interaction_data:
            evaluation['understanding_accuracy'] = (
                1.0 if interaction_data['expected_intent'] == interaction_data['recognized_intent'] else 0.0
            )
        
        # Response appropriateness
        if 'user_reaction' in interaction_data:
            # Positive reactions indicate appropriate responses
            positive_indicators = ['laugh', 'nod', 'smile', 'thank', 'yes', 'good']
            user_text = interaction_data['user_reaction'].lower()
            evaluation['response_appropriateness'] = (
                1.0 if any(indicator in user_text for indicator in positive_indicators) else 0.0
            )
        
        # Task success (if applicable)
        if 'task_attempted' in interaction_data and 'task_successful' in interaction_data:
            evaluation['task_success_rate'] = 1.0 if interaction_data['task_successful'] else 0.0
        
        return evaluation
    
    def calculate_overall_metrics(self):
        """Calculate overall system metrics from all interactions"""
        if not self.interaction_logs:
            return self.metrics
        
        # Calculate averages
        for metric in self.metrics.keys():
            values = [log.get('evaluation', {}).get(metric, 0.0) 
                     for log in self.interaction_logs 
                     if 'evaluation' in log]
            
            if values:
                self.metrics[metric] = sum(values) / len(values)
        
        return self.metrics
    
    def get_improvement_suggestions(self):
        """Get suggestions for improving the conversational system"""
        suggestions = []
        metrics = self.calculate_overall_metrics()
        
        if metrics['understanding_accuracy'] < 0.7:
            suggestions.append("Improve intent classification accuracy")
        
        if metrics['response_appropriateness'] < 0.7:
            suggestions.append("Enhance response appropriateness for context")
        
        if metrics['task_success_rate'] < 0.8:
            suggestions.append("Improve task execution success rate")
        
        if metrics['user_satisfaction'] < 0.7:
            suggestions.append("Implement user satisfaction feedback mechanism")
        
        if metrics['conversation_flow'] < 0.7:
            suggestions.append("Improve dialogue state management")
        
        return suggestions
```

## Practical Exercise: Building a Conversational Robot

Create a complete conversational robot system:

1. **Implement speech recognition and processing**
2. **Create intent classification and entity extraction**
3. **Design dialogue management system**
4. **Integrate with robot actions**
5. **Test with various conversational scenarios**

### Complete Conversational Robot Example

```python
class CompleteConversationalRobot:
    def __init__(self):
        # Initialize all components
        self.robot_controller = MockRobotController()  # Mock for demonstration
        self.conversational_system = MultiModalConversationalSystem(self.robot_controller)
        self.evaluator = ConversationalEvaluator()
        
        print("Complete Conversational Robot initialized")
    
    def start_demo(self):
        """Start a demonstration of conversational capabilities"""
        print("Starting Conversational Robot Demo")
        print("=" * 50)
        
        print("\nDemo scenarios:")
        print("1. Basic conversation")
        print("2. Navigation request")
        print("3. Manipulation request")
        print("4. Mixed interaction")
        print("5. Exit")
        
        while True:
            try:
                choice = input("\nEnter choice (1-5): ").strip()
                
                if choice == '1':
                    self.demo_basic_conversation()
                elif choice == '2':
                    self.demo_navigation()
                elif choice == '3':
                    self.demo_manipulation()
                elif choice == '4':
                    self.demo_mixed_interaction()
                elif choice == '5':
                    print("Demo ended.")
                    break
                else:
                    print("Invalid choice. Please enter 1-5.")
                    
            except KeyboardInterrupt:
                print("\nDemo interrupted by user.")
                break
    
    def demo_basic_conversation(self):
        """Demonstrate basic conversation capabilities"""
        print("\n--- Basic Conversation Demo ---")
        print("Simulating: User says 'Hello, how are you?'")
        
        # Simulate the conversation flow
        response_data = self.conversational_system.dialogue_policy.generate_response(
            "Hello, how are you?"
        )
        
        print(f"Robot responds: {response_data['response_text']}")
        print("Intent recognized:", response_data['intent'])
        print("Entities extracted:", response_data['entities'])
    
    def demo_navigation(self):
        """Demonstrate navigation request handling"""
        print("\n--- Navigation Demo ---")
        print("Simulating: User says 'Can you go to the kitchen?'")
        
        response_data = self.conversational_system.dialogue_policy.generate_response(
            "Can you go to the kitchen?"
        )
        
        print(f"Robot responds: {response_data['response_text']}")
        print("Intent recognized:", response_data['intent'])
        print("Entities extracted:", response_data['entities'])
        
        # Simulate action execution
        if response_data['intent'] == 'navigation':
            success = self.conversational_system.action_integrator.execute_intent(
                response_data['intent'],
                response_data['entities'],
                response_data['context']
            )
            print(f"Navigation action {'successful' if success else 'failed'}")
    
    def demo_manipulation(self):
        """Demonstrate manipulation request handling"""
        print("\n--- Manipulation Demo ---")
        print("Simulating: User says 'Please pick up the red cup'")
        
        response_data = self.conversational_system.dialogue_policy.generate_response(
            "Please pick up the red cup"
        )
        
        print(f"Robot responds: {response_data['response_text']}")
        print("Intent recognized:", response_data['intent'])
        print("Entities extracted:", response_data['entities'])
        
        # Simulate action execution
        if response_data['intent'] == 'manipulation':
            success = self.conversational_system.action_integrator.execute_intent(
                response_data['intent'],
                response_data['entities'],
                response_data['context']
            )
            print(f"Manipulation action {'successful' if success else 'failed'}")
    
    def demo_mixed_interaction(self):
        """Demonstrate mixed interaction handling"""
        print("\n--- Mixed Interaction Demo ---")
        print("Simulating a conversation with multiple intent types")
        
        conversation = [
            "Hello robot!",
            "How are you today?",
            "Can you move to the living room?",
            "What objects do you see there?",
            "Please bring me the blue bottle"
        ]
        
        for i, utterance in enumerate(conversation):
            print(f"\nUser: {utterance}")
            
            response_data = self.conversational_system.dialogue_policy.generate_response(utterance)
            print(f"Robot: {response_data['response_text']}")
            print(f"  Intent: {response_data['intent']}")
            
            # Execute actions when appropriate
            if response_data['intent'] in ['navigation', 'manipulation']:
                success = self.conversational_system.action_integrator.execute_intent(
                    response_data['intent'],
                    response_data['entities'],
                    response_data['context']
                )
                print(f"  Action {'successful' if success else 'failed'}")
    
    def run_evaluation(self):
        """Run evaluation of the conversational system"""
        print("\n--- System Evaluation ---")
        metrics = self.evaluator.calculate_overall_metrics()
        
        print("Current system metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.2f}")
        
        suggestions = self.evaluator.get_improvement_suggestions()
        print("\nImprovement suggestions:")
        for suggestion in suggestions:
            print(f"  - {suggestion}")

# Mock robot controller for demonstration
class MockRobotController:
    def __init__(self):
        self.position = [0, 0, 0]
        self.locations = {
            'kitchen': [5, 0, 0],
            'living room': [3, 4, 0],
            'bedroom': [-2, 3, 0]
        }
    
    def navigate_to(self, location_name):
        """Mock navigation function"""
        if location_name in self.locations:
            target = self.locations[location_name]
            print(f"  (Mock) Navigating to {location_name} at {target}")
            self.position = target
            return True
        else:
            print(f"  (Mock) Unknown location: {location_name}")
            return False
    
    def manipulate_object(self, object_name):
        """Mock manipulation function"""
        print(f"  (Mock) Attempting to manipulate {object_name}")
        return True
    
    def wave(self):
        """Mock waving action"""
        print("  (Mock) Robot waving")
        return True
    
    def nod(self):
        """Mock nodding action"""
        print("  (Mock) Robot nodding")
        return True
    
    def show_expression(self, emotion):
        """Mock expression showing"""
        print(f"  (Mock) Showing {emotion} expression")
        return True

# Example usage
if __name__ == "__main__":
    robot = CompleteConversationalRobot()
    robot.start_demo()
```

## Summary

Conversational robotics combines multiple technologies to enable natural human-robot interaction through speech and dialogue. Key components include:

- **Speech recognition**: Converting spoken language to text
- **Natural language understanding**: Interpreting user intent and extracting entities
- **Dialogue management**: Coordinating conversation flow and maintaining context
- **Action integration**: Connecting language understanding to robot behaviors
- **Multi-modal coordination**: Integrating speech with vision, gestures, and expressions

Successful conversational robots require careful attention to real-time processing, context management, and seamless integration between understanding and action execution. The field continues to advance with developments in large language models and improved multimodal AI systems.

## Next Steps

In the next lesson, we'll explore integrating GPT models and other large language models specifically for conversational AI in robots, building on the foundation established here.