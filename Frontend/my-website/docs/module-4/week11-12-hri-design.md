---
sidebar_position: 6
title: "Natural Human-Robot Interaction Design"
---

# Natural Human-Robot Interaction Design

This lesson covers the principles and techniques for designing natural and intuitive interactions between humans and humanoid robots.

## Learning Objectives

After completing this lesson, you will be able to:
- Understand the principles of natural human-robot interaction
- Design intuitive interaction modalities (speech, gesture, facial expressions)
- Implement attention and engagement mechanisms
- Create socially appropriate robot behaviors
- Evaluate the effectiveness of human-robot interactions
- Design multimodal interaction systems

## Introduction to Human-Robot Interaction

Human-Robot Interaction (HRI) is a multidisciplinary field that combines insights from psychology, computer science, robotics, and design to create effective interactions between humans and robots.

### What Makes Interaction "Natural"?

Natural HRI involves:
- **Intuitive communication**: Using familiar human communication channels
- **Appropriate social behavior**: Following social norms and expectations
- **Context-aware responses**: Adapting to the situation and environment
- **Predictable behavior**: Acting in ways humans expect
- **Expressive capabilities**: Communicating internal states and intentions

### Key Components of HRI

1. **Perception**: Understanding human behavior, speech, and gestures
2. **Cognition**: Interpreting meaning and deciding on responses
3. **Expression**: Communicating through speech, gestures, and expressions
4. **Social rules**: Following cultural and social conventions
5. **Adaptation**: Learning and adjusting to individual users

## Communication Modalities

### Speech and Language

Speech is the primary communication modality for natural HRI:

#### Speech Recognition
```python
import speech_recognition as sr
import threading
import queue

class SpeechRecognizer:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.audio_queue = queue.Queue()
        self.listening = False
        self.user_intent = None
        
        # Set up microphone
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
    
    def start_listening(self):
        """Start listening for speech"""
        self.listening = True
        self.listen_thread = threading.Thread(target=self._listen_continuously)
        self.listen_thread.start()
    
    def _listen_continuously(self):
        """Continuously listen for speech"""
        with self.microphone as source:
            while self.listening:
                try:
                    # Listen for audio
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                    
                    # Add to queue for processing
                    self.audio_queue.put(audio)
                    
                except sr.WaitTimeoutError:
                    # No speech detected, continue listening
                    continue
                except Exception as e:
                    print(f"Listening error: {e}")
                    continue
    
    def process_audio(self):
        """Process audio from queue"""
        try:
            audio = self.audio_queue.get(timeout=0.1)
            
            # Recognize speech
            try:
                text = self.recognizer.recognize_google(audio)
                print(f"Recognized: {text}")
                
                # Parse intent
                self.user_intent = self.parse_intent(text)
                return text, self.user_intent
            except sr.UnknownValueError:
                print("Could not understand audio")
                return None, None
            except sr.RequestError as e:
                print(f"Recognition error: {e}")
                return None, None
        except queue.Empty:
            return None, None
    
    def parse_intent(self, text):
        """Parse user intent from recognized text"""
        text_lower = text.lower()
        
        # Simple intent parsing (in practice, use NLP models)
        if any(word in text_lower for word in ["hello", "hi", "hey"]):
            return "greeting"
        elif any(word in text_lower for word in ["help", "assist", "can you"]):
            return "request_assistance"
        elif any(word in text_lower for word in ["move", "go", "walk", "step"]):
            return "navigation_request"
        elif any(word in text_lower for word in ["grasp", "take", "pick", "lift"]):
            return "manipulation_request"
        elif any(word in text_lower for word in ["stop", "cancel", "abort"]):
            return "cancel_request"
        elif any(word in text_lower for word in ["thank", "thanks", "thank you"]):
            return "appreciation"
        else:
            return "unknown"
```

#### Text-to-Speech
```python
import pyttsx3
import threading

class TextToSpeech:
    def __init__(self):
        self.engine = pyttsx3.init()
        
        # Configure voice properties
        voices = self.engine.getProperty('voices')
        if voices:
            self.engine.setProperty('voice', voices[0].id)  # Use first available voice
        
        self.engine.setProperty('rate', 150)  # Speed of speech
        self.engine.setProperty('volume', 0.9)  # Volume level
        
        self.speaking = False
        self.speech_queue = queue.Queue()
    
    def speak(self, text, blocking=False):
        """Speak the given text"""
        if blocking:
            self.engine.say(text)
            self.engine.runAndWait()
        else:
            # Speak in separate thread to avoid blocking
            speak_thread = threading.Thread(target=self._speak_non_blocking, args=(text,))
            speak_thread.start()
    
    def _speak_non_blocking(self, text):
        """Speak text without blocking"""
        self.speaking = True
        self.engine.say(text)
        self.engine.runAndWait()
        self.speaking = False
    
    def interrupt_speech(self):
        """Interrupt current speech"""
        self.engine.stop()
    
    def set_emotion(self, emotion="neutral"):
        """Set emotional tone for speech"""
        if emotion == "happy":
            self.engine.setProperty('rate', 160)  # Slightly faster
        elif emotion == "sad":
            self.engine.setProperty('rate', 130)  # Slightly slower
        elif emotion == "excited":
            self.engine.setProperty('rate', 170)  # Much faster
            self.engine.setProperty('volume', 1.0)  # Louder
        elif emotion == "calm":
            self.engine.setProperty('rate', 140)  # Slower
            self.engine.setProperty('volume', 0.7)  # Softer
```

### Gesture and Body Language

#### Gesture Recognition
```python
import cv2
import mediapipe as mp
import numpy as np

class GestureRecognizer:
    def __init__(self):
        # Initialize MediaPipe for hand tracking
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Gesture patterns
        self.gesture_patterns = {
            'wave': self.is_wave_gesture,
            'point': self.is_pointing_gesture,
            'come_here': self.is_come_here_gesture,
            'stop': self.is_stop_gesture,
            'thumbs_up': self.is_thumbs_up_gesture,
            'peace_sign': self.is_peace_sign
        }
    
    def recognize_gesture(self, image):
        """Recognize gesture from image"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                
                # Check for gestures
                for gesture_name, gesture_func in self.gesture_patterns.items():
                    if gesture_func(hand_landmarks):
                        return gesture_name, hand_landmarks
        
        return None, None
    
    def is_wave_gesture(self, landmarks):
        """Detect wave gesture"""
        # Wave is typically characterized by moving hand with fingers extended
        # and thumb tucked in, moving side to side
        return self.is_fingers_extended(landmarks) and self.is_thumb_folded(landmarks)
    
    def is_pointing_gesture(self, landmarks):
        """Detect pointing gesture"""
        # Pointing: index finger extended, others folded
        return (self.is_finger_extended(landmarks, 8) and  # Index finger tip
                self.are_other_fingers_folded(landmarks, [8]))  # Except index
    
    def is_come_here_gesture(self, landmarks):
        """Detect come here gesture (palm facing out, fingers curled)"""
        # Come here: palm facing out, fingers curled as if beckoning
        return (self.is_palm_facing_out(landmarks) and
                self.is_fingers_curled(landmarks))
    
    def is_stop_gesture(self, landmarks):
        """Detect stop gesture (palm facing forward)"""
        # Stop: palm facing forward, all fingers extended
        return (self.is_palm_facing_forward(landmarks) and
                self.are_all_fingers_extended(landmarks))
    
    def is_thumbs_up_gesture(self, landmarks):
        """Detect thumbs up gesture"""
        # Thumbs up: thumb extended, other fingers folded
        return (self.is_finger_extended(landmarks, 4) and  # Thumb tip
                self.are_other_fingers_folded(landmarks, [4]))  # Except thumb
    
    def is_peace_sign(self, landmarks):
        """Detect peace sign (index and middle fingers extended)"""
        # Peace sign: index and middle fingers extended, others folded
        return (self.is_finger_extended(landmarks, 8) and  # Index tip
                self.is_finger_extended(landmarks, 12) and  # Middle tip
                self.are_other_fingers_folded(landmarks, [8, 12]))  # Except index and middle
    
    def is_finger_extended(self, landmarks, landmark_idx):
        """Check if a specific finger is extended"""
        # Compare fingertip position to base position
        fingertip = np.array([landmarks.landmark[landmark_idx].x,
                             landmarks.landmark[landmark_idx].y,
                             landmarks.landmark[landmark_idx].z])
        
        base = np.array([landmarks.landmark[landmark_idx - 3].x,
                        landmarks.landmark[landmark_idx - 3].y,
                        landmarks.landmark[landmark_idx - 3].z])
        
        # If finger is extended, distance will be larger
        distance = np.linalg.norm(fingertip - base)
        return distance > 0.1  # Threshold value
    
    def is_fingers_extended(self, landmarks):
        """Check if all fingers are extended"""
        finger_tips = [8, 12, 16, 20]  # Tips of index, middle, ring, pinky
        return all(self.is_finger_extended(landmarks, tip) for tip in finger_tips)
    
    def are_all_fingers_extended(self, landmarks):
        """Check if all fingers (except thumb) are extended"""
        finger_tips = [8, 12, 16, 20]  # Index, middle, ring, pinky
        return all(self.is_finger_extended(landmarks, tip) for tip in finger_tips)
    
    def are_other_fingers_folded(self, landmarks, extended_fingers):
        """Check if all fingers except specified ones are folded"""
        all_finger_tips = [4, 8, 12, 16, 20]  # Thumb, index, middle, ring, pinky
        other_fingers = [tip for tip in all_finger_tips if tip not in extended_fingers]
        return all(not self.is_finger_extended(landmarks, tip) for tip in other_fingers)
    
    def is_thumb_folded(self, landmarks):
        """Check if thumb is folded"""
        return not self.is_finger_extended(landmarks, 4)
    
    def is_palm_facing_out(self, landmarks):
        """Check if palm is facing out (towards camera)"""
        # Compare wrist to finger base positions
        wrist = np.array([landmarks.landmark[0].x, landmarks.landmark[0].y, landmarks.landmark[0].z])
        index_mcp = np.array([landmarks.landmark[5].x, landmarks.landmark[5].y, landmarks.landmark[5].z])
        middle_mcp = np.array([landmarks.landmark[9].x, landmarks.landmark[9].y, landmarks.landmark[9].z])
        
        # Palm facing out when wrist is behind (higher z-value) than finger bases
        return (wrist[2] > index_mcp[2] and wrist[2] > middle_mcp[2])
    
    def is_palm_facing_forward(self, landmarks):
        """Check if palm is facing forward"""
        # Simplified check - in practice would use more sophisticated analysis
        return self.is_palm_facing_out(landmarks) and self.is_fingers_extended(landmarks)
    
    def is_fingers_curled(self, landmarks):
        """Check if fingers are curled (for beckoning)"""
        # Fingers curled when fingertips are closer to palm than extended position
        finger_tips = [8, 12, 16, 20]
        return all(not self.is_finger_extended(landmarks, tip) for tip in finger_tips)
```

#### Gesture Response System
```python
class GestureResponseSystem:
    def __init__(self, robot_controller):
        self.robot_controller = robot_controller
        self.gesture_recognizer = GestureRecognizer()
        self.response_mapping = {
            'wave': self.respond_to_wave,
            'point': self.respond_to_point,
            'come_here': self.move_towards_user,
            'stop': self.stop_movement,
            'thumbs_up': self.acknowledge_positive_feedback,
            'peace_sign': self.relax_posture
        }
    
    def process_gesture(self, gesture_name, landmarks):
        """Process recognized gesture and trigger appropriate response"""
        if gesture_name in self.response_mapping:
            response_func = self.response_mapping[gesture_name]
            response_func(landmarks)
    
    def respond_to_wave(self, landmarks):
        """Respond to waving gesture"""
        print("Detected wave! Waving back...")
        self.robot_controller.wave_back()
    
    def respond_to_point(self, landmarks):
        """Respond to pointing gesture"""
        print("Detected pointing! Looking in that direction...")
        direction = self.calculate_pointing_direction(landmarks)
        self.robot_controller.look_at_direction(direction)
    
    def move_towards_user(self, landmarks):
        """Move towards user when beckoned"""
        print("User beckoning! Moving closer...")
        self.robot_controller.move_towards_user()
    
    def stop_movement(self, landmarks):
        """Stop all movement"""
        print("Stop gesture detected! Stopping...")
        self.robot_controller.stop_all_motors()
    
    def acknowledge_positive_feedback(self, landmarks):
        """Acknowledge positive feedback"""
        print("Thumbs up detected! Responding positively...")
        self.robot_controller.acknowledge_with_positive_expression()
    
    def relax_posture(self, landmarks):
        """Adopt relaxed posture"""
        print("Peace sign detected! Relaxing posture...")
        self.robot_controller.adopt_relaxed_posture()
    
    def calculate_pointing_direction(self, landmarks):
        """Calculate direction of pointing gesture"""
        # Simplified: calculate direction from wrist to index finger
        wrist = np.array([landmarks.landmark[0].x, landmarks.landmark[0].y, landmarks.landmark[0].z])
        index_tip = np.array([landmarks.landmark[8].x, landmarks.landmark[8].y, landmarks.landmark[8].z])
        
        direction = index_tip - wrist
        direction = direction / np.linalg.norm(direction)  # Normalize
        
        return direction
```

### Facial Expressions and Emotions

#### Expression Generation
```python
class FacialExpressionController:
    def __init__(self):
        # Define facial expression parameters
        self.expressions = {
            'neutral': {'eyebrows': 0, 'eyes': 0, 'mouth': 0},
            'happy': {'eyebrows': -0.2, 'eyes': 0.8, 'mouth': 0.8},
            'sad': {'eyebrows': 0.3, 'eyes': -0.3, 'mouth': -0.5},
            'surprised': {'eyebrows': 0.8, 'eyes': 0.9, 'mouth': 0.6},
            'angry': {'eyebrows': 0.7, 'eyes': 0.6, 'mouth': -0.3},
            'confused': {'eyebrows': 0.4, 'eyes': 0.2, 'mouth': 0.1},
            'attentive': {'eyebrows': 0.1, 'eyes': 0.7, 'mouth': 0.2}
        }
        
        # Current expression state
        self.current_expression = 'neutral'
        self.expression_intensity = 1.0
    
    def set_expression(self, expression_name, intensity=1.0):
        """Set facial expression with specified intensity"""
        if expression_name in self.expressions:
            self.current_expression = expression_name
            self.expression_intensity = intensity
            self.apply_expression(expression_name, intensity)
    
    def apply_expression(self, expression_name, intensity):
        """Apply the expression to the robot's face"""
        expression_params = self.expressions[expression_name]
        
        # Apply parameters with intensity scaling
        scaled_params = {key: value * intensity for key, value in expression_params.items()}
        
        # In a real robot, this would control facial servos
        self.control_facial_servos(scaled_params)
    
    def control_facial_servos(self, params):
        """Control the facial expression servos"""
        # This would interface with the actual robot hardware
        print(f"Setting facial expression: {params}")
        
        # Example: move eyebrow servos
        # self.eyebrow_left_servo.move_to(params['eyebrows'] * 90)
        # self.eyebrow_right_servo.move_to(params['eyebrows'] * 90)
        
        # Example: control eye displays
        # self.eye_display.show_expression(params['eyes'])
        
        # Example: control mouth servos
        # self.mouth_servo.move_to(params['mouth'] * 45)
    
    def blend_expressions(self, expr1, expr2, ratio):
        """Blend between two expressions"""
        if expr1 in self.expressions and expr2 in self.expressions:
            params1 = self.expressions[expr1]
            params2 = self.expressions[expr2]
            
            blended_params = {}
            for key in params1.keys():
                blended_value = params1[key] * (1 - ratio) + params2[key] * ratio
                blended_params[key] = blended_value
            
            # Apply blended expression
            self.apply_expression_blended(blended_params)
    
    def apply_expression_blended(self, params):
        """Apply blended expression parameters"""
        self.control_facial_servos(params)
    
    def react_to_interaction(self, interaction_type):
        """React to different types of interactions with appropriate expressions"""
        reaction_map = {
            'greeting': 'happy',
            'question': 'attentive',
            'instruction': 'attentive',
            'compliment': 'happy',
            'error': 'confused',
            'success': 'happy',
            'farewell': 'happy'
        }
        
        if interaction_type in reaction_map:
            self.set_expression(reaction_map[interaction_type], intensity=0.7)
```

## Attention and Engagement Systems

### Attention Mechanisms

```python
class AttentionSystem:
    def __init__(self):
        self.attended_person = None
        self.attention_history = []
        self.focus_level = 0.0
        self.gaze_target = None
        
        # Social attention rules
        self.social_attention_rules = {
            'closest_person': 0.3,
            'speaking_person': 0.5,
            'gesturing_person': 0.2
        }
    
    def update_attention(self, detected_people):
        """Update attention based on detected people"""
        if not detected_people:
            self.attended_person = None
            return
        
        # Determine who to attend to based on social rules
        attended_candidate = self.select_attended_person(detected_people)
        
        if attended_candidate != self.attended_person:
            self.switch_attention(attended_candidate)
        
        # Update attention history
        self.attention_history.append({
            'timestamp': time.time(),
            'attended_person': attended_candidate,
            'focus_level': self.focus_level
        })
        
        # Keep history to reasonable length
        if len(self.attention_history) > 100:
            self.attention_history = self.attention_history[-50:]
    
    def select_attended_person(self, detected_people):
        """Select which person to attend to based on social rules"""
        scores = {}
        
        for person_id, person_data in detected_people.items():
            score = 0.0
            
            # Distance-based attention (closer gets higher priority)
            distance = person_data.get('distance', float('inf'))
            if distance < float('inf'):
                distance_score = max(0, 1 - (distance / 5.0))  # 5m normalization
                score += self.social_attention_rules['closest_person'] * distance_score
            
            # Speaking-based attention
            if person_data.get('is_speaking', False):
                score += self.social_attention_rules['speaking_person']
            
            # Gesturing-based attention
            if person_data.get('is_gesturing', False):
                score += self.social_attention_rules['gesturing_person']
            
            scores[person_id] = score
        
        # Return person with highest score
        if scores:
            return max(scores, key=scores.get)
        else:
            return None
    
    def switch_attention(self, new_person):
        """Switch attention to a new person"""
        old_person = self.attended_person
        self.attended_person = new_person
        
        # Trigger attention switch behavior
        self.on_attention_switch(old_person, new_person)
    
    def on_attention_switch(self, old_person, new_person):
        """Handle behavior when attention switches"""
        print(f"Attention switched from {old_person} to {new_person}")
        
        # Turn gaze towards new person
        if new_person:
            self.turn_gaze_towards(new_person)
        
        # Potentially acknowledge the switch
        self.acknowledge_attention_switch(new_person)
    
    def turn_gaze_towards(self, person_id):
        """Turn robot's gaze towards a person"""
        # In a real robot, this would control neck/head servos
        print(f"Turning gaze towards person {person_id}")
        
        # This would involve calculating the position of the person
        # and commanding the head to look in that direction
        # self.head_controller.look_at(person_position)
    
    def acknowledge_attention_switch(self, person_id):
        """Acknowledge attention switch with subtle behavior"""
        # Small acknowledgment like a nod or slight head turn
        print(f"Acknowledging attention to person {person_id}")
        
        # self.head_controller.nod_subtly()
```

### Engagement Strategies

```python
class EngagementSystem:
    def __init__(self):
        self.engagement_level = 0.0
        self.engagement_history = []
        self.active_interactions = []
        
        # Engagement parameters
        self.engagement_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        }
    
    def update_engagement(self, interaction_data):
        """Update engagement based on interaction"""
        # Calculate engagement score based on interaction
        engagement_score = self.calculate_engagement_score(interaction_data)
        
        # Update engagement level with smoothing
        self.engagement_level = 0.7 * self.engagement_level + 0.3 * engagement_score
        
        # Store in history
        self.engagement_history.append({
            'timestamp': time.time(),
            'score': engagement_score,
            'level': self.engagement_level
        })
        
        # Trigger appropriate behaviors based on engagement level
        self.trigger_engagement_behaviors()
    
    def calculate_engagement_score(self, interaction_data):
        """Calculate engagement score from interaction data"""
        score = 0.0
        
        # Positive interactions increase engagement
        if interaction_data.get('positive_feedback', False):
            score += 0.3
        
        # Eye contact increases engagement
        if interaction_data.get('eye_contact', False):
            score += 0.2
        
        # Response to robot's actions increases engagement
        if interaction_data.get('responded_to_robot', False):
            score += 0.25
        
        # Sustained interaction increases engagement
        interaction_duration = interaction_data.get('duration', 0)
        if interaction_duration > 10:  # More than 10 seconds
            score += 0.15
        
        # Conversation turns increase engagement
        conversation_turns = interaction_data.get('conversation_turns', 0)
        if conversation_turns > 2:
            score += 0.1 * min(conversation_turns, 5)  # Up to 0.5 for 5+ turns
        
        return min(score, 1.0)  # Cap at 1.0
    
    def trigger_engagement_behaviors(self):
        """Trigger appropriate behaviors based on engagement level"""
        if self.engagement_level > self.engagement_thresholds['high']:
            self.high_engagement_behavior()
        elif self.engagement_level > self.engagement_thresholds['medium']:
            self.medium_engagement_behavior()
        elif self.engagement_level > self.engagement_thresholds['low']:
            self.low_engagement_behavior()
        else:
            self.disengaged_behavior()
    
    def high_engagement_behavior(self):
        """Behaviors for high engagement"""
        print("High engagement detected - maintaining interaction")
        
        # Maintain eye contact
        # self.face_tracker.maintain_eye_contact()
        
        # Show interested expressions
        # self.facial_controller.set_expression('attentive')
        
        # Ask follow-up questions
        # self.conversation_manager.ask_follow_up_question()
    
    def medium_engagement_behavior(self):
        """Behaviors for medium engagement"""
        print("Medium engagement - encouraging continuation")
        
        # Slight nodding
        # self.head_controller.nod_occasionally()
        
        # Positive expressions
        # self.facial_controller.set_expression('happy', intensity=0.5)
        
        # Invite further interaction
        # self.speech_system.speak("What else can I help you with?")
    
    def low_engagement_behavior(self):
        """Behaviors for low engagement"""
        print("Low engagement - trying to re-engage")
        
        # Gentle attention-getting behavior
        # self.head_controller.turn_slightly()
        
        # Friendly expression
        # self.facial_controller.set_expression('friendly')
        
        # Attempt to re-engage
        # self.speech_system.speak("Are you still interested in talking?")
    
    def disengaged_behavior(self):
        """Behaviors for disengaged state"""
        print("Disengaged - backing off respectfully")
        
        # Reduce intensity of behaviors
        # self.head_controller.return_to_neutral()
        
        # Neutral expression
        # self.facial_controller.set_expression('neutral')
        
        # Prepare for potential farewell
        # self.conversation_manager.prepare_farewell()
```

## Multimodal Interaction Systems

### Integrating Multiple Modalities

```python
class MultimodalInteractionManager:
    def __init__(self):
        # Initialize different modalities
        self.speech_recognizer = SpeechRecognizer()
        self.text_to_speech = TextToSpeech()
        self.gesture_recognizer = GestureRecognizer()
        self.facial_controller = FacialExpressionController()
        self.attention_system = AttentionSystem()
        self.engagement_system = EngagementSystem()
        
        # Interaction state
        self.current_conversation_partner = None
        self.interaction_context = {}
        self.last_interaction_time = time.time()
        
        # Start listening systems
        self.speech_recognizer.start_listening()
    
    def process_interaction(self):
        """Process incoming interactions from multiple modalities"""
        # Process speech
        speech_result = self.speech_recognizer.process_audio()
        if speech_result[0]:  # If speech was recognized
            text, intent = speech_result
            self.handle_speech_input(text, intent)
        
        # Process gestures (this would be done with video input)
        # gesture_result = self.gesture_recognizer.recognize_gesture(video_frame)
        # if gesture_result[0]:  # If gesture was recognized
        #     gesture_name, landmarks = gesture_result
        #     self.handle_gesture_input(gesture_name, landmarks)
        
        # Update attention system
        # self.attention_system.update_attention(detected_people)
        
        # Update engagement system
        # self.update_engagement_metrics()
    
    def handle_speech_input(self, text, intent):
        """Handle speech input and determine appropriate response"""
        print(f"Received speech: '{text}' with intent: {intent}")
        
        # Update interaction context
        self.interaction_context['last_speech'] = {
            'text': text,
            'intent': intent,
            'timestamp': time.time()
        }
        
        # Determine response based on intent
        response = self.generate_response(intent, text)
        
        # Execute response
        self.execute_response(response)
        
        # Update engagement metrics
        self.update_engagement_after_interaction(intent, response)
    
    def handle_gesture_input(self, gesture_name, landmarks):
        """Handle gesture input"""
        print(f"Detected gesture: {gesture_name}")
        
        # Update interaction context
        self.interaction_context['last_gesture'] = {
            'gesture': gesture_name,
            'timestamp': time.time()
        }
        
        # Process gesture and generate response
        response = self.process_gesture(gesture_name, landmarks)
        self.execute_response(response)
    
    def generate_response(self, intent, input_text):
        """Generate appropriate response based on intent"""
        response_templates = {
            'greeting': [
                "Hello! Nice to meet you!",
                "Hi there! How can I help you?",
                "Greetings! What brings you here today?"
            ],
            'request_assistance': [
                "I'd be happy to help. What do you need assistance with?",
                "Sure, I can help with that. Tell me more about what you need.",
                "Of course! I'm here to assist. What can I do for you?"
            ],
            'navigation_request': [
                "I can help with navigation. Where would you like me to go?",
                "Sure, I can navigate to a location. Where should I go?",
                "I'm equipped for navigation. Please specify your destination."
            ],
            'manipulation_request': [
                "I can help with manipulation tasks. What would you like me to grasp?",
                "Sure, I can manipulate objects. What do you need me to handle?",
                "I'm capable of manipulation. What object would you like me to work with?"
            ],
            'appreciation': [
                "You're welcome! I'm glad I could help.",
                "Thank you! It's my pleasure to assist.",
                "I appreciate that! Is there anything else I can do for you?"
            ],
            'unknown': [
                "I'm sorry, I didn't quite understand that. Could you repeat?",
                "I'm not sure I caught that. Could you say it again?",
                "I didn't understand. Could you rephrase that?"
            ]
        }
        
        import random
        if intent in response_templates:
            response = random.choice(response_templates[intent])
        else:
            response = random.choice(response_templates['unknown'])
        
        return {
            'type': 'verbal_response',
            'text': response,
            'emotion': self.determine_response_emotion(intent)
        }
    
    def process_gesture(self, gesture_name, landmarks):
        """Process gesture and generate appropriate response"""
        response = {
            'type': 'gesture_response',
            'gesture': gesture_name,
            'accompanying_speech': self.get_gesture_response_speech(gesture_name)
        }
        
        return response
    
    def get_gesture_response_speech(self, gesture_name):
        """Get appropriate speech response for a gesture"""
        speech_responses = {
            'wave': "Hello! I see you waving. How can I help?",
            'point': "I see you're pointing. Are you directing my attention?",
            'come_here': "I'm coming over now!",
            'stop': "I understand. I'll stop what I'm doing.",
            'thumbs_up': "Thank you for the thumbs up!",
            'peace_sign': "Nice peace sign! I'm relaxed and ready to help."
        }
        
        return speech_responses.get(gesture_name, "I noticed your gesture.")
    
    def determine_response_emotion(self, intent):
        """Determine appropriate emotional tone for response"""
        emotion_map = {
            'greeting': 'happy',
            'request_assistance': 'helpful',
            'navigation_request': 'confident',
            'manipulation_request': 'capable',
            'appreciation': 'pleased',
            'unknown': 'curious'
        }
        
        return emotion_map.get(intent, 'neutral')
    
    def execute_response(self, response):
        """Execute the generated response"""
        if response['type'] == 'verbal_response':
            # Speak the response
            self.text_to_speech.speak(response['text'])
            
            # Show appropriate facial expression
            emotion = response.get('emotion', 'neutral')
            self.facial_controller.set_expression(emotion)
        
        elif response['type'] == 'gesture_response':
            # Perform accompanying gesture
            # self.perform_accompanying_gesture(response['gesture'])
            
            # Speak the accompanying text
            if response.get('accompanying_speech'):
                self.text_to_speech.speak(response['accompanying_speech'])
    
    def update_engagement_after_interaction(self, intent, response):
        """Update engagement metrics after interaction"""
        interaction_data = {
            'intent': intent,
            'response_type': response['type'],
            'timestamp': time.time(),
            'positive_feedback': intent in ['appreciation', 'greeting']
        }
        
        self.engagement_system.update_engagement(interaction_data)
    
    def maintain_interaction_flow(self):
        """Maintain natural flow of interaction"""
        # Check if interaction has been idle
        time_since_last = time.time() - self.last_interaction_time
        
        if time_since_last > 10:  # 10 seconds idle
            self.handle_interaction_idle()
    
    def handle_interaction_idle(self):
        """Handle case when interaction is idle"""
        print("Interaction idle - checking engagement")
        
        if self.engagement_system.engagement_level > 0.5:
            # Still engaged, initiate follow-up
            self.initiate_follow_up()
        else:
            # Disengaged, consider ending interaction
            self.consider_ending_interaction()
    
    def initiate_follow_up(self):
        """Initiate a follow-up to maintain interaction"""
        follow_ups = [
            "Is there anything else I can help you with?",
            "Do you have any other questions?",
            "What else would you like to know?"
        ]
        
        import random
        follow_up = random.choice(follow_ups)
        self.text_to_speech.speak(follow_up)
        self.facial_controller.set_expression('attentive')
    
    def consider_ending_interaction(self):
        """Consider ending the interaction respectfully"""
        if self.current_conversation_partner:
            farewell = "Well, it was nice talking with you!"
            self.text_to_speech.speak(farewell)
            self.facial_controller.set_expression('happy')
            self.current_conversation_partner = None
```

## Social Norms and Cultural Considerations

### Cultural Adaptation

```python
class CulturalAdapter:
    def __init__(self):
        self.cultural_profiles = {
            'default': {
                'personal_space': 1.0,  # meters
                'eye_contact_norms': 'moderate',
                'greeting_style': 'handshake',
                'formality_level': 'neutral',
                'touch_norms': 'limited'
            },
            'japanese': {
                'personal_space': 1.2,
                'eye_contact_norms': 'respectful_avoidance',
                'greeting_style': 'bow',
                'formality_level': 'high',
                'touch_norms': 'avoid_physical_contact'
            },
            'middle_eastern': {
                'personal_space': 0.8,
                'eye_contact_norms': 'direct_male_same_sex',
                'greeting_style': 'handshake_same_gender',
                'formality_level': 'high',
                'touch_norms': 'gender_segregated'
            },
            'mediterranean': {
                'personal_space': 0.6,
                'eye_contact_norms': 'direct',
                'greeting_style': 'handshake_close_contact',
                'formality_level': 'warm',
                'touch_norms': 'accepting'
            }
        }
        
        self.current_culture = 'default'
        self.user_preferences = {}
    
    def set_cultural_context(self, culture_name):
        """Set the cultural context for interaction"""
        if culture_name in self.cultural_profiles:
            self.current_culture = culture_name
            print(f"Set cultural context to: {culture_name}")
        else:
            print(f"Unknown culture: {culture_name}, using default")
    
    def get_cultural_preference(self, aspect):
        """Get cultural preference for a specific aspect"""
        profile = self.cultural_profiles.get(self.current_culture, self.cultural_profiles['default'])
        return profile.get(aspect, None)
    
    def adapt_behavior_to_culture(self, behavior):
        """Adapt robot behavior based on cultural norms"""
        adapted_behavior = behavior.copy()
        
        # Adjust personal space based on culture
        if 'approach_distance' in adapted_behavior:
            cultural_space = self.get_cultural_preference('personal_space')
            if cultural_space:
                adapted_behavior['approach_distance'] = cultural_space
        
        # Adjust eye contact based on culture
        if 'eye_contact_duration' in adapted_behavior:
            eye_norms = self.get_cultural_preference('eye_contact_norms')
            if eye_norms == 'respectful_avoidance':
                adapted_behavior['eye_contact_duration'] *= 0.5  # Reduce eye contact
            elif eye_norms == 'direct':
                adapted_behavior['eye_contact_duration'] *= 1.2  # Increase eye contact
        
        # Adjust formality based on culture
        if 'formal_language' in adapted_behavior:
            formality = self.get_cultural_preference('formality_level')
            if formality == 'high':
                adapted_behavior['formal_language'] = True
            elif formality == 'warm':
                adapted_behavior['formal_language'] = False
        
        return adapted_behavior
```

## Evaluation and Design Principles

### HRI Design Principles

```python
class HRIDesignEvaluator:
    def __init__(self):
        self.design_principles = {
            'predictability': {
                'description': 'Robot behavior should be predictable',
                'importance': 0.9,
                'metrics': ['behavior_consistency', 'response_time_variance']
            },
            'transparency': {
                'description': 'Robot should communicate its intentions clearly',
                'importance': 0.85,
                'metrics': ['intent_communication', 'state_visibility']
            },
            'appropriateness': {
                'description': 'Robot behavior should be appropriate for context',
                'importance': 0.8,
                'metrics': ['context_awareness', 'social_norm_adherence']
            },
            'efficiency': {
                'description': 'Interaction should be efficient',
                'importance': 0.75,
                'metrics': ['task_completion_time', 'misunderstanding_rate']
            },
            'engagement': {
                'description': 'Robot should maintain appropriate engagement',
                'importance': 0.7,
                'metrics': ['attention_retention', 'interaction_duration']
            }
        }
    
    def evaluate_interaction(self, interaction_log):
        """Evaluate interaction based on design principles"""
        evaluations = {}
        
        for principle, config in self.design_principles.items():
            metric_values = []
            
            for metric in config['metrics']:
                if metric in interaction_log:
                    value = self.calculate_metric_score(metric, interaction_log[metric])
                    metric_values.append(value)
            
            if metric_values:
                avg_score = sum(metric_values) / len(metric_values)
                evaluations[principle] = {
                    'score': avg_score,
                    'importance': config['importance'],
                    'weighted_score': avg_score * config['importance']
                }
        
        return evaluations
    
    def calculate_metric_score(self, metric_name, metric_data):
        """Calculate score for a specific metric"""
        # This would contain specific scoring logic for each metric
        if metric_name == 'behavior_consistency':
            # Higher consistency scores better
            return min(1.0, metric_data.get('consistency_score', 0.5))
        elif metric_name == 'response_time_variance':
            # Lower variance scores better
            max_acceptable_variance = 0.5
            variance = metric_data.get('variance', 1.0)
            return max(0.0, 1.0 - (variance / max_acceptable_variance))
        elif metric_name == 'intent_communication':
            # Higher communication scores better
            return metric_data.get('communication_score', 0.5)
        else:
            # Default scoring
            return 0.5
```

## Practical Exercise: Designing an HRI System

Create a complete human-robot interaction system:

1. **Implement speech recognition and synthesis**
2. **Add gesture recognition capabilities**
3. **Create facial expression system**
4. **Implement attention and engagement systems**
5. **Test with natural interaction scenarios**

### Complete HRI System Example

```python
class CompleteHRI:
    def __init__(self):
        self.multimodal_manager = MultimodalInteractionManager()
        self.cultural_adapter = CulturalAdapter()
        self.design_evaluator = HRIDesignEvaluator()
        
        # Interaction state
        self.interaction_log = []
        self.user_profiles = {}
        
        print("Complete HRI System initialized")
    
    def start_interaction_session(self):
        """Start a new interaction session"""
        print("Starting new interaction session...")
        
        # Initialize systems
        self.multimodal_manager.current_conversation_partner = None
        self.interaction_log = []
        
        # Begin monitoring for interactions
        self.monitor_interactions()
    
    def monitor_interactions(self):
        """Monitor and process ongoing interactions"""
        import time
        
        try:
            while True:
                # Process multimodal inputs
                self.multimodal_manager.process_interaction()
                
                # Maintain interaction flow
                self.multimodal_manager.maintain_interaction_flow()
                
                # Small delay to prevent overwhelming CPU
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nInteraction session ended by user")
            self.end_interaction_session()
    
    def end_interaction_session(self):
        """End current interaction session and evaluate"""
        print("Ending interaction session...")
        
        # Evaluate the interaction
        evaluation = self.design_evaluator.evaluate_interaction(self.interaction_log)
        print("Interaction evaluation:")
        for principle, results in evaluation.items():
            print(f"  {principle}: {results['score']:.2f} (weighted: {results['weighted_score']:.2f})")
    
    def run_demo_scenario(self):
        """Run a demonstration scenario"""
        print("Running HRI demonstration...")
        
        # Simulate a simple interaction scenario
        print("\nScenario: User approaches robot and asks for help")
        print("- Robot detects user and turns attention towards them")
        print("- User waves and says 'Hi, can you help me?'")
        print("- Robot responds with 'Hello! Of course, how can I help?'")
        print("- Interaction continues...")
        
        # In a real implementation, this would involve:
        # 1. Person detection and attention
        # 2. Gesture recognition (wave)
        # 3. Speech recognition ("Hi, can you help me?")
        # 4. Appropriate response generation and execution
        
        print("\nDemonstration complete!")

# Example usage
if __name__ == "__main__":
    hri_system = CompleteHRI()
    
    print("HRI System Demo")
    print("1. Start interaction session")
    print("2. Run demo scenario")
    print("3. Exit")
    
    choice = input("Enter choice (1-3): ")
    
    if choice == "1":
        hri_system.start_interaction_session()
    elif choice == "2":
        hri_system.run_demo_scenario()
    else:
        print("Exiting...")
```

## Summary

Natural human-robot interaction design involves creating systems that can communicate effectively with humans using familiar modalities like speech, gestures, and expressions. Key elements include:

- **Multimodal communication**: Using speech, gesture, and facial expressions together
- **Attention and engagement**: Maintaining appropriate focus and interest
- **Social appropriateness**: Following cultural and social norms
- **Predictable behavior**: Acting in ways humans expect
- **Context awareness**: Adapting to the situation and environment

Successful HRI requires integrating multiple technologies and considering human factors, cultural differences, and social norms. The field continues to evolve as robots become more prevalent in human environments.

## Next Steps

In the next lesson, we'll explore conversational robotics, building on the interaction design principles to create robots that can engage in natural conversations with humans.