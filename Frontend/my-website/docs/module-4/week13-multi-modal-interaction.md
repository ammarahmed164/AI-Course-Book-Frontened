---
sidebar_position: 10
title: "Multi-Modal Interaction: Speech, Gesture, Vision"
---

# Multi-Modal Interaction: Speech, Gesture, Vision

This lesson covers how to integrate multiple interaction modalities (speech, gesture, vision) to create natural and intuitive human-robot interaction.

## Learning Objectives

After completing this lesson, you will be able to:
- Integrate speech, gesture, and vision modalities into a unified interaction system
- Design multi-modal fusion algorithms
- Implement attention and gaze systems
- Create coordinated multi-modal responses
- Evaluate multi-modal interaction quality
- Handle conflicts between modalities

## Introduction to Multi-Modal Interaction

Multi-modal interaction combines multiple communication channels (speech, gesture, vision, touch, etc.) to create more natural and intuitive human-robot interaction. This approach mimics how humans naturally communicate using multiple channels simultaneously.

### Benefits of Multi-Modal Interaction

1. **Naturalness**: Mirrors human communication patterns
2. **Redundancy**: Multiple channels provide backup communication
3. **Expressiveness**: Enables richer expression of intent and emotion
4. **Robustness**: Can handle noisy or incomplete information in one modality
5. **Context**: Provides richer context for understanding

### Challenges in Multi-Modal Integration

1. **Synchronization**: Aligning information across different modalities
2. **Fusion**: Combining information from different sources effectively
3. **Timing**: Managing different processing latencies
4. **Conflict Resolution**: Handling contradictory information across modalities
5. **Attention**: Determining which modalities to focus on

## Multi-Modal Architecture

### Component-Based Architecture

```python
import threading
import queue
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class ModalityType(Enum):
    SPEECH = "speech"
    GESTURE = "gesture"
    VISION = "vision"
    TOUCH = "touch"
    AUDIO = "audio"

@dataclass
class MultiModalInput:
    """Represents input from a single modality"""
    modality: ModalityType
    data: Any
    timestamp: float
    confidence: float = 1.0
    source_id: str = ""

@dataclass
class FusedInput:
    """Represents fused multi-modal input"""
    fused_data: Dict[str, Any]
    timestamp: float
    confidence: float
    contributing_modalities: List[ModalityType]
    fusion_strategy: str

class MultiModalInputProcessor:
    def __init__(self):
        # Input queues for each modality
        self.input_queues = {
            ModalityType.SPEECH: queue.Queue(),
            ModalityType.GESTURE: queue.Queue(),
            ModalityType.VISION: queue.Queue(),
            ModalityType.TOUCH: queue.Queue(),
            ModalityType.AUDIO: queue.Queue()
        }
        
        # Fused output queue
        self.fused_queue = queue.Queue()
        
        # Processing threads
        self.processing_threads = []
        self.is_running = False
        
        # Modality synchronizer
        self.synchronizer = ModalitySynchronizer()
        
        # Fusion engine
        self.fusion_engine = MultiModalFusionEngine()
    
    def start_processing(self):
        """Start multi-modal processing"""
        self.is_running = True
        
        # Start input processing thread
        input_thread = threading.Thread(target=self._process_inputs)
        input_thread.daemon = True
        input_thread.start()
        self.processing_threads.append(input_thread)
        
        print("Multi-modal input processor started")
    
    def submit_input(self, modality: ModalityType, data: Any, confidence: float = 1.0, source_id: str = ""):
        """Submit input from a specific modality"""
        input_obj = MultiModalInput(
            modality=modality,
            data=data,
            timestamp=time.time(),
            confidence=confidence,
            source_id=source_id
        )
        
        if modality in self.input_queues:
            self.input_queues[modality].put_nowait(input_obj)
        else:
            print(f"Unknown modality: {modality}")
    
    def get_fused_input(self, timeout: float = 0.1) -> Optional[FusedInput]:
        """Get fused multi-modal input"""
        try:
            return self.fused_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def _process_inputs(self):
        """Process inputs from all modalities"""
        while self.is_running:
            try:
                # Collect inputs from all modalities
                collected_inputs = {}
                
                for modality, q in self.input_queues.items():
                    try:
                        while True:  # Get all available inputs
                            input_obj = q.get_nowait()
                            if modality not in collected_inputs:
                                collected_inputs[modality] = []
                            collected_inputs[modality].append(input_obj)
                    except queue.Empty:
                        continue
                
                # Synchronize collected inputs
                synchronized_inputs = self.synchronizer.synchronize(collected_inputs)
                
                # Fuse synchronized inputs
                if synchronized_inputs:
                    fused_input = self.fusion_engine.fuse(synchronized_inputs)
                    if fused_input:
                        self.fused_queue.put_nowait(fused_input)
                
                # Small delay to prevent busy waiting
                time.sleep(0.01)
                
            except Exception as e:
                print(f"Multi-modal processing error: {e}")
                time.sleep(0.1)
    
    def stop_processing(self):
        """Stop multi-modal processing"""
        self.is_running = False
        
        # Wait for threads to finish
        for thread in self.processing_threads:
            thread.join(timeout=1.0)
        
        print("Multi-modal input processor stopped")

class ModalitySynchronizer:
    def __init__(self, sync_window: float = 0.5):
        self.sync_window = sync_window  # Window for synchronization (seconds)
        self.buffer = {}
        self.modality_weights = {
            ModalityType.SPEECH: 0.4,
            ModalityType.GESTURE: 0.3,
            ModalityType.VISION: 0.3
        }
    
    def synchronize(self, inputs: Dict[ModalityType, List[MultiModalInput]]) -> List[Dict[ModalityType, MultiModalInput]]:
        """Synchronize inputs from different modalities"""
        # For now, return all combinations within sync window
        # In a real system, this would implement more sophisticated synchronization
        
        synchronized_groups = []
        
        # Simple temporal grouping
        current_time = time.time()
        
        # Group inputs by time windows
        time_windows = {}
        for modality, input_list in inputs.items():
            for input_obj in input_list:
                window_id = int(input_obj.timestamp / self.sync_window)
                if window_id not in time_windows:
                    time_windows[window_id] = {}
                time_windows[window_id][modality] = input_obj
        
        # Create synchronized groups
        for window_id, window_inputs in time_windows.items():
            if len(window_inputs) >= 1:  # At least one modality present
                synchronized_groups.append(window_inputs)
        
        return synchronized_groups

class MultiModalFusionEngine:
    def __init__(self):
        self.fusion_strategies = {
            'early': self._early_fusion,
            'late': self._late_fusion,
            'intermediate': self._intermediate_fusion
        }
        self.default_strategy = 'intermediate'
    
    def fuse(self, synchronized_inputs: List[Dict[ModalityType, MultiModalInput]]) -> Optional[FusedInput]:
        """Fuse synchronized multi-modal inputs"""
        if not synchronized_inputs:
            return None
        
        # Use the most recent synchronized group
        latest_group = synchronized_inputs[-1]
        
        # Apply fusion strategy
        fused_data, confidence = self.fusion_strategies[self.default_strategy](latest_group)
        
        if fused_data:
            return FusedInput(
                fused_data=fused_data,
                timestamp=time.time(),
                confidence=confidence,
                contributing_modalities=list(latest_group.keys()),
                fusion_strategy=self.default_strategy
            )
        
        return None
    
    def _early_fusion(self, inputs: Dict[ModalityType, MultiModalInput]) -> tuple:
        """Early fusion: combine raw features before processing"""
        fused_data = {}
        total_confidence = 0
        count = 0
        
        for modality, input_obj in inputs.items():
            fused_data[f"{modality.value}_raw"] = input_obj.data
            total_confidence += input_obj.confidence
            count += 1
        
        avg_confidence = total_confidence / count if count > 0 else 0.5
        
        return fused_data, avg_confidence
    
    def _late_fusion(self, inputs: Dict[ModalityType, MultiModalInput]) -> tuple:
        """Late fusion: combine processed results"""
        fused_data = {'processed_results': {}}
        total_confidence = 0
        count = 0
        
        for modality, input_obj in inputs.items():
            # In a real system, this would process the input first
            processed_result = self._process_modality_input(modality, input_obj)
            fused_data['processed_results'][modality.value] = processed_result
            total_confidence += input_obj.confidence
            count += 1
        
        avg_confidence = total_confidence / count if count > 0 else 0.5
        
        return fused_data, avg_confidence
    
    def _intermediate_fusion(self, inputs: Dict[ModalityType, MultiModalInput]) -> tuple:
        """Intermediate fusion: partial processing before combination"""
        fused_data = {}
        total_confidence = 0
        count = 0
        
        for modality, input_obj in inputs.items():
            # Partial processing
            if modality == ModalityType.SPEECH:
                processed = self._process_speech_input(input_obj)
            elif modality == ModalityType.GESTURE:
                processed = self._process_gesture_input(input_obj)
            elif modality == ModalityType.VISION:
                processed = self._process_vision_input(input_obj)
            else:
                processed = input_obj.data
            
            fused_data[modality.value] = processed
            total_confidence += input_obj.confidence * self._get_modality_weight(modality)
            count += 1
        
        avg_confidence = total_confidence / count if count > 0 else 0.5
        
        return fused_data, avg_confidence
    
    def _process_modality_input(self, modality: ModalityType, input_obj: MultiModalInput) -> Any:
        """Process input from a specific modality"""
        # This would contain modality-specific processing
        return input_obj.data
    
    def _process_speech_input(self, input_obj: MultiModalInput) -> Dict[str, Any]:
        """Process speech input"""
        # In a real system, this would run NLP processing
        return {
            'transcript': input_obj.data,
            'intent': 'unknown',  # Would be extracted by NLP
            'entities': []  # Would be extracted by NLP
        }
    
    def _process_gesture_input(self, input_obj: MultiModalInput) -> Dict[str, Any]:
        """Process gesture input"""
        # In a real system, this would analyze gesture features
        return {
            'gesture_type': input_obj.data.get('type', 'unknown'),
            'direction': input_obj.data.get('direction', [0, 0, 0]),
            'confidence': input_obj.confidence
        }
    
    def _process_vision_input(self, input_obj: MultiModalInput) -> Dict[str, Any]:
        """Process vision input"""
        # In a real system, this would analyze visual features
        return {
            'detected_objects': input_obj.data.get('objects', []),
            'people_present': input_obj.data.get('people', []),
            'scene_description': input_obj.data.get('scene', '')
        }
    
    def _get_modality_weight(self, modality: ModalityType) -> float:
        """Get weight for a specific modality"""
        weights = {
            ModalityType.SPEECH: 1.0,
            ModalityType.GESTURE: 0.8,
            ModalityType.VISION: 0.7,
            ModalityType.TOUCH: 0.9,
            ModalityType.AUDIO: 0.6
        }
        return weights.get(modality, 0.5)
```

## Attention and Gaze Systems

### Visual Attention System

```python
class VisualAttentionSystem:
    def __init__(self, attention_span: float = 3.0):
        self.attention_span = attention_span  # How long to maintain attention (seconds)
        self.attended_objects = {}
        self.attended_people = {}
        self.focus_object = None
        self.focus_person = None
        self.last_attention_update = time.time()
        
        # Attention priorities
        self.attention_priorities = {
            'person': 1.0,
            'moving_object': 0.9,
            'interactable_object': 0.8,
            'salient_object': 0.7,
            'static_object': 0.5
        }
    
    def update_attention(self, vision_data: Dict[str, Any]):
        """Update attention based on visual input"""
        current_time = time.time()
        
        # Update object attention
        if 'objects' in vision_data:
            self._update_object_attention(vision_data['objects'], current_time)
        
        # Update person attention
        if 'people' in vision_data:
            self._update_person_attention(vision_data['people'], current_time)
        
        # Determine primary focus
        self._determine_primary_focus(current_time)
        
        self.last_attention_update = current_time
    
    def _update_object_attention(self, objects: List[Dict], current_time: float):
        """Update attention for detected objects"""
        for obj in objects:
            obj_id = obj.get('id', obj.get('name', 'unknown'))
            
            # Calculate attention score
            attention_score = self._calculate_object_attention_score(obj)
            
            # Update or add object to attention list
            if obj_id in self.attended_objects:
                # Update existing object
                self.attended_objects[obj_id]['last_seen'] = current_time
                self.attended_objects[obj_id]['attention_score'] = attention_score
                self.attended_objects[obj_id]['position'] = obj.get('position', [0, 0, 0])
            else:
                # Add new object
                self.attended_objects[obj_id] = {
                    'last_seen': current_time,
                    'attention_score': attention_score,
                    'position': obj.get('position', [0, 0, 0]),
                    'properties': obj.get('properties', {}),
                    'is_interactable': obj.get('interactable', False)
                }
        
        # Remove objects not seen for a while
        self._cleanup_old_objects(current_time)
    
    def _update_person_attention(self, people: List[Dict], current_time: float):
        """Update attention for detected people"""
        for person in people:
            person_id = person.get('id', person.get('name', 'unknown_person'))
            
            # Calculate attention score
            attention_score = self._calculate_person_attention_score(person)
            
            # Update or add person to attention list
            if person_id in self.attended_people:
                self.attended_people[person_id]['last_seen'] = current_time
                self.attended_people[person_id]['attention_score'] = attention_score
                self.attended_people[person_id]['position'] = person.get('position', [0, 0, 0])
            else:
                self.attended_people[person_id] = {
                    'last_seen': current_time,
                    'attention_score': attention_score,
                    'position': person.get('position', [0, 0, 0]),
                    'is_interacting': person.get('is_interacting', False),
                    'gaze_direction': person.get('gaze_direction', [0, 0, 1])
                }
        
        # Remove people not seen for a while
        self._cleanup_old_people(current_time)
    
    def _calculate_object_attention_score(self, obj: Dict) -> float:
        """Calculate attention score for an object"""
        score = 0.0
        
        # Size matters (larger objects get more attention)
        size = obj.get('size', 0.1)
        score += min(size, 1.0) * 0.3
        
        # Movement (moving objects get more attention)
        is_moving = obj.get('is_moving', False)
        if is_moving:
            score += 0.4
        
        # Interactability (interactable objects get more attention)
        is_interactable = obj.get('interactable', False)
        if is_interactable:
            score += 0.3
        
        # Novelty (newly detected objects get temporary boost)
        # This would be handled by tracking when object was first detected
        
        return min(score, 1.0)
    
    def _calculate_person_attention_score(self, person: Dict) -> float:
        """Calculate attention score for a person"""
        score = 0.0
        
        # Is person interacting with robot
        is_interacting = person.get('is_interacting', False)
        if is_interacting:
            score += 0.5
        
        # Proximity (closer people get more attention)
        distance = person.get('distance', float('inf'))
        if distance < float('inf'):
            proximity_score = max(0, 1 - (distance / 5.0))  # Normalized to 5m
            score += proximity_score * 0.3
        
        # Gaze direction (people looking at robot get more attention)
        is_looking_at_robot = person.get('looking_at_robot', False)
        if is_looking_at_robot:
            score += 0.2
        
        return min(score, 1.0)
    
    def _determine_primary_focus(self, current_time: float):
        """Determine which object/person to focus on"""
        # Clean up old entries
        self._cleanup_old_objects(current_time)
        self._cleanup_old_people(current_time)
        
        # Find highest scoring object
        best_object_score = 0
        best_object_id = None
        for obj_id, obj_data in self.attended_objects.items():
            if obj_data['attention_score'] > best_object_score:
                best_object_score = obj_data['attention_score']
                best_object_id = obj_id
        
        # Find highest scoring person
        best_person_score = 0
        best_person_id = None
        for person_id, person_data in self.attended_people.items():
            if person_data['attention_score'] > best_person_score:
                best_person_score = person_data['attention_score']
                best_person_id = person_id
        
        # Determine primary focus based on scores and context
        if best_person_score >= best_object_score and best_person_id:
            self.focus_person = best_person_id
            self.focus_object = None
        elif best_object_score > best_person_score and best_object_id:
            self.focus_object = best_object_id
            self.focus_person = None
        else:
            # No strong focus, maintain current or clear
            if not self.focus_person and not self.focus_object:
                # Clear focus
                self.focus_person = None
                self.focus_object = None
    
    def _cleanup_old_objects(self, current_time: float):
        """Remove objects not seen for attention_span seconds"""
        old_objects = []
        for obj_id, obj_data in self.attended_objects.items():
            if current_time - obj_data['last_seen'] > self.attention_span:
                old_objects.append(obj_id)
        
        for obj_id in old_objects:
            del self.attended_objects[obj_id]
    
    def _cleanup_old_people(self, current_time: float):
        """Remove people not seen for attention_span seconds"""
        old_people = []
        for person_id, person_data in self.attended_people.items():
            if current_time - person_data['last_seen'] > self.attention_span:
                old_people.append(person_id)
        
        for person_id in old_people:
            del self.attended_people[person_id]
    
    def get_attention_status(self) -> Dict[str, Any]:
        """Get current attention status"""
        return {
            'focused_object': self.focus_object,
            'focused_person': self.focus_person,
            'attended_objects': list(self.attended_objects.keys()),
            'attended_people': list(self.attended_people.keys()),
            'object_attention_scores': {k: v['attention_score'] for k, v in self.attended_objects.items()},
            'person_attention_scores': {k: v['attention_score'] for k, v in self.attended_people.items()}
        }
    
    def look_at_focused_object(self):
        """Generate command to look at focused object"""
        if self.focus_object and self.focus_object in self.attended_objects:
            obj_pos = self.attended_objects[self.focus_object]['position']
            return {
                'command': 'look_at',
                'target': 'object',
                'position': obj_pos,
                'object_id': self.focus_object
            }
        return None
    
    def look_at_focused_person(self):
        """Generate command to look at focused person"""
        if self.focus_person and self.focus_person in self.attended_people:
            person_pos = self.attended_people[self.focus_person]['position']
            return {
                'command': 'look_at',
                'target': 'person',
                'position': person_pos,
                'person_id': self.focus_person
            }
        return None
```

### Gaze and Head Control System

```python
class GazeControlSystem:
    def __init__(self, visual_attention_system: VisualAttentionSystem):
        self.attention_system = visual_attention_system
        self.gaze_target = None
        self.gaze_smoothness = 0.1  # For smooth transitions
        self.gaze_priority = 'social'  # 'social', 'task', 'exploratory'
        self.is_tracking = False
        
        # Gaze patterns
        self.gaze_patterns = {
            'social': self._social_gaze_pattern,
            'task': self._task_gaze_pattern,
            'exploratory': self._exploratory_gaze_pattern
        }
    
    def update_gaze(self):
        """Update gaze based on attention system"""
        attention_status = self.attention_system.get_attention_status()
        
        # Determine gaze target based on priority and focus
        if self.gaze_priority == 'social':
            gaze_cmd = self._determine_social_gaze(attention_status)
        elif self.gaze_priority == 'task':
            gaze_cmd = self._determine_task_gaze(attention_status)
        else:  # exploratory
            gaze_cmd = self._determine_exploratory_gaze(attention_status)
        
        if gaze_cmd:
            self.gaze_target = gaze_cmd['position']
            return self._execute_gaze_command(gaze_cmd)
        
        return False
    
    def _determine_social_gaze(self, attention_status: Dict) -> Optional[Dict]:
        """Determine social gaze target"""
        if attention_status['focused_person']:
            return self.attention_system.look_at_focused_person()
        elif attention_status['focused_object']:
            return self.attention_system.look_at_focused_object()
        else:
            # Look ahead or at neutral position
            return {
                'command': 'look_at',
                'target': 'neutral',
                'position': [2, 0, 1.5],  # Ahead at eye level
                'description': 'Looking ahead socially'
            }
    
    def _determine_task_gaze(self, attention_status: Dict) -> Optional[Dict]:
        """Determine task-oriented gaze target"""
        if attention_status['focused_object']:
            obj_id = attention_status['focused_object']
            obj_data = self.attention_system.attended_objects[obj_id]
            
            if obj_data.get('is_interactable', False):
                return {
                    'command': 'look_at',
                    'target': 'object',
                    'position': obj_data['position'],
                    'description': f'Focusing on interactable object {obj_id}'
                }
        
        # If no specific task object, look at hands/manipulation area
        return {
            'command': 'look_at',
            'target': 'manipulation_area',
            'position': [0.5, 0, 0.8],  # Typical reach area
            'description': 'Looking at manipulation area'
        }
    
    def _determine_exploratory_gaze(self, attention_status: Dict) -> Optional[Dict]:
        """Determine exploratory gaze target"""
        # Look at most salient object
        if attention_status['attended_objects']:
            # Find object with highest attention score
            best_obj = max(
                self.attention_system.attended_objects.items(),
                key=lambda x: x[1]['attention_score']
            )
            obj_id, obj_data = best_obj
            
            return {
                'command': 'look_at',
                'target': 'object',
                'position': obj_data['position'],
                'description': f'Exploring object {obj_id}'
            }
        
        # If no objects, look around
        import random
        look_around_positions = [
            [3, -1, 1.5],  # Left
            [3, 1, 1.5],   # Right
            [2, 0, 2],     # Up
            [2, 0, 1]      # Down
        ]
        
        target_pos = random.choice(look_around_positions)
        
        return {
            'command': 'look_at',
            'target': 'exploration',
            'position': target_pos,
            'description': 'Exploring environment'
        }
    
    def _execute_gaze_command(self, command: Dict) -> bool:
        """Execute gaze command (in simulation, would control actual servos)"""
        target_pos = command['position']
        
        print(f"Gaze system: Looking at {command['target']} at position {target_pos}")
        
        # In a real robot, this would:
        # 1. Calculate joint angles for head/neck servos
        # 2. Send commands to servo controllers
        # 3. Verify execution
        # 4. Handle smooth transitions
        
        # Simulate gaze movement
        self._simulate_gaze_movement(target_pos)
        
        return True
    
    def _simulate_gaze_movement(self, target_pos: List[float]):
        """Simulate gaze movement for demonstration"""
        print(f"  (Simulated) Moving gaze to {target_pos}")
        # In real implementation, this would control actual servos
        
        # For demonstration, just print the movement
        current_pos = [1, 0, 1.5]  # Default position
        print(f"  From {current_pos} to {target_pos}")
    
    def set_gaze_priority(self, priority: str):
        """Set gaze priority mode"""
        if priority in self.gaze_patterns:
            self.gaze_priority = priority
            print(f"Gaze priority set to: {priority}")
        else:
            print(f"Invalid gaze priority: {priority}. Valid options: {list(self.gaze_patterns.keys())}")
    
    def start_tracking(self):
        """Start continuous gaze tracking"""
        self.is_tracking = True
        print("Gaze tracking started")
    
    def stop_tracking(self):
        """Stop gaze tracking"""
        self.is_tracking = False
        print("Gaze tracking stopped")
```

## Gesture Recognition and Interpretation

### Gesture Recognition System

```python
import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, List, Dict, Optional

class GestureRecognitionSystem:
    def __init__(self):
        # Initialize MediaPipe for hand tracking
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Gesture vocabulary
        self.gesture_vocabulary = {
            'wave': self._is_wave_gesture,
            'point': self._is_pointing_gesture,
            'beckon': self._is_beckon_gesture,
            'stop': self._is_stop_gesture,
            'thumbs_up': self._is_thumbs_up,
            'peace_sign': self._is_peace_sign,
            'okay': self._is_okay_sign,
            'fist_bump': self._is_fist_bump,
            'clap': self._is_clap,
            'heart': self._is_heart_gesture
        }
        
        # Gesture history for temporal patterns
        self.gesture_history = []
        self.max_history = 10
    
    def recognize_gestures(self, image: np.ndarray) -> List[Dict]:
        """Recognize gestures in an image"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image for hands
        hand_results = self.hands.process(image_rgb)
        # Process image for body pose
        pose_results = self.pose.process(image_rgb)
        
        recognized_gestures = []
        
        # Process hand gestures
        if hand_results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                
                # Recognize hand gestures
                hand_gestures = self._recognize_hand_gestures(hand_landmarks, i)
                recognized_gestures.extend(hand_gestures)
        
        # Process body gestures
        if pose_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )
            
            # Recognize body gestures
            body_gestures = self._recognize_body_gestures(pose_results.pose_landmarks)
            recognized_gestures.extend(body_gestures)
        
        # Update gesture history
        self._update_gesture_history(recognized_gestures)
        
        return recognized_gestures
    
    def _recognize_hand_gestures(self, landmarks, hand_index) -> List[Dict]:
        """Recognize hand gestures from landmarks"""
        gestures = []
        
        for gesture_name, gesture_func in self.gesture_vocabulary.items():
            if gesture_func(landmarks):
                gesture_data = {
                    'type': 'hand',
                    'name': gesture_name,
                    'hand_index': hand_index,
                    'confidence': 0.8,  # Would be calculated in real system
                    'timestamp': time.time(),
                    'landmarks': [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]
                }
                gestures.append(gesture_data)
        
        return gestures
    
    def _recognize_body_gestures(self, pose_landmarks) -> List[Dict]:
        """Recognize body/posture gestures"""
        gestures = []
        
        # Example: Raise both arms (like cheering)
        left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_elbow = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
        
        # Check if both arms are raised above shoulder level
        if (left_elbow.y < left_shoulder.y and right_elbow.y < right_shoulder.y):
            gesture_data = {
                'type': 'body',
                'name': 'cheering',
                'confidence': 0.7,
                'timestamp': time.time()
            }
            gestures.append(gesture_data)
        
        return gestures
    
    def _is_wave_gesture(self, landmarks) -> bool:
        """Detect wave gesture (fingers extended, thumb tucked, side-to-side motion)"""
        # Check if fingers are extended and thumb is tucked
        return (self._are_fingers_extended(landmarks, [8, 12, 16, 20]) and  # Index, middle, ring, pinky extended
                not self._is_finger_extended(landmarks, 4))  # Thumb not extended
    
    def _is_pointing_gesture(self, landmarks) -> bool:
        """Detect pointing gesture (index finger extended, others curled)"""
        return (self._is_finger_extended(landmarks, 8) and  # Index finger extended
                self._are_other_fingers_curled(landmarks, [8]))  # Others curled
    
    def _is_beckon_gesture(self, landmarks) -> bool:
        """Detect beckon gesture (index/middle fingers curled like beckoning)"""
        return (self._is_finger_extended(landmarks, 4) and  # Thumb extended
                self._is_finger_curled(landmarks, 8) and  # Index finger curled
                self._is_finger_curled(landmarks, 12))  # Middle finger curled
    
    def _is_stop_gesture(self, landmarks) -> bool:
        """Detect stop gesture (palm facing forward, fingers extended)"""
        # This would require checking palm orientation
        return (self._are_fingers_extended(landmarks, [8, 12, 16, 20]) and  # All fingers extended
                self._is_palm_facing_observer(landmarks))
    
    def _is_thumbs_up(self, landmarks) -> bool:
        """Detect thumbs up gesture"""
        return (self._is_finger_extended(landmarks, 4) and  # Thumb extended
                self._are_other_fingers_curled(landmarks, [4]))  # Others curled
    
    def _is_peace_sign(self, landmarks) -> bool:
        """Detect peace sign (index and middle fingers extended, others curled)"""
        return (self._is_finger_extended(landmarks, 8) and  # Index finger extended
                self._is_finger_extended(landmarks, 12) and  # Middle finger extended
                self._are_other_fingers_curled(landmarks, [8, 12]))  # Others curled
    
    def _is_okay_sign(self, landmarks) -> bool:
        """Detect okay sign (thumb and index finger forming circle)"""
        # Check if thumb tip is near index finger tip
        thumb_tip = landmarks.landmark[4]
        index_tip = landmarks.landmark[8]
        
        distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + 
                          (thumb_tip.y - index_tip.y)**2 + 
                          (thumb_tip.z - index_tip.z)**2)
        
        return distance < 0.05 and self._are_other_fingers_extended(landmarks, [4, 8])
    
    def _is_fist_bump(self, landmarks) -> bool:
        """Detect fist bump gesture (closed fists)"""
        return all(not self._is_finger_extended(landmarks, fid) 
                  for fid in [8, 12, 16, 20])  # All fingers curled
    
    def _is_clap(self, landmarks) -> bool:
        """Detect clap gesture (both hands together)"""
        # This would require detecting both hands and their relative positions
        # Implemented in multi-hand context
        return False  # Placeholder
    
    def _is_heart_gesture(self, landmarks) -> bool:
        """Detect heart gesture (both hands forming heart shape)"""
        # This would require detecting both hands
        return False  # Placeholder
    
    def _is_finger_extended(self, landmarks, finger_tip_id) -> bool:
        """Check if a specific finger is extended"""
        # Compare finger tip to base position
        finger_tip = landmarks.landmark[finger_tip_id]
        finger_base = landmarks.landmark[finger_tip_id - 3]  # MCP joint
        
        # If tip is significantly higher than base (for upright hand)
        return (finger_tip.y < finger_base.y - 0.1)  # Adjust threshold as needed
    
    def _is_finger_curled(self, landmarks, finger_tip_id) -> bool:
        """Check if a specific finger is curled"""
        return not self._is_finger_extended(landmarks, finger_tip_id)
    
    def _are_fingers_extended(self, landmarks, finger_tip_ids) -> bool:
        """Check if multiple fingers are extended"""
        return all(self._is_finger_extended(landmarks, fid) for fid in finger_tip_ids)
    
    def _are_other_fingers_curled(self, landmarks, extended_finger_ids) -> bool:
        """Check if all fingers except specified ones are curled"""
        all_finger_ids = [8, 12, 16, 20]  # Index, middle, ring, pinky
        other_finger_ids = [fid for fid in all_finger_ids if fid not in extended_finger_ids]
        return all(self._is_finger_curled(landmarks, fid) for fid in other_finger_ids)
    
    def _are_other_fingers_extended(self, landmarks, curled_finger_ids) -> bool:
        """Check if all fingers except specified ones are extended"""
        all_finger_ids = [8, 12, 16, 20]  # Index, middle, ring, pinky
        other_finger_ids = [fid for fid in all_finger_ids if fid not in curled_finger_ids]
        return all(self._is_finger_extended(landmarks, fid) for fid in other_finger_ids)
    
    def _is_palm_facing_observer(self, landmarks) -> bool:
        """Check if palm is facing the observer"""
        # Compare wrist to finger base positions
        wrist = landmarks.landmark[0]
        middle_mcp = landmarks.landmark[9]  # Middle finger MCP
        
        # Palm facing observer when wrist is behind (higher y-value) than finger base
        return wrist.y > middle_mcp.y
    
    def _update_gesture_history(self, new_gestures: List[Dict]):
        """Update gesture history for temporal pattern recognition"""
        for gesture in new_gestures:
            self.gesture_history.append(gesture)
        
        # Keep history within limits
        if len(self.gesture_history) > self.max_history:
            self.gesture_history = self.gesture_history[-self.max_history:]
    
    def get_temporal_gesture_pattern(self) -> List[Dict]:
        """Get recent gesture pattern for temporal analysis"""
        return self.gesture_history.copy()
    
    def interpret_gesture_sequence(self, sequence: List[Dict]) -> Dict:
        """Interpret a sequence of gestures"""
        if not sequence:
            return {'interpretation': 'no_gesture', 'meaning': 'No gesture detected'}
        
        # Simple interpretations
        if len(sequence) == 1:
            gesture = sequence[0]
            if gesture['name'] == 'wave':
                return {'interpretation': 'greeting', 'meaning': 'User is greeting'}
            elif gesture['name'] == 'thumbs_up':
                return {'interpretation': 'approval', 'meaning': 'User approves'}
            elif gesture['name'] == 'point':
                return {'interpretation': 'directive', 'meaning': 'User is directing attention'}
            elif gesture['name'] == 'stop':
                return {'interpretation': 'halt', 'meaning': 'User wants robot to stop'}
        
        # Multi-gesture patterns
        gesture_names = [g['name'] for g in sequence]
        
        if 'wave' in gesture_names and 'point' in gesture_names:
            return {'interpretation': 'guided_greeting', 'meaning': 'User greeting and directing'}
        
        # Default interpretation
        return {
            'interpretation': 'multiple_gestures',
            'meaning': f"Sequence of {len(sequence)} gestures detected",
            'gesture_sequence': gesture_names
        }
```

## Multi-Modal Fusion and Coordination

### High-Level Fusion System

```python
class MultiModalFusionSystem:
    def __init__(self):
        # Initialize subsystems
        self.visual_attention = VisualAttentionSystem()
        self.gaze_control = GazeControlSystem(self.visual_attention)
        self.gesture_recognition = GestureRecognitionSystem()
        
        # Context manager
        self.context_manager = InteractionContextManager()
        
        # Response generator
        self.response_generator = MultiModalResponseGenerator()
        
        # State tracking
        self.current_interaction_partner = None
        self.interaction_history = []
        self.max_history = 20
    
    def process_multi_modal_input(self, 
                                  speech_data: Optional[Dict] = None,
                                  vision_data: Optional[Dict] = None,
                                  gesture_data: Optional[List[Dict]] = None,
                                  audio_data: Optional[Dict] = None) -> Dict:
        """Process multi-modal input and generate coordinated response"""
        
        # Update visual attention system
        if vision_data:
            self.visual_attention.update_attention(vision_data)
        
        # Update gaze system
        self.gaze_control.update_gaze()
        
        # Process gesture data
        processed_gestures = gesture_data or []
        if vision_data and 'image' in vision_data:
            # Perform gesture recognition on image
            recognized_gestures = self.gesture_recognition.recognize_gestures(vision_data['image'])
            processed_gestures.extend(recognized_gestures)
        
        # Update context
        context_update = {
            'speech': speech_data,
            'vision': vision_data,
            'gestures': processed_gestures,
            'audio': audio_data,
            'attention_status': self.visual_attention.get_attention_status(),
            'timestamp': time.time()
        }
        self.context_manager.update_context(context_update)
        
        # Determine interaction partner
        self._update_interaction_partner(processed_gestures, vision_data)
        
        # Generate response
        response = self.response_generator.generate_response(
            speech_data, 
            processed_gestures, 
            self.context_manager.get_context()
        )
        
        # Update history
        interaction_record = {
            'input': context_update,
            'response': response,
            'timestamp': time.time()
        }
        self.interaction_history.append(interaction_record)
        
        if len(self.interaction_history) > self.max_history:
            self.interaction_history = self.interaction_history[-self.max_history:]
        
        return {
            'response': response,
            'gaze_command': self.gaze_control.update_gaze(),
            'context': self.context_manager.get_context(),
            'attention_status': self.visual_attention.get_attention_status()
        }
    
    def _update_interaction_partner(self, gestures: List[Dict], vision_data: Optional[Dict]):
        """Update the current interaction partner based on modalities"""
        # Priority: gestures -> gaze direction -> proximity
        
        # Check if someone is gesturing toward the robot
        for gesture in gestures:
            if gesture.get('type') == 'hand':
                # In a real system, we'd check if gesture is directed at robot
                # For now, assume if person is attending to robot, they're the partner
                pass
        
        # Update based on visual attention
        attention_status = self.visual_attention.get_attention_status()
        if attention_status['focused_person']:
            self.current_interaction_partner = attention_status['focused_person']
    
    def get_interaction_state(self) -> Dict:
        """Get current interaction state"""
        return {
            'current_partner': self.current_interaction_partner,
            'attention_status': self.visual_attention.get_attention_status(),
            'gaze_target': self.gaze_control.gaze_target,
            'recent_gestures': self.gesture_recognition.get_temporal_gesture_pattern(),
            'context': self.context_manager.get_context()
        }

class InteractionContextManager:
    def __init__(self):
        self.context = {
            'conversation_history': [],
            'user_preferences': {},
            'environment_state': {},
            'task_context': {},
            'social_context': {},
            'temporal_context': {}
        }
        self.max_conversation_length = 10
    
    def update_context(self, new_input: Dict):
        """Update interaction context with new input"""
        timestamp = time.time()
        
        # Update conversation history
        if new_input.get('speech'):
            self.context['conversation_history'].append({
                'type': 'speech',
                'content': new_input['speech'],
                'timestamp': timestamp
            })
        
        # Update environment state if vision data available
        if new_input.get('vision'):
            self.context['environment_state'] = new_input['vision']
        
        # Update temporal context
        self.context['temporal_context'] = {
            'last_update': timestamp,
            'session_duration': timestamp - self.context.get('session_start', timestamp),
            'interaction_count': self.context.get('interaction_count', 0) + 1
        }
        
        # Keep conversation history within limits
        if len(self.context['conversation_history']) > self.max_conversation_length:
            self.context['conversation_history'] = self.context['conversation_history'][-self.max_conversation_length:]
    
    def get_context(self) -> Dict:
        """Get current context"""
        return self.context.copy()
    
    def update_user_preference(self, user_id: str, preference: str, value: Any):
        """Update user preference"""
        if user_id not in self.context['user_preferences']:
            self.context['user_preferences'][user_id] = {}
        self.context['user_preferences'][user_id][preference] = value

class MultiModalResponseGenerator:
    def __init__(self):
        self.response_templates = {
            'greeting': [
                "Hello! I see you greeting me.",
                "Hi there! Nice to see you waving.",
                "Hello! How can I help you today?"
            ],
            'directive': [
                "I see you're pointing. How can I help?",
                "You seem to be directing my attention. What do you need?",
                "I notice you're pointing. Can you tell me more?"
            ],
            'acknowledgment': [
                "I understand.",
                "Got it, thank you for letting me know.",
                "Thanks for the information."
            ],
            'question': [
                "That's a good question. Let me think...",
                "I'm not sure I understand. Could you clarify?",
                "Interesting question! I'll do my best to help."
            ]
        }
    
    def generate_response(self, speech_data: Optional[Dict], 
                         gestures: List[Dict], 
                         context: Dict) -> Dict:
        """Generate multi-modal response"""
        
        response_parts = []
        
        # Process speech
        if speech_data:
            speech_response = self._generate_speech_response(speech_data)
            response_parts.append(speech_response)
        
        # Process gestures
        if gestures:
            gesture_response = self._generate_gesture_response(gestures)
            response_parts.append(gesture_response)
        
        # Generate coordinated response
        coordinated_response = self._coordinate_response(response_parts, context)
        
        return coordinated_response
    
    def _generate_speech_response(self, speech_data: Dict) -> Dict:
        """Generate response to speech input"""
        text = speech_data.get('transcript', '').lower()
        
        if any(word in text for word in ['hello', 'hi', 'hey']):
            response_type = 'greeting'
        elif any(word in text for word in ['what', 'how', 'where', 'when', 'who', 'why']):
            response_type = 'question'
        else:
            response_type = 'acknowledgment'
        
        import random
        response_text = random.choice(self.response_templates[response_type])
        
        return {
            'modality': 'speech',
            'type': response_type,
            'content': response_text,
            'confidence': 0.9
        }
    
    def _generate_gesture_response(self, gestures: List[Dict]) -> Dict:
        """Generate response to gesture input"""
        if not gestures:
            return {'modality': 'gesture', 'type': 'none', 'content': '', 'confidence': 0.0}
        
        # Take the first gesture for simplicity
        gesture = gestures[0]
        
        if gesture['name'] == 'wave':
            response_type = 'greeting'
            content = "Hello back! I see you waving."
        elif gesture['name'] == 'point':
            response_type = 'directive'
            content = "I see you're pointing. How can I help?"
        elif gesture['name'] == 'thumbs_up':
            response_type = 'acknowledgment'
            content = "Thank you for the thumbs up!"
        else:
            response_type = 'acknowledgment'
            content = f"I noticed your {gesture['name']} gesture."
        
        return {
            'modality': 'gesture',
            'type': response_type,
            'content': content,
            'confidence': gesture.get('confidence', 0.8)
        }
    
    def _coordinate_response(self, response_parts: List[Dict], context: Dict) -> Dict:
        """Coordinate responses from multiple modalities"""
        if not response_parts:
            return {
                'speech': "I'm here and ready to help.",
                'actions': ['maintain_eye_contact'],
                'expressions': ['neutral_attention']
            }
        
        # Combine responses
        speech_responses = [rp for rp in response_parts if rp['modality'] == 'speech']
        gesture_responses = [rp for rp in response_parts if rp['modality'] == 'gesture']
        
        # Create combined response
        combined_speech = []
        for resp in speech_responses:
            combined_speech.append(resp['content'])
        for resp in gesture_responses:
            combined_speech.append(resp['content'])
        
        # Determine actions based on context
        actions = []
        if gesture_responses:
            actions.append('acknowledge_gesture')
        if speech_responses:
            actions.append('respond_verbally')
        
        # Add gaze actions
        attention_status = context.get('attention_status', {})
        if attention_status.get('focused_person'):
            actions.append('maintain_eye_contact')
        elif attention_status.get('focused_object'):
            actions.append('attend_to_object')
        
        return {
            'speech': " ".join(combined_speech) if combined_speech else "I acknowledge your input.",
            'actions': actions,
            'expressions': ['attentive']  # Facial expression
        }
```

## Practical Exercise: Multi-Modal Interaction System

Create a complete multi-modal interaction system:

1. **Implement visual attention system**
2. **Add gesture recognition capabilities**
3. **Create gaze control system**
4. **Build multi-modal fusion engine**
5. **Test with various interaction scenarios**

### Complete Multi-Modal System Example

```python
class CompleteMultiModalSystem:
    def __init__(self):
        # Initialize all components
        self.visual_attention = VisualAttentionSystem()
        self.gaze_control = GazeControlSystem(self.visual_attention)
        self.gesture_recognition = GestureRecognitionSystem()
        self.fusion_system = MultiModalFusionSystem()
        
        # Mock sensor inputs
        self.mock_vision_data = {
            'objects': [
                {'id': 'red_ball', 'position': [1.0, 0.5, 0.2], 'interactable': True, 'size': 0.1},
                {'id': 'blue_box', 'position': [0.8, -0.3, 0.1], 'interactable': True, 'size': 0.2}
            ],
            'people': [
                {'id': 'person1', 'position': [2.0, 0.0, 0.0], 'distance': 2.0, 'is_interacting': True, 'looking_at_robot': True}
            ],
            'image': np.zeros((480, 640, 3), dtype=np.uint8)  # Mock image
        }
        
        self.mock_speech_data = {
            'transcript': 'Hello robot, can you see me?',
            'confidence': 0.95,
            'timestamp': time.time()
        }
        
        print("Complete Multi-Modal System initialized")
    
    def run_demo_scenario(self):
        """Run a demonstration of multi-modal interaction"""
        print("Multi-Modal Interaction Demo")
        print("=" * 50)
        
        # Scenario 1: Person enters field of view
        print("\nScenario 1: Person detected")
        vision_with_person = self.mock_vision_data.copy()
        vision_with_person['people'] = [
            {'id': 'user1', 'position': [2.0, 0.0, 0.0], 'distance': 2.0, 'is_interacting': True, 'looking_at_robot': True}
        ]
        
        result1 = self.fusion_system.process_multi_modal_input(
            vision_data=vision_with_person
        )
        
        print(f"Attention status: {result1['attention_status']['focused_person']}")
        print(f"Gaze command: {result1.get('gaze_command', 'None')}")
        
        # Scenario 2: Speech input
        print("\nScenario 2: Speech received")
        result2 = self.fusion_system.process_multi_modal_input(
            speech_data=self.mock_speech_data,
            vision_data=vision_with_person
        )
        
        print(f"Response: {result2['response']['speech']}")
        print(f"Actions: {result2['response']['actions']}")
        
        # Scenario 3: Gesture input
        print("\nScenario 3: Gesture detected")
        mock_gestures = [{
            'type': 'hand',
            'name': 'wave',
            'confidence': 0.85,
            'timestamp': time.time()
        }]
        
        result3 = self.fusion_system.process_multi_modal_input(
            speech_data=self.mock_speech_data,
            vision_data=vision_with_person,
            gesture_data=mock_gestures
        )
        
        print(f"Response: {result3['response']['speech']}")
        print(f"Actions: {result3['response']['actions']}")
        
        # Scenario 4: Combined input
        print("\nScenario 4: Multi-modal input")
        mock_speech2 = {'transcript': 'Can you pick up the red ball?', 'confidence': 0.92, 'timestamp': time.time()}
        mock_gesture2 = [{
            'type': 'hand', 
            'name': 'point', 
            'confidence': 0.78, 
            'timestamp': time.time()
        }]
        
        result4 = self.fusion_system.process_multi_modal_input(
            speech_data=mock_speech2,
            vision_data=vision_with_person,
            gesture_data=mock_gesture2
        )
        
        print(f"Response: {result4['response']['speech']}")
        print(f"Actions: {result4['response']['actions']}")
        
        # Show final interaction state
        state = self.fusion_system.get_interaction_state()
        print(f"\nFinal interaction state:")
        print(f"  Partner: {state['current_partner']}")
        print(f"  Attended objects: {state['attention_status']['attended_objects']}")
        print(f"  Recent gestures: {[g['name'] for g in state['recent_gestures']]}")
    
    def simulate_real_time_interaction(self):
        """Simulate real-time multi-modal interaction"""
        print("\nReal-time Interaction Simulation")
        print("-" * 40)
        
        import random
        
        # Simulate a series of events
        events = [
            ("Person enters room", lambda: self._event_person_enters()),
            ("Person waves", lambda: self._event_person_waves()),
            ("Person speaks", lambda: self._event_person_speaks()),
            ("Person points at object", lambda: self._event_person_points()),
            ("Person leaves", lambda: self._event_person_leaves())
        ]
        
        for i, (event_desc, event_func) in enumerate(events):
            print(f"\nEvent {i+1}: {event_desc}")
            result = event_func()
            if result:
                print(f"  System response: {result['response']['speech']}")
                print(f"  Actions taken: {result['response']['actions']}")
            time.sleep(0.5)  # Brief pause between events
    
    def _event_person_enters(self):
        """Simulate person entering"""
        vision_data = {
            'people': [{'id': 'visitor', 'position': [3.0, 0.0, 0.0], 'distance': 3.0, 'is_interacting': True, 'looking_at_robot': True}],
            'objects': [],
            'image': np.zeros((480, 640, 3), dtype=np.uint8)
        }
        return self.fusion_system.process_multi_modal_input(vision_data=vision_data)
    
    def _event_person_waves(self):
        """Simulate person waving"""
        gesture_data = [{'type': 'hand', 'name': 'wave', 'confidence': 0.85, 'timestamp': time.time()}]
        return self.fusion_system.process_multi_modal_input(gesture_data=gesture_data)
    
    def _event_person_speaks(self):
        """Simulate person speaking"""
        speech_data = {'transcript': 'Hello there!', 'confidence': 0.90, 'timestamp': time.time()}
        return self.fusion_system.process_multi_modal_input(speech_data=speech_data)
    
    def _event_person_points(self):
        """Simulate person pointing at object"""
        vision_data = {
            'objects': [{'id': 'interesting_object', 'position': [1.5, 0.2, 0.5], 'interactable': True, 'size': 0.1}],
            'people': [{'id': 'visitor', 'position': [2.0, 0.0, 0.0], 'distance': 2.0, 'is_interacting': True, 'looking_at_robot': True}],
            'image': np.zeros((480, 640, 3), dtype=np.uint8)
        }
        gesture_data = [{'type': 'hand', 'name': 'point', 'confidence': 0.80, 'timestamp': time.time()}]
        speech_data = {'transcript': 'Can you see that object?', 'confidence': 0.88, 'timestamp': time.time()}
        
        return self.fusion_system.process_multi_modal_input(
            speech_data=speech_data,
            vision_data=vision_data,
            gesture_data=gesture_data
        )
    
    def _event_person_leaves(self):
        """Simulate person leaving"""
        vision_data = {'people': [], 'objects': [], 'image': np.zeros((480, 640, 3), dtype=np.uint8)}
        return self.fusion_system.process_multi_modal_input(vision_data=vision_data)

# Example usage
def main():
    system = CompleteMultiModalSystem()
    
    # Run demonstration
    system.run_demo_scenario()
    
    # Run real-time simulation
    system.simulate_real_time_interaction()
    
    print("\nMulti-Modal Interaction System Demo Complete!")

if __name__ == "__main__":
    main()
```

## Evaluation and Performance Metrics

### Multi-Modal Interaction Evaluation

```python
class MultiModalEvaluator:
    def __init__(self):
        self.metrics = {
            'response_accuracy': [],
            'modal_coherence': [],
            'interaction_fluency': [],
            'user_satisfaction': [],
            'processing_latency': []
        }
    
    def evaluate_interaction(self, input_modalities: Dict, system_response: Dict, ground_truth: Dict) -> Dict:
        """Evaluate multi-modal interaction"""
        evaluation = {
            'response_accuracy': self._evaluate_response_accuracy(system_response, ground_truth),
            'modal_coherence': self._evaluate_modal_coherence(input_modalities, system_response),
            'interaction_fluency': self._evaluate_interaction_fluency(input_modalities, system_response),
            'processing_latency': self._evaluate_latency(system_response),
            'overall_score': 0.0
        }
        
        # Store metrics
        self.metrics['response_accuracy'].append(evaluation['response_accuracy'])
        self.metrics['modal_coherence'].append(evaluation['modal_coherence'])
        self.metrics['interaction_fluency'].append(evaluation['interaction_fluency'])
        self.metrics['processing_latency'].append(evaluation['processing_latency'])
        
        # Calculate overall score
        evaluation['overall_score'] = (
            evaluation['response_accuracy'] * 0.4 +
            evaluation['modal_coherence'] * 0.3 +
            evaluation['interaction_fluency'] * 0.2 +
            (1.0 - min(evaluation['processing_latency']/1.0, 1.0)) * 0.1  # Lower latency is better
        )
        
        return evaluation
    
    def _evaluate_response_accuracy(self, system_response: Dict, ground_truth: Dict) -> float:
        """Evaluate how accurately the system responded to input"""
        # This would compare system response to expected response
        # For demo, return a simulated score
        return 0.85  # Simulated accuracy
    
    def _evaluate_modal_coherence(self, input_modalities: Dict, system_response: Dict) -> float:
        """Evaluate how well the system integrated different modalities"""
        # Check if response acknowledges multiple modalities when present
        modalities_present = sum(1 for v in input_modalities.values() if v is not None)
        
        if modalities_present > 1:
            # Should have integrated response
            return 0.9 if 'integrated' in str(system_response.get('actions', [])).lower() else 0.6
        else:
            # Single modality, should respond appropriately
            return 0.8
    
    def _evaluate_interaction_fluency(self, input_modalities: Dict, system_response: Dict) -> float:
        """Evaluate the fluency of the interaction"""
        # Consider timing, naturalness, and appropriateness
        return 0.82  # Simulated fluency score
    
    def _evaluate_latency(self, system_response: Dict) -> float:
        """Evaluate processing latency"""
        # In a real system, this would measure actual processing time
        return 0.3  # Simulated latency in seconds
    
    def get_performance_report(self) -> Dict:
        """Get overall performance report"""
        report = {}
        for metric, values in self.metrics.items():
            if values:
                report[metric] = {
                    'average': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
            else:
                report[metric] = {'average': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0}
        
        return report
```

## Summary

Multi-modal interaction combines speech, gesture, vision, and other modalities to create natural and intuitive human-robot interaction. Key components include:

- **Visual Attention System**: Managing focus on relevant objects and people
- **Gesture Recognition**: Understanding human gestures and body language
- **Gaze Control**: Directing robot attention appropriately
- **Fusion Engine**: Combining information from multiple modalities
- **Response Generation**: Creating coordinated responses across modalities

Successful multi-modal interaction requires careful synchronization, conflict resolution, and contextual understanding. The system must be able to process different modalities at different speeds and integrate them into a coherent understanding of human intent.

## Next Steps

With a solid foundation in multi-modal interaction, we can now explore advanced topics in conversational robotics, including integration with large language models, advanced dialogue management, and evaluation methodologies for human-robot interaction systems.