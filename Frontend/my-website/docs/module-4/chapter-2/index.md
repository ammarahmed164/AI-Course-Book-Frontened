# Chapter 2: Voice-to-Action Systems with OpenAI Whisper

## Learning Objectives

After completing this chapter, you will be able to:
- Implement voice command recognition using OpenAI Whisper
- Integrate speech-to-text with robotics command systems
- Design voice interaction flows for robotic systems
- Handle real-time voice processing and command mapping
- Optimize voice recognition for robotics applications

## Introduction to Voice Commands in Robotics

Voice commands provide a natural and intuitive interface for human-robot interaction. With the advancement of automatic speech recognition (ASR) technologies like OpenAI Whisper, robots can now understand and execute spoken commands with high accuracy.

### Benefits of Voice Commands
- Natural interaction without physical interfaces
- Hands-free operation
- Accessibility for users with physical limitations
- Efficient for complex command sequences

## OpenAI Whisper for Robotics

### Whisper Architecture

Whisper is a transformer-based model trained on a large dataset of multilingual speech. It's particularly well-suited for robotics applications because of its:
- Robustness to background noise
- Multi-language support
- Accuracy across different accents
- Open-source availability

### Installation and Setup

```bash
pip install openai-whisper
pip install pyaudio  # For live audio capture
pip install soundfile  # For audio file processing
```

### Basic Whisper Integration

```python
import whisper
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import pyaudio
import wave
import tempfile
import os

class VoiceToActionNode(Node):
    def __init__(self):
        super().__init__('voice_to_action_node')

        # Load Whisper model (this can be done at different sizes: tiny, base, small, medium, large)
        self.get_logger().info("Loading Whisper model...")
        self.model = whisper.load_model("base")  # Choose size based on performance needs

        # Publishers and subscribers
        self.speech_command_pub = self.create_publisher(
            String,
            '/robot/speech_command',
            10
        )

        # Audio configuration
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000  # Whisper expects 16kHz
        self.chunk = 1024
        self.record_seconds = 5

        # Start voice recognition
        self.p = pyaudio.PyAudio()

        self.get_logger().info("Voice-to-Action node initialized")

    def start_voice_recognition(self):
        """Begin listening for voice commands."""
        self.get_logger().info("Starting voice recognition...")

        while rclpy.ok():
            try:
                # Record audio from microphone
                frames = []

                stream = self.p.open(
                    format=self.audio_format,
                    channels=self.channels,
                    rate=self.rate,
                    input=True,
                    frames_per_buffer=self.chunk
                )

                self.get_logger().info("Listening... Speak now.")

                for _ in range(0, int(self.rate / self.chunk * self.record_seconds)):
                    data = stream.read(self.chunk)
                    frames.append(data)

                # Stop recording
                stream.stop_stream()
                stream.close()

                # Save audio to temporary file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    wf = wave.open(tmp_file.name, 'wb')
                    wf.setnchannels(self.channels)
                    wf.setsampwidth(self.p.get_sample_size(self.audio_format))
                    wf.setframerate(self.rate)
                    wf.writeframes(b''.join(frames))
                    wf.close()

                    # Transcribe using Whisper
                    result = self.model.transcribe(tmp_file.name)

                    # Clean up temp file
                    os.unlink(tmp_file.name)

                # Process the transcribed text
                if result['text'].strip():
                    self.process_voice_command(result['text'].strip())

            except Exception as e:
                self.get_logger().error(f"Error in voice recognition: {e}")

    def process_voice_command(self, text):
        """Process transcribed text and generate appropriate command."""
        self.get_logger().info(f"Recognized: {text}")

        # Publish command to robot system
        cmd_msg = String()
        cmd_msg.data = text
        self.speech_command_pub.publish(cmd_msg)

def main(args=None):
    rclpy.init(args=args)
    node = VoiceToActionNode()

    try:
        # Start voice recognition in a separate thread to keep ROS2 spinning
        import threading
        voice_thread = threading.Thread(target=node.start_voice_recognition)
        voice_thread.daemon = True
        voice_thread.start()

        rclpy.spin(node)

    except KeyboardInterrupt:
        pass
    finally:
        node.p.terminate()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary

Voice commands provide a natural and intuitive interface for human-robot interaction. With the advancement of automatic speech recognition technologies like OpenAI Whisper, robots can now understand and execute spoken commands with high accuracy, enabling more accessible and efficient human-robot collaboration.

## Diagrams and Visual Aids

![Voice Command Architecture](/img/voice-command-arch.png)

*Figure 1: Voice command processing architecture*

![Whisper Integration](/img/whisper-integration.png)

*Figure 2: Integration of Whisper ASR in robotic systems*