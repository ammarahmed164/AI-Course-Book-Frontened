---
id: 1
title: Create Physical AI Constitution
stage: constitution
date_iso: 2025-12-16
surface: agent
model: Qwen
feature: none
branch: main
user: Specify Plus User
command: /sp.constitution Project: Physical AI & Humanoid Robotics Textbook (Hackathon I)
labels: ["constitution", "physical-ai", "textbook", "hackathon"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files_yaml:
  - .specify/memory/constitution.md
tests_yaml: []
---

# PHR: Create Physical AI & Humanoid Robotics Textbook Constitution

## Outcome
Successfully created the project constitution for the Physical AI & Humanoid Robotics Textbook project, establishing six core principles and governance guidelines.

## Evaluation
The constitution was created based on the specified project requirements and includes:
- Six core principles: Embodied Intelligence Focus, Technical Accuracy & Real-World Alignment, Reproducible Simulation-to-Real Pipeline, Industry-Grade Practices, Comprehensive Testing & Validation, and Academic Excellence & RIGOR
- Course-aligned scope section covering the key topics
- RAG Chatbot Integration section
- Governance section with amendment procedures
- Version 1.0.0 with proper ratification date

## Prompt Text
```
/sp.constitution Project: Physical AI & Humanoid Robotics Textbook (Hackathon I)

Purpose:
Create an AI-native academic textbook for teaching Physical AI and Humanoid Robotics, focused on embodied intelligence and AI systems operating in the physical world. The book must support both simulation-first learning and sim-to-real deployment.

Audience:
Advanced students and professionals with background in AI, Python, and computer science.

Core principles:
- Embodied intelligence over purely digital AI
- Accuracy aligned with real robotics toolchains
- Reproducible simulations and deployments
- Industry-grade, production-relevant practices

Delivery requirements:
- Author the book in Markdown using Docusaurus
- Deploy the textbook to GitHub Pages
- Use Spec-Kit Plus and Claude Code as primary authoring tools

Course-aligned scope:
- Physical AI foundations and humanoid robotics
- ROS 2 middleware (nodes, topics, services, rclpy, URDF)
- Digital twins and simulation (Gazebo, Unity)
- NVIDIA Isaac Sim and Isaac ROS
- VSLAM, navigation, and humanoid motion (Nav2)
- Vision-Language-Action systems
- Voice-to-action using OpenAI Whisper
- Capstone: fully autonomous simulated humanoid robot

RAG chatbot requirements:
- Embed a Retrieval-Augmented Generation chatbot in the book
- Answers must be grounded strictly in book content
- Support answering questions from user-selected text only
- Stack: OpenAI Agents/ChatKit SDKs, FastAPI, Neon Serverless Postgres, Qdrant Cloud

Quality standards:
- No hallucinated APIs, hardware, or workflows
- All technical descriptions must reflect real systems
- Step-by-step explanations for simulations and pipelines
- Clear separation of theory, implementation, and deployment

Constraints:
- Zero plagiarism tolerance
- Static-site compatible content only
- Executable, realistic code examples

Success criteria:
- Operates as a complete quarter-level university course
- Builds and deploys cleanly via Docusaurus
- RAG chatbot answers accurately from textbook data
- Supports both simulated and physical AI learning paths
```

## Response Text
Constitution created with version 1.0.0 containing six core principles:
1. Embodied Intelligence Focus
2. Technical Accuracy & Real-World Alignment
3. Reproducible Simulation-to-Real Pipeline
4. Industry-Grade Practices
5. Comprehensive Testing & Validation
6. Academic Excellence & RIGOR

Plus sections for course-aligned scope and RAG chatbot integration.