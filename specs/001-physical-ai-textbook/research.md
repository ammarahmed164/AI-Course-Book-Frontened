# Research for Physical AI & Humanoid Robotics Textbook

## Decision: Docusaurus as Static Site Generator
**Rationale**: Docusaurus is a well-established, open-source static site generator specifically designed for documentation. It provides features like versioning, search, and easy navigation which are ideal for textbook content. It's supported by Meta and has a large community.

**Alternatives considered**: 
- GitBook: Good for books but less customizable than Docusaurus
- Sphinx: Python-focused, would limit flexibility for multi-language content
- Custom React app: Would require much more development time for basic documentation features

## Decision: GitHub Pages for Deployment
**Rationale**: GitHub Pages provides free hosting that integrates seamlessly with GitHub repositories. It's reliable, fast, and appropriate for static textbook content. Supports custom domains and HTTPS by default.

**Alternatives considered**:
- Netlify: More features but would add complexity without significant benefit
- AWS S3: Overkill for static textbook content with increased costs
- Vercel: Good alternative but GitHub Pages is more tightly integrated with the development workflow

## Decision: Module Structure with 4 Core Modules + Capstone
**Rationale**: The 4-module structure aligns with the progressive learning approach from basic ROS 2 concepts to advanced AI-Robot integration, culminating in a comprehensive capstone project. This structure follows pedagogical best practices of building complexity gradually.

**Alternatives considered**:
- Single sequential structure: Less modular, harder to update individual components
- More/less modules: Would either fragment the learning path or make modules too comprehensive

## Decision: Markdown for Content Format
**Rationale**: Markdown is lightweight, version-controllable, and easily editable. It's well-supported by Docusaurus and allows for embedding code examples and diagrams. It's also familiar to the target audience of students and professionals.

**Alternatives considered**:
- Jupyter notebooks: Good for executable content but harder to maintain as static documentation
- RestructuredText: Used by Sphinx but less familiar to broader audience
- HTML: Too verbose for content creation and maintenance

## Decision: RAG Chatbot Implementation Stack
**Rationale**: Using OpenAI Agents/ChatKit SDKs with FastAPI, Neon Postgres, and Qdrant Cloud provides a robust, scalable solution for the textbook's RAG chatbot. FastAPI offers excellent performance and documentation, Neon provides serverless PostgreSQL, and Qdrant Cloud handles vector storage for semantic search.

**Alternatives considered**:
- LangChain with different backends: More complex but similar outcome
- Custom solution with different vector DB: Would require more development time
- Simple keyword search: Insufficient for understanding complex robotics concepts

## Decision: ROS 2 Distribution
**Rationale**: ROS 2 Humble Hawksbill is an LTS (Long Term Support) version with extensive documentation and community support. It's well-suited for educational content as it will remain supported throughout the textbook's lifecycle.

**Alternatives considered**:
- Rolling Ridley: Always up-to-date but changes frequently, making textbook maintenance difficult
- Iron Irwini: Non-LTS version, would reach end-of-life during textbook use
- Galactic Geochelone: Reaching end-of-life, not appropriate for new educational material

## Decision: Simulation Environment Stack
**Rationale**: Using Gazebo Garden for physics simulation provides realistic physics and sensor simulation. Unity for visualization offers high-quality rendering for Human-Robot Interaction (HRI) concepts. NVIDIA Isaac Sim provides advanced synthetic data generation and photorealistic simulation for AI training.

**Alternatives considered**:
- Only one simulation environment: Would limit the range of concepts that could be taught
- Different simulation tools: Would not offer the same breadth of capabilities for different learning objectives

## Decision: Content Organization with Theory, Simulation, Deployment Separation
**Rationale**: The clear separation of theory, simulation, and deployment aligns with the constitution's Academic Excellence & RIGOR principle, allowing students to understand concepts before implementing them in simulation, and finally considering real-world deployment considerations.

**Alternatives considered**:
- Integrated approach: Could confuse beginners by mixing concepts
- Different separation model: Would not align with the simulation-first, sim-to-real pedagogical approach specified in requirements