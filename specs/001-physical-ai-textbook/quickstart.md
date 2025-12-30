# Quickstart Guide: Physical AI & Humanoid Robotics Textbook

## Overview
This guide provides instructions for setting up the development environment for the Physical AI & Humanoid Robotics Textbook project, including the Docusaurus-based textbook site and the RAG chatbot backend.

## Prerequisites
- Node.js v18 or higher
- npm or yarn package manager
- Git
- Python 3.8 or higher (for ROS 2 examples)
- Docker (for running backend services locally)

## Setting Up the Textbook Site

### 1. Clone the Repository
```bash
git clone https://github.com/your-organization/physical-ai-textbook.git
cd physical-ai-textbook
```

### 2. Install Dependencies
```bash
npm install
```

### 3. Start the Development Server
```bash
npm start
```
This command starts a local development server and opens the textbook in your browser at `http://localhost:3000`. Most changes are reflected live without restarting the server.

## Project Structure
The textbook content is organized as follows:
```
docs/
├── front-matter/        # Intro, motivation, embodied intelligence overview
├── module-1/            # Robotic Nervous System (ROS 2)
│   ├── chapter-1/
│   ├── chapter-2/
│   └── chapter-3/
├── module-2/            # Digital Twin (Gazebo & Unity)
│   ├── chapter-1/
│   ├── chapter-2/
│   └── chapter-3/
├── module-3/            # AI-Robot Brain (NVIDIA Isaac)
│   ├── chapter-1/
│   ├── chapter-2/
│   └── chapter-3/
├── module-4/            # Vision-Language-Action (VLA)
│   ├── chapter-1/
│   ├── chapter-2/
│   └── chapter-3/
├── capstone/            # Autonomous humanoid robot project
└── assets/              # Images, diagrams, code snippets
```

## Adding New Content

### Creating a New Chapter
1. Navigate to the appropriate module directory
2. Create a new directory for the chapter (e.g., `chapter-4`)
3. Add a `README.md` file with the chapter content
4. Update `sidebar.js` to include the new chapter in the navigation

### Content Guidelines
- Use Markdown format for all content
- Include learning objectives at the beginning of each chapter
- Provide executable code examples with explanations
- Add appropriate diagrams and images to the `assets/` directory

## Setting Up the RAG Chatbot Backend (Development)

### 1. Environment Configuration
Create a `.env` file in the project root with the following variables:
```env
OPENAI_API_KEY=your_openai_api_key
DATABASE_URL=postgresql://user:password@localhost:5432/textbook_chatbot
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key
```

### 2. Run Backend Services
The backend services can be run with Docker:
```bash
docker-compose up -d
```

### 3. Start the Backend Server
```bash
cd backend
pip install -r requirements.txt
python main.py
```

## Building for Production
To build the textbook site for production deployment:

```bash
npm run build
```

The built site will be in the `build/` directory and can be deployed to any static hosting service.

## Deploying to GitHub Pages
The textbook is configured to deploy to GitHub Pages. To trigger a deployment:

1. Commit your changes
2. Push to the `main` branch
3. The GitHub Actions workflow will automatically build and deploy the site

## Running Tests
To verify the textbook content and build process:

```bash
# Run content validation tests
python tests/test_textbook_content.py

# Run build verification
npm run build

# Run the application locally to validate all functionality
npm start
```

This runs build verification and content validation checks.

## Validation Checklist
Before deployment, ensure the following validations pass:

- [ ] Docusaurus site builds without errors: `npm run build`
- [ ] All documentation pages render correctly
- [ ] Module 1-4 chapters contain learning objectives and summaries
- [ ] Code examples in assets directories are present and syntactically valid
- [ ] All navigation links work correctly
- [ ] Sidebar navigation includes all modules and chapters
- [ ] Content validation tests pass: `python tests/test_textbook_content.py`
- [ ] The site is responsive and accessible across devices

## Troubleshooting

### Common Issues
1. **Module not found errors**: Run `npm install` to ensure all dependencies are installed
2. **Build errors**: Check for syntax errors in Markdown files
3. **Slow page loads**: Optimize images and reduce the number of heavy components

### Verification Steps
1. Ensure the development server starts without errors
2. Navigate through several textbook pages to verify content displays correctly
3. Test the search functionality
4. Verify that code examples are properly formatted

## Next Steps
- Review the [contribution guidelines](CONTRIBUTING.md) for submitting changes
- Check the [task list](specs/001-physical-ai-textbook/tasks.md) for current work items
- Join our development community for questions and collaboration