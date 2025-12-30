# Data Model for Physical AI & Humanoid Robotics Textbook

## Entities

### Textbook Content
- **ID**: Unique identifier for the textbook
- **Title**: Physical AI & Humanoid Robotics Textbook
- **Version**: Version identifier following semantic versioning
- **Metadata**: Author, publication date, course alignment information
- **Modules**: Collection of modules in the textbook

### Module
- **ID**: Unique identifier for the module
- **Title**: Descriptive name of the module (e.g. "Robotic Nervous System (ROS 2)")
- **Description**: Brief overview of the module content
- **LearningObjectives**: List of learning objectives for the module
- **Chapters**: Collection of chapters in the module
- **ModuleNumber**: Sequential number of the module (1-4)
- **RelatedTechnologies**: List of technologies covered in the module

### Chapter
- **ID**: Unique identifier for the chapter
- **Title**: Descriptive name of the chapter
- **Description**: Brief overview of the chapter content
- **LearningObjectives**: List of learning objectives for the chapter
- **Content**: Markdown content of the chapter
- **CodeExamples**: Collection of executable code examples in the chapter
- **ChapterNumber**: Sequential number of the chapter within the module
- **ModuleID**: Reference to the parent module
- **Assets**: List of images, diagrams, and other assets used in the chapter

### CodeExample
- **ID**: Unique identifier for the code example
- **Title**: Descriptive name of the example
- **Language**: Programming language (Python, C++, etc.)
- **Code**: The actual code content
- **Description**: Explanation of what the code does
- **ExecutionInstructions**: Steps to run the code example
- **Requirements**: Dependencies or setup needed to run the example
- **ChapterID**: Reference to the parent chapter

### Asset
- **ID**: Unique identifier for the asset
- **FileName**: Name of the asset file
- **Type**: Image, diagram, video, etc.
- **Description**: Brief description of the asset
- **ChapterID**: Reference to the chapter where the asset is used
- **UsageContext**: Where in the textbook the asset appears

### CapstoneProject
- **ID**: Unique identifier for the capstone project
- **Title**: "Autonomous Humanoid Robot"
- **Description**: Detailed overview of the capstone project
- **LearningObjectives**: List of learning objectives for the capstone
- **Requirements**: List of requirements to complete the capstone
- **Steps**: Sequential steps to complete the project
- **ValidationCriteria**: Criteria to validate project completion

### UserQuestion
- **ID**: Unique identifier for the question
- **Text**: The text of the user's question
- **SourceText**: Specific text in the textbook that the question refers to (if applicable)
- **Timestamp**: When the question was asked
- **UserID**: Identifier for the user asking the question (if available)

### ChatbotResponse
- **ID**: Unique identifier for the response
- **QuestionID**: Reference to the associated question
- **ResponseText**: The text of the response provided by the chatbot
- **SourceContent**: Specific textbook content used to generate the response
- **Timestamp**: When the response was generated
- **Accuracy**: Confidence in the response accuracy

## Relationships

### Module-Chapters
- One module contains many chapters
- Each chapter belongs to one module

### Chapter-CodeExamples
- One chapter may contain many code examples
- Each code example belongs to one chapter

### Chapter-Assets
- One chapter may use many assets
- Each asset is associated with one chapter (though may be referenced in multiple places)

### Textbook-Modules
- One textbook contains many modules
- Each module belongs to one textbook

## Validation Rules

### Module Validation
- Module must have a unique title within the textbook
- Module must have at least one chapter
- Module number must be between 1 and 4

### Chapter Validation
- Chapter must have a unique title within its module
- Chapter content must be in valid Markdown format
- Chapter must have associated learning objectives

### CodeExample Validation
- Code must be executable and tested
- Language must be specified
- Execution instructions must be provided

### CapstoneProject Validation
- Capstone must have detailed requirements
- Capstone must have validation criteria
- Capstone must integrate concepts from all modules

## State Transitions

### Content States
- Draft → Review → Approved → Published → Archived
- Each state transition requires validation and approval