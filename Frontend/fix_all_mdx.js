const fs = require('fs');
const path = require('path');

// Comprehensive function to fix MDX issues in a file
function fixMDXIssues(filePath) {
    console.log(`Processing ${filePath}...`);
    
    let content = fs.readFileSync(filePath, 'utf8');
    const originalContent = content;
    
    // Define patterns that indicate Python code that should be in code blocks
    const pythonPatterns = [
        /^import\s+/,
        /^from\s+[a-zA-Z_]/,
        /^def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(/,
        /^class\s+[a-zA-Z_][a-zA-Z0-9_]*\s*/,
        /^if\s+/,
        /^elif\s+/,
        /^else:/,
        /^for\s+/,
        /^while\s+/,
        /^try:/,
        /^except\s+/,
        /^finally:/,
        /^with\s+/,
        /^async\s+/,
        /^[a-zA-Z_][a-zA-Z0-9_]*\s*=/,  // Variable assignments
        /^[a-zA-Z_][a-zA-Z0-9_]*\s*\(/  // Function calls
    ];
    
    const lines = content.split('\n');
    const newLines = [];
    let inCodeBlock = false;
    let i = 0;
    
    while (i < lines.length) {
        const line = lines[i];
        
        // Check if we're entering or leaving a code block
        if (line.trim().startsWith('```')) {
            inCodeBlock = !inCodeBlock;
            newLines.push(line);
            i++;
            continue;
        }
        
        // If we're already in a code block, just add the line
        if (inCodeBlock) {
            newLines.push(line);
            i++;
            continue;
        }
        
        // Check if this line looks like Python code
        const isPythonCode = pythonPatterns.some(pattern => pattern.test(line.trim()));
        
        if (isPythonCode) {
            // Check if the previous line is not a code block start
            const prevLine = i > 0 ? lines[i-1] : '';
            if (!prevLine.trim().startsWith('```')) {
                // Add opening code block
                newLines.push('```python');
            }
            
            // Add the current line
            newLines.push(line);
            
            // Look ahead to see how many consecutive Python lines we have
            let j = i + 1;
            while (j < lines.length) {
                const nextLine = lines[j];
                
                // Skip if we encounter a markdown element or code block
                if (nextLine.trim().startsWith('#') || 
                    nextLine.trim().startsWith('>') || 
                    nextLine.trim().startsWith('- ') || 
                    nextLine.trim().startsWith('* ') || 
                    nextLine.trim().startsWith('1. ') ||
                    nextLine.trim().startsWith('```')) {
                    break;
                }
                
                // If it's a blank line, include it in the code block
                if (nextLine.trim() === '') {
                    newLines.push(nextLine);
                    j++;
                    continue;
                }
                
                // Check if this also looks like Python code
                if (pythonPatterns.some(pattern => pattern.test(nextLine.trim()))) {
                    newLines.push(nextLine);
                    j++;
                } else {
                    // This line is not Python code, so end the code block
                    break;
                }
            }
            
            // Add closing code block
            newLines.push('```');
            i = j;
        } else {
            // Regular line, just add it
            newLines.push(line);
            i++;
        }
    }
    
    const newContent = newLines.join('\n');
    
    // Only write if content has changed
    if (newContent !== originalContent) {
        fs.writeFileSync(filePath, newContent);
        console.log(`  Fixed MDX issues in ${filePath}`);
    } else {
        console.log(`  No changes needed for ${filePath}`);
    }
}

// List of files that still have MDX issues
const filesWithIssues = [
    'docs/capstone/index.md',
    'docs/front-matter/README.md',
    'docs/front-matter/intro.md',
    'docs/front-matter/motivation.md',
    'docs/front-matter/outcomes.md',
    'docs/front-matter/roadmap.md',
    'docs/module-1/chapter-1/index.md',
    'docs/module-1/chapter-2/index.md',
    'docs/module-1/chapter-3/index.md',
    'docs/module-2/chapter-1/index.md',
    'docs/module-2/chapter-2/index.md',
    'docs/module-2/chapter-3/index.md',
    'docs/module-3/chapter-1/index.md',
    'docs/module-3/chapter-2/index.md',
    'docs/module-3/chapter-3/index.md',
    'docs/module-4/chapter-1/index.md',
    'docs/module-4/chapter-2/index.md',
    'docs/module-4/chapter-3/index.md'
];

// Process each file
filesWithIssues.forEach(file => {
    const fullPath = path.join(__dirname, file);
    if (fs.existsSync(fullPath)) {
        fixMDXIssues(fullPath);
    } else {
        console.log(`File does not exist: ${fullPath}`);
    }
});

console.log('All MDX fixes applied. You can now run: npx docusaurus start');