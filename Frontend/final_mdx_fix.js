const fs = require('fs');
const path = require('path');

// Function to fix MDX issues in a specific file by ensuring Python code is in code blocks
function fixMDXInFile(filePath) {
    console.log(`Fixing MDX issues in ${filePath}...`);
    
    let content = fs.readFileSync(filePath, 'utf8');
    const originalContent = content;
    
    // Split content into lines for processing
    const lines = content.split('\n');
    const newLines = [];
    
    let i = 0;
    while (i < lines.length) {
        const line = lines[i];
        const trimmedLine = line.trim();
        
        // Check if this looks like Python code that should be in a code block
        if (!line.startsWith('```') && 
            (trimmedLine.startsWith('import ') || 
             trimmedLine.startsWith('from ') || 
             trimmedLine.startsWith('def ') || 
             trimmedLine.startsWith('class ') || 
             trimmedLine.startsWith('if ') || 
             trimmedLine.startsWith('for ') || 
             trimmedLine.startsWith('while ') || 
             trimmedLine.startsWith('try:') || 
             trimmedLine.startsWith('except ') || 
             trimmedLine.startsWith('with ') || 
             trimmedLine.startsWith('async ') || 
             /^[a-zA-Z_][a-zA-Z0-9_]*\s*=/g.test(trimmedLine) ||  // variable assignments
             /^[a-zA-Z_][a-zA-Z0-9_]*\s*\(.*\)/g.test(trimmedLine) // function calls
            )) {
            
            // Check if previous line is not a code block start
            const prevLine = i > 0 ? lines[i-1].trim() : '';
            if (!prevLine.startsWith('```')) {
                // Add code block start before this line
                newLines.push('```python');
            }
            
            // Add the current line
            newLines.push(line);
            
            // Look ahead to see how many consecutive Python lines we have
            let j = i + 1;
            while (j < lines.length) {
                const nextLine = lines[j].trim();
                
                // If it's a blank line, include it in the code block
                if (nextLine === '') {
                    newLines.push(lines[j]);
                    j++;
                    continue;
                }
                
                // Check if this also looks like Python code
                if (!nextLine.startsWith('```') && 
                    (nextLine.startsWith('import ') || 
                     nextLine.startsWith('from ') || 
                     nextLine.startsWith('def ') || 
                     nextLine.startsWith('class ') || 
                     nextLine.startsWith('if ') || 
                     nextLine.startsWith('for ') || 
                     nextLine.startsWith('while ') || 
                     nextLine.startsWith('try:') || 
                     nextLine.startsWith('except ') || 
                     nextLine.startsWith('with ') || 
                     nextLine.startsWith('async ') || 
                     nextLine.startsWith('    ') ||  // Indented code
                     nextLine.startsWith('else:') || 
                     nextLine.startsWith('elif ') || 
                     nextLine.startsWith('finally:') || 
                     /^[a-zA-Z_][a-zA-Z0-9_]*\s*=/g.test(nextLine) ||  
                     /^[a-zA-Z_][a-zA-Z0-9_]*\s*\(.*\)/g.test(nextLine)
                    )) {
                    newLines.push(lines[j]);
                    j++;
                } else if (nextLine.startsWith('```')) {
                    // Next line starts a new code block, so don't end this one
                    i = j;
                    break;
                } else {
                    // This line is not Python code, so end the code block
                    newLines.push('```');
                    i = j;
                    break;
                }
            }
            
            // If we reached the end of the file while in a code block
            if (j === lines.length) {
                newLines.push('```');
                i = j;
            }
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

// List of files that still have MDX issues based on the error log
const problematicFiles = [
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

// Process each problematic file
problematicFiles.forEach(file => {
    const fullPath = path.join(__dirname, file);
    if (fs.existsSync(fullPath)) {
        fixMDXInFile(fullPath);
    } else {
        console.log(`File does not exist: ${fullPath}`);
    }
});

console.log('All MDX fixes applied. Now run: npx docusaurus build');