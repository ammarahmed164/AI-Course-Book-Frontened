const fs = require('fs');
const path = require('path');

// Function to specifically fix MDX issues by wrapping Python code in proper code blocks
function fixMDXFile(filePath) {
    console.log(`Fixing MDX issues in ${filePath}...`);
    
    let content = fs.readFileSync(filePath, 'utf8');
    const originalContent = content;
    
    // Look for Python code that's not properly wrapped in code blocks
    // This uses a more sophisticated approach to identify and wrap Python code
    
    // Split content into lines
    const lines = content.split('\n');
    const newLines = [];
    let i = 0;
    
    while (i < lines.length) {
        const line = lines[i];
        
        // Check if this line starts with common Python patterns
        const pythonPattern = /^(import|from|def |class |if:|if\s+|elif\s+|else:|for\s+|while\s+|try:|except|finally:|with\s+|async\s+|and\s+|or\s+|not\s+|[a-zA-Z_][a-zA-Z0-9_]*\s*=\s|print\(|return\s|yield\s|raise\s|assert\s|global\s|nonlocal\s|del\s|pass\s)/;
        
        if (!line.trim().startsWith('```') && pythonPattern.test(line.trim())) {
            // Check if the previous line is not a code block start
            const prevLine = i > 0 ? lines[i-1] : '';
            if (!prevLine.trim().startsWith('```')) {
                // Add opening code block before this line
                newLines.push('```python');
            }
            newLines.push(line);
            
            // Look ahead for more Python code lines
            let j = i + 1;
            while (j < lines.length) {
                const nextLine = lines[j].trim();
                
                // If it's a blank line, include it in the code block
                if (nextLine === '') {
                    newLines.push(lines[j]);
                    j++;
                    continue;
                }
                
                // If it's a markdown element, stop the code block
                if (nextLine.startsWith('#') || nextLine.startsWith('>') || 
                    nextLine.startsWith('- ') || nextLine.startsWith('* ') || 
                    nextLine.startsWith('1. ') || nextLine.startsWith('```')) {
                    break;
                }
                
                // If it's more Python code, include it
                if (pythonPattern.test(nextLine)) {
                    newLines.push(lines[j]);
                    j++;
                } else {
                    // Not Python code, so end the code block
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

// Apply fixes to all problematic files
const filesToFix = [
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

filesToFix.forEach(file => {
    const fullPath = path.join(__dirname, file);
    if (fs.existsSync(fullPath)) {
        fixMDXFile(fullPath);
    } else {
        console.log(`File does not exist: ${fullPath}`);
    }
});

console.log('All MDX fixes applied. Run: npx docusaurus clear && npx docusaurus start');