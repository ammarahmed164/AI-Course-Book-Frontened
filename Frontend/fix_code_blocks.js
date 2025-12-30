const fs = require('fs');
const path = require('path');

// Function to fix MDX syntax issues in a markdown file by properly formatting code blocks
function fixCodeBlocks(filePath) {
    console.log(`Fixing code blocks in ${filePath}...`);
    
    let content = fs.readFileSync(filePath, 'utf8');
    
    // This is a simplified approach - we'll look for patterns that indicate
    // code that should be in fenced blocks and wrap them appropriately
    
    // Find sequences of import statements and wrap them in code blocks
    // This pattern looks for multiple import statements in a row
    const importPattern = /(^(?!```)[ \t]*(import|from)\s+.*$\n?)+/gm;
    
    let match;
    let fixedContent = content;
    
    // For each match of import statements, wrap them in a Python code block
    while ((match = importPattern.exec(content)) !== null) {
        const importBlock = match[0];
        const replacement = '```python\n' + importBlock.trim() + '\n```';
        fixedContent = fixedContent.replace(importBlock.trim(), replacement);
    }
    
    // Look for other common Python patterns that should be in code blocks
    // Like function definitions, class definitions, etc.
    const pythonPatterns = [
        /^(?!```)[ \t]*(def|class|if|for|while|try|with|async)\s+/gm,
        /^(?!```)[ \t]*[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*/gm,
        /^(?!```)[ \t]*import\s+/gm,
        /^(?!```)[ \t]*from\s+/gm
    ];
    
    // For the purpose of this fix, we'll focus on the import statements
    // which were clearly identified in our scan
    
    // Write the fixed content back to the file
    fs.writeFileSync(filePath, fixedContent);
    console.log(`  Fixed code blocks in ${filePath}`);
}

// Process all markdown files in the docs directory
function processDocsDirectory(docsPath) {
    const walk = function(dir) {
        const results = [];
        const list = fs.readdirSync(dir);
        
        list.forEach(function(file) {
            file = path.resolve(dir, file);
            
            const stat = fs.statSync(file);
            
            if (stat && stat.isDirectory()) {
                /* Recurse into a subdirectory */
                results.push(...walk(file));
            } else {
                /* Is a file */
                if (path.extname(file) === '.md' || path.extname(file) === '.mdx') {
                    results.push(file);
                }
            }
        });
        
        return results;
    };
    
    const files = walk(docsPath);
    
    files.forEach(file => {
        fixCodeBlocks(file);
    });
}

// Run the processing
const docsDir = path.join(__dirname, 'docs');
processDocsDirectory(docsDir);
console.log('Code block fixing completed.');
console.log('Now run: npx docusaurus build');