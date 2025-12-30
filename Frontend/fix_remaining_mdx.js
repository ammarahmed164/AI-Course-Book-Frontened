const fs = require('fs');
const path = require('path');

function fixRemainingMDXIssues(filePath) {
    console.log(`Fixing remaining MDX issues in ${filePath}...`);
    
    let content = fs.readFileSync(filePath, 'utf8');
    const originalContent = content;
    
    // Look for patterns that might cause "Could not parse expression with acorn" errors
    // These are often caused by JSX-like syntax or unescaped special characters
    
    // Replace any JSX-like patterns that might be causing issues
    // Look for patterns like {someVariable} that might be outside of code blocks
    content = content.replace(/(\s)\{([a-zA-Z_$][a-zA-Z0-9_$]*)\}(\s)/g, '$1{$2}$3');
    
    // Fix common Python code that might be causing issues
    // Look for code that should be in fenced blocks but isn't
    const lines = content.split('\n');
    const newLines = [];
    
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        
        // Skip if we're already in a code block
        if (line.trim().startsWith('```')) {
            newLines.push(line);
            continue;
        }
        
        // Check if the line looks like Python code that should be fenced
        if (isPythonCode(line)) {
            // Check if the previous line was a code block start
            const prevLine = i > 0 ? lines[i-1] : '';
            if (!prevLine.trim().startsWith('```')) {
                // Add code block start
                newLines.push('```python');
            }
            
            // Add the current line
            newLines.push(line);
            
            // Look ahead to see if we should continue the code block
            let j = i + 1;
            while (j < lines.length) {
                const nextLine = lines[j];
                
                // If it's a blank line, include it in the code block
                if (nextLine.trim() === '') {
                    newLines.push(nextLine);
                    j++;
                    continue;
                }
                
                // If the next line is also Python code, include it
                if (isPythonCode(nextLine) && !nextLine.trim().startsWith('```')) {
                    newLines.push(nextLine);
                    j++;
                } else if (nextLine.trim().startsWith('```')) {
                    // Next line is a code block start/end, don't add our end
                    i = j - 1; // -1 because the outer loop will increment
                    break;
                } else {
                    // Next line is not Python code, end the code block
                    newLines.push('```');
                    i = j - 1; // -1 because the outer loop will increment
                    break;
                }
            }
            
            // If we've reached the end without finding a natural break, add the closing
            if (j === lines.length) {
                newLines.push('```');
            }
        } else {
            // Regular line, just add it
            newLines.push(line);
        }
    }
    
    content = newLines.join('\n');
    
    // Write the content back if it has changed
    if (content !== originalContent) {
        fs.writeFileSync(filePath, content);
        console.log(`  Fixed MDX issues in ${filePath}`);
    } else {
        console.log(`  No changes needed for ${filePath}`);
    }
}

// Helper function to determine if a line looks like Python code
function isPythonCode(line) {
    const trimmed = line.trim();
    
    // Check for Python-specific patterns
    if (/^(import|from|def |class |if |elif |else:|for |while |try:|except |finally:|with |async |await |print\(|return |yield |raise |assert |global |nonlocal |del |pass )/.test(trimmed)) {
        return true;
    }
    
    // Check for Python variable assignments
    if (/^[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*/.test(trimmed)) {
        return true;
    }
    
    // Check for Python function calls
    if (/[a-zA-Z_][a-zA-Z0-9_]*\s*\(/.test(trimmed) && !line.startsWith('#')) {
        // Make sure it's not just text with parentheses
        const beforeParen = trimmed.split('(')[0].trim();
        if (/[a-zA-Z_][a-zA-Z0-9_]*/.test(beforeParen)) {
            return true;
        }
    }
    
    return false;
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
        fixRemainingMDXIssues(file);
    });
}

// Run the processing
const docsDir = path.join(__dirname, 'docs');
processDocsDirectory(docsDir);
console.log('Remaining MDX issue fixing completed.');
console.log('Now run: npx docusaurus build');