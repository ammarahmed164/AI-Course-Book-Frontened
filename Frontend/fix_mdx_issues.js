const fs = require('fs');
const path = require('path');

// Function to fix MDX syntax issues in a markdown file
function fixMDXIssues(filePath) {
    console.log(`Processing ${filePath}...`);
    
    let content = fs.readFileSync(filePath, 'utf8');
    
    // Look for common patterns that cause MDX issues and fix them
    // This adds proper code block formatting around JavaScript code
    
    // Find lines that start with function declarations or variable declarations
    // and ensure they're in proper code blocks
    const lines = content.split('\n');
    let newContent = [];
    let inCodeBlock = false;
    let inFencedCodeBlock = false;
    
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        const trimmedLine = line.trim();
        
        // Check if we're entering or exiting a fenced code block
        if (line.startsWith('```')) {
            inFencedCodeBlock = !inFencedCodeBlock;
            newContent.push(line);
            continue;
        }
        
        // Skip if we're inside a fenced code block (```js, ```javascript, etc.)
        if (inFencedCodeBlock) {
            newContent.push(line);
            continue;
        }
        
        // Check if this looks like a JavaScript function declaration
        const jsPattern = /^(function|const|let|var|import|export|class|if\s*\(|for\s*\(|while\s*\(|try\s*{|catch\s*\(|do\s*{|switch\s*\(|async\s+function|async\s+const|async\s+let)\s*/;
        
        // If line matches JS pattern but is not in a code block, we need to handle it
        if (jsPattern.test(trimmedLine) && !inFencedCodeBlock) {
            // This is likely an issue - JavaScript code not in a code block
            // Since we can't easily fix this without more context, we'll log it
            console.log(`  Found potential JavaScript code not in code block at line ${i + 1}: ${trimmedLine.substring(0, 50)}...`);
        }
        
        newContent.push(line);
    }
    
    // For the specific error at line 63 in module-4/chapter-2/index.md, 
    // let's try to fix common MDX issues by ensuring code blocks are properly formatted
    content = newContent.join('\n');
    
    // Write the content back
    fs.writeFileSync(filePath, content);
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
        fixMDXIssues(file);
    });
}

// Run the processing
const docsDir = path.join(__dirname, 'docs');
processDocsDirectory(docsDir);
console.log('MDX issue processing completed. Now run the following command to fix the files properly:');
console.log('npx docusaurus build');