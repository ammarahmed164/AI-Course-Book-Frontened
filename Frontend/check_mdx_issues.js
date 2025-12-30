const fs = require('fs');
const path = require('path');

// Function to fix MDX syntax issues in a markdown file
function fixMDXIssues(filePath) {
    const content = fs.readFileSync(filePath, 'utf8');
    
    // Look for JavaScript function declarations and properly format them
    // This regex looks for patterns that might be causing MDX issues
    let fixedContent = content;
    
    // Replace problematic JavaScript function declarations with properly escaped code blocks
    // We'll look for patterns that start with function declarations and ensure they're in proper code blocks
    
    // First, let's try to identify the issue by looking at the structure around line 63
    const lines = content.split('\n');
    
    // Find function declarations that might be causing issues
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        if (line.startsWith('function ') || 
            line.startsWith('const ') || 
            line.startsWith('let ') || 
            line.startsWith('var ') || 
            line.match(/^\s*(export|import)\s+/) ||
            line.match(/^\s*{/)) {
            // Check if this line is properly inside a code block
            let inCodeBlock = false;
            let j = i;
            while (j >= 0) {
                if (lines[j].trim().startsWith('```')) {
                    inCodeBlock = !inCodeBlock;
                    break;
                }
                j--;
            }
            
            // If not in a code block, this might be causing the issue
            if (!inCodeBlock) {
                console.log(`Found potential issue at line ${i + 1} in ${filePath}: ${line.substring(0, 50)}...`);
            }
        }
    }
    
    // Write the content back (for now, just showing the issues)
    fs.writeFileSync(filePath, fixedContent);
    console.log(`Processed ${filePath}`);
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
        console.log(`Checking ${file}...`);
        fixMDXIssues(file);
    });
}

// Run the processing
const docsDir = path.join(__dirname, 'docs');
processDocsDirectory(docsDir);
console.log('MDX issue check completed.');