const fs = require('fs');
const path = require('path');

// More conservative function to fix MDX syntax issues in markdown files
function fixSpecificFile(filePath) {
    console.log(`Fixing ${filePath}...`);
    
    let content = fs.readFileSync(filePath, 'utf8');
    const originalContent = content;
    
    // Split content into lines for processing
    const lines = content.split('\n');
    const newLines = [];
    
    let i = 0;
    while (i < lines.length) {
        const line = lines[i];
        const trimmedLine = line.trim();
        
        // Check if this line looks like a Python import statement
        if (/^(import|from)\s+/.test(trimmedLine) && !line.startsWith('```')) {
            // Check if the previous line is not a code block start
            const prevLine = i > 0 ? lines[i-1].trim() : '';
            if (!prevLine.startsWith('```')) {
                // Add code block start before this import
                newLines.push('```python');
            }
            
            // Add the import line
            newLines.push(line);
            
            // Look ahead for more import statements or Python code
            let j = i + 1;
            while (j < lines.length) {
                const nextLine = lines[j].trim();
                
                // If we encounter a blank line followed by regular text, or a markdown element, stop
                if (nextLine.startsWith('```') || 
                    (!nextLine.startsWith('#') && !nextLine.startsWith('>') && 
                     !nextLine.startsWith('-') && !nextLine.startsWith('1.') && 
                     nextLine !== '' && /^[a-zA-Z]/.test(nextLine))) {
                    break;
                }
                
                // If it's another import statement or Python code, add it to the code block
                if (/^(import|from|def |class |[a-zA-Z_]\w*\s*=|if |for |while |try:|with |async )/.test(nextLine)) {
                    newLines.push(lines[j]);
                    j++;
                } else if (nextLine === '') {
                    // Include empty lines within the code block
                    newLines.push(lines[j]);
                    j++;
                } else {
                    // Stop when we reach a non-code element
                    break;
                }
            }
            
            // Add code block end
            newLines.push('```');
            
            // Skip the processed lines
            i = j;
        } else {
            // Add the line as is
            newLines.push(line);
            i++;
        }
    }
    
    const newContent = newLines.join('\n');
    
    // Only write if content has changed
    if (newContent !== originalContent) {
        fs.writeFileSync(filePath, newContent);
        console.log(`  Fixed code blocks in ${filePath}`);
    } else {
        console.log(`  No changes needed for ${filePath}`);
    }
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
        fixSpecificFile(file);
    });
}

// Run the processing
const docsDir = path.join(__dirname, 'docs');
processDocsDirectory(docsDir);
console.log('Code block fixing completed.');
console.log('Now run: npx docusaurus build');