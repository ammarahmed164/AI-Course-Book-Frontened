const fs = require('fs');
const path = require('path');

// Function to fix MDX compilation errors in a markdown file
function fixMDXCompilationErrors(filePath) {
    console.log(`Processing ${filePath}...`);
    
    let content = fs.readFileSync(filePath, 'utf8');
    const originalContent = content;
    
    // Step 1: Replace problematic inline expressions with backticks
    // Match {something} that are not inside code blocks
    const lines = content.split('\n');
    const processedLines = [];
    let inCodeBlock = false;
    
    for (const line of lines) {
        if (line.trim().startsWith('```')) {
            inCodeBlock = !inCodeBlock;
            processedLines.push(line);
            continue;
        }
        
        if (!inCodeBlock) {
            // Replace standalone {variable} with `variable` outside of code blocks
            let processedLine = line;
            // Replace inline expressions like {foo} with `foo`
            processedLine = processedLine.replace(/\{([^{}]+)\}(?!\s*:)/g, '`$1`');
            
            // Handle any remaining unmatched curly braces
            processedLine = processedLine.replace(/\{(?=\s*(?:\n|$))/g, '`{`');
            processedLine = processedLine.replace(/(?<=^|\s)\}(?=\s*(?:\n|$))/g, '`}`');
            
            processedLines.push(processedLine);
        } else {
            processedLines.push(line);
        }
    }
    
    content = processedLines.join('\n');
    
    // Step 2: Identify and properly wrap code that looks like JavaScript/Python
    const lines2 = content.split('\n');
    const processedLines2 = [];
    inCodeBlock = false;
    
    for (let i = 0; i < lines2.length; i++) {
        const line = lines2[i];
        
        if (line.trim().startsWith('```')) {
            inCodeBlock = !inCodeBlock;
            processedLines2.push(line);
            continue;
        }
        
        if (!inCodeBlock) {
            // Check if line looks like JavaScript/Python code that should be in a code block
            const trimmedLine = line.trim();
            
            // Patterns that indicate code that should be in fenced blocks
            const codePatterns = [
                /^\s*function\s+/,
                /^\s*(const|let|var)\s+[a-zA-Z_$]/,
                /^\s*import\s+/,
                /^\s*export\s+/,
                /^\s*class\s+[a-zA-Z_$]/,
                /^\s*if\s*\(/,
                /^\s*else\s*:/,
                /^\s*for\s*\(/,
                /^\s*while\s*\(/,
                /^\s*try\s*:/,
                /^\s*except\s+/,
                /^\s*finally\s*:/,
                /^\s*with\s+/,
                /^\s*def\s+[a-zA-Z_$]/,
                /^\s*[a-zA-Z_$][a-zA-Z0-9_$]*\s*=\s*/,  // Variable assignments
                /^\s*[a-zA-Z_$][a-zA-Z0-9_$]*\s*\(/,   // Function calls
                /^\s*{\s*[^}]*\s*}/,                   // Object literals
                /^\s*\[\s*[^\]]*\s*\]/,                // Array literals
                /^\s*"[^"]*"/,                         // String literals
                /^\s*'[^']*'/,                         // Single quote strings
                /^\s*`[^`]*`/,                         // Template literals
                /^\s*\/\//,                            // Comments
                /^\s*\/\*/,                            // Block comment start
                /^\s*\*\/$/,                           // Block comment end
                /^\s*#.*$/,                            // Python/Ruby-style comments
            ];
            
            // Check if this line matches any of the code patterns
            const isCodeLike = codePatterns.some(pattern => pattern.test(trimmedLine));
            
            if (isCodeLike) {
                // Check if the previous line is not already a code block start
                const prevLine = i > 0 ? processedLines2[processedLines2.length - 1] : '';
                if (!prevLine.trim().startsWith('```')) {
                    // Determine language based on content
                    let lang = 'text'; // default
                    if (/import|export|function|const|let|var/.test(trimmedLine)) {
                        lang = 'javascript';
                    } else if (/def |class |import |from |if:|for |while |try:/.test(trimmedLine)) {
                        lang = 'python';
                    } else if (/{.*}|".*"|true|false|null/.test(trimmedLine)) {
                        lang = 'json';
                    } else if (/^[a-z_]+:/.test(trimmedLine) || /-\s+\w+:/.test(trimmedLine)) {
                        lang = 'yaml';
                    }
                    
                    processedLines2.push(`\`\`\`${lang}`);
                }
                
                processedLines2.push(line);
                
                // Look ahead for consecutive code-like lines
                let j = i + 1;
                while (j < lines2.length) {
                    const nextLine = lines2[j].trim();
                    
                    // Stop if we hit a markdown element or empty line followed by non-code
                    if (nextLine.startsWith('#') || nextLine.startsWith('>') || 
                        nextLine.startsWith('- ') || nextLine.startsWith('* ') || 
                        nextLine.startsWith('1. ') || nextLine.startsWith('|') ||
                        nextLine === '') {
                        
                        // Check if next few lines are also code-like
                        let isNextCodeLike = false;
                        let k = j + 1;
                        while (k < lines2.length && lines2[k].trim() === '') k++;
                        if (k < lines2.length) {
                            const nextNonEmptyLine = lines2[k].trim();
                            isNextCodeLike = codePatterns.some(pattern => pattern.test(nextNonEmptyLine));
                        }
                        
                        if (!isNextCodeLike) {
                            break;
                        }
                    }
                    
                    const nextLineFull = lines2[j];
                    const nextLineTrimmed = nextLineFull.trim();
                    
                    if (codePatterns.some(pattern => pattern.test(nextLineTrimmed))) {
                        processedLines2.push(nextLineFull);
                        j++;
                    } else if (nextLineTrimmed === '') {
                        // Include empty lines within code blocks
                        processedLines2.push(nextLineFull);
                        j++;
                    } else {
                        // Not code-like, so end the code block
                        break;
                    }
                }
                
                // Add closing code block
                processedLines2.push('```');
                i = j - 1; // Adjust index since we've processed additional lines
            } else {
                // Regular markdown line
                processedLines2.push(line);
            }
        } else {
            // Inside a code block, just add the line
            processedLines2.push(line);
        }
    }
    
    content = processedLines2.join('\n');
    
    // Step 3: Fix image references that don't exist
    const imagePattern = /!\[([^\]]*)\]\(([^)]+)\)/g;
    content = content.replace(imagePattern, (match, alt, imagePath) => {
        // Check if the image file exists
        const fullPath = path.join(__dirname, 'static', imagePath);
        if (!fs.existsSync(fullPath)) {
            // Image doesn't exist, replace with placeholder text
            return `<!-- Image \`${imagePath}\` referenced in this document does not exist. -->\n> **Note:** Image "${alt}" (${imagePath}) is referenced here but does not exist in the static assets.\n\n`;
        }
        return match; // Keep the original if the image exists
    });
    
    // Only write if content has changed
    if (content !== originalContent) {
        fs.writeFileSync(filePath, content);
        console.log(`  Fixed MDX compilation errors in ${filePath}`);
        return true;
    } else {
        console.log(`  No changes needed for ${filePath}`);
        return false;
    }
}

// Process all markdown files in the docs directory
function processDocsDirectory(dirPath) {
    const items = fs.readdirSync(dirPath);
    
    for (const item of items) {
        const fullPath = path.join(dirPath, item);
        const stat = fs.statSync(fullPath);
        
        if (stat.isDirectory()) {
            // Skip node_modules and other non-docs directories
            if (item !== 'node_modules' && item !== '.docusaurus' && item !== '.git' && item !== 'static' && item !== 'src') {
                processDocsDirectory(fullPath);
            }
        } else if (path.extname(item) === '.md' || path.extname(item) === '.mdx') {
            fixMDXCompilationErrors(fullPath);
        }
    }
}

// Start processing from the docs directory
const docsDir = path.join(__dirname, 'docs');
if (fs.existsSync(docsDir)) {
    processDocsDirectory(docsDir);
} else {
    console.log('Docs directory does not exist');
}

// Also process any md/mdx files in the root
const rootFiles = fs.readdirSync('.');
for (const file of rootFiles) {
    if ((path.extname(file) === '.md' || path.extname(file) === '.mdx') &&
        file !== 'package.json' && file !== 'package-lock.json' && file !== 'README.md') {
        const filePath = path.join('.', file);
        if (fs.statSync(filePath).isFile()) {
            fixMDXCompilationErrors(filePath);
        }
    }
}

console.log('All MDX compilation errors have been fixed. Run: npx docusaurus clear && npx docusaurus start');