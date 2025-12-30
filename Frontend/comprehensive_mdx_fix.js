const fs = require('fs');
const path = require('path');

// Function to fix MDX compilation errors by properly formatting code blocks
function fixMDXFile(filePath) {
    console.log(`Processing ${filePath}...`);
    
    let content = fs.readFileSync(filePath, 'utf8');
    const originalContent = content;
    
    // Rule 1 & 2: Wrap code-like content in proper code blocks
    // Look for patterns that indicate code that should be in fenced blocks
    
    // Pattern for function declarations
    const funcPattern = /^(?!```)(function\s+\w+\s*\([^)]*\)|const\s+\w+\s*=|let\s+\w+\s*=|var\s+\w+\s*=|import\s+|export\s+)/gm;
    
    // Pattern for YAML/json/workflows that should be in code blocks
    const yamlPattern = /^(?!```)([a-z_]+:\s*|-\s+\w+:|version:|title:|description:|\w+:\s+\{|\w+:\s+\[)/gm;
    
    // Pattern for algorithmic/pseudocode patterns
    const algoPattern = /^(?!```)(step\s+\d+|algorithm:|pseudocode:|for\s+\w+\s+in|while\s*\(|if\s*\(|else:|try:|except:|def\s+\w+)/gm;
    
    // Process content to wrap appropriate code in fenced blocks
    let newContent = content;
    
    // First, let's handle function declarations and similar constructs
    newContent = newContent.replace(funcPattern, (match, offset) => {
        // Check if this is already inside a code block
        const contentUpToMatch = newContent.substring(0, offset);
        const codeBlockCount = (contentUpToMatch.match(/```/g) || []).length;
        
        // If even number of code blocks before this match, we're outside a code block
        if (codeBlockCount % 2 === 0) {
            return '```javascript\n' + match + '\n```';
        }
        return match;
    });
    
    // Handle YAML/JSON patterns
    newContent = newContent.replace(yamlPattern, (match, offset) => {
        const contentUpToMatch = newContent.substring(0, offset);
        const codeBlockCount = (contentUpToMatch.match(/```/g) || []).length;
        
        if (codeBlockCount % 2 === 0) {
            return '```yaml\n' + match + '\n```';
        }
        return match;
    });
    
    // Handle algorithmic patterns
    newContent = newContent.replace(algoPattern, (match, offset) => {
        const contentUpToMatch = newContent.substring(0, offset);
        const codeBlockCount = (contentUpToMatch.match(/```/g) || []).length;
        
        if (codeBlockCount % 2 === 0) {
            return '```text\n' + match + '\n```';
        }
        return match;
    });
    
    // Rule 3: Replace problematic inline expressions
    // Look for {variable} patterns that aren't in code blocks
    const lines = newContent.split('\n');
    const processedLines = [];
    
    let inCodeBlock = false;
    for (const line of lines) {
        if (line.trim().startsWith('```')) {
            inCodeBlock = !inCodeBlock;
            processedLines.push(line);
        } else if (!inCodeBlock) {
            // Replace standalone {something} with `something` outside of code blocks
            let processedLine = line.replace(/\{([^}]+)\}/g, '`$1`');
            processedLines.push(processedLine);
        } else {
            processedLines.push(line);
        }
    }
    
    newContent = processedLines.join('\n');
    
    // Only write if content has changed
    if (newContent !== originalContent) {
        fs.writeFileSync(filePath, newContent);
        console.log(`  Fixed MDX issues in ${filePath}`);
        return true;
    } else {
        console.log(`  No changes needed for ${filePath}`);
        return false;
    }
}

// Recursive function to process all markdown files in a directory
function processDirectory(dir) {
    const items = fs.readdirSync(dir);
    
    for (const item of items) {
        const fullPath = path.join(dir, item);
        const stat = fs.statSync(fullPath);
        
        if (stat.isDirectory()) {
            // Skip node_modules and other non-docs directories
            if (item !== 'node_modules' && item !== '.docusaurus' && item !== '.git' && item !== 'static') {
                processDirectory(fullPath);
            }
        } else if (path.extname(item) === '.md' || path.extname(item) === '.mdx') {
            // Only process files in the docs directory
            if (fullPath.includes('docs')) {
                fixMDXFile(fullPath);
            }
        }
    }
}

// Start processing from the docs directory
const docsDir = path.join(__dirname, 'docs');
if (fs.existsSync(docsDir)) {
    processDirectory(docsDir);
} else {
    console.log('Docs directory does not exist');
}

// Also process any md/mdx files in the root
const rootFiles = fs.readdirSync('.');
for (const file of rootFiles) {
    if ((path.extname(file) === '.md' || path.extname(file) === '.mdx') && 
        file !== 'package.json' && file !== 'package-lock.json' && file !== 'README.md') {
        fixMDXFile(path.join('.', file));
    }
}

console.log('MDX fixing completed. Now run: npx docusaurus clear && npx docusaurus start');