const fs = require('fs');
const path = require('path');

// Create the missing image
const imgDir = path.join(__dirname, 'static', 'img');
if (!fs.existsSync(imgDir)) {
    fs.mkdirSync(imgDir, { recursive: true });
}

// Create a simple 1x1 pixel PNG placeholder image
const pngHeader = Buffer.from('89504E470D0A1A0A0000000D49484452000000010000000108060000001F15C4890000000A49444154789C63000100000500010D0A2DB4D0180000000049454E44AE426082', 'hex');
const imagePath = path.join(imgDir, 'voice-command-arch.png');
if (!fs.existsSync(imagePath)) {
    fs.writeFileSync(imagePath, pngHeader);
    console.log('Created placeholder image: voice-command-arch.png');
} else {
    console.log('Image already exists: voice-command-arch.png');
}

console.log('Missing image created. Now run: npx docusaurus build');