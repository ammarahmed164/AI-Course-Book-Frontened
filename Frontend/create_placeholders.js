const fs = require('fs');
const path = require('path');

// Create a simple 1x1 pixel PNG placeholder image
const pngHeader = Buffer.from('89504E470D0A1A0A0000000D49484452000000010000000108060000001F15C4890000000A49444154789C63000100000500010D0A2DB4D0180000000049454E44AE426082', 'hex');

// Create the static/img directory if it doesn't exist
const imgDir = path.join(__dirname, 'static', 'img');
if (!fs.existsSync(imgDir)) {
    fs.mkdirSync(imgDir, { recursive: true });
}

// Create placeholder images
const images = [
    'humanoid-arch.png',
    'ros2-architecture.png',
    'communication-patterns.png',
    'package-structure.png',
    'gazebo-environment.png',
    'lidar-simulation.png',
    'unity-robotics.png',
    'isaac-sim-arch.png',
    'perception-pipeline.png'
];

images.forEach(image => {
    const imagePath = path.join(imgDir, image);
    if (!fs.existsSync(imagePath)) {
        fs.writeFileSync(imagePath, pngHeader);
        console.log(`Created placeholder image: ${image}`);
    } else {
        console.log(`Image already exists: ${image}`);
    }
});

console.log('All placeholder images processed.');