// @ts-check

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.

 @type {import('@docusaurus/plugin-content-docs').SidebarsConfig}
 */
const sidebars = {
  // Manual sidebar structure for the Physical AI & Humanoid Robotics Course
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Module 1: Introduction to Physical AI',
      items: [
        'module-1/intro',
        'module-1/week1-2-foundations',
        'module-1/week1-2-physical-ai',
        'module-1/week1-2-humanoid-landscape',
        'module-1/week1-2-sensor-systems'
      ],
      link: {
        type: 'generated-index',
        title: 'Module 1: Introduction to Physical AI',
        description: 'Weeks 1-2: Foundations of Physical AI and embodied intelligence',
        slug: '/module-1',
      },
    },
    {
      type: 'category',
      label: 'Module 2: ROS 2 Fundamentals',
      items: [
        'module-2/intro',
        'module-2/week3-5-ros2-architecture',
        'module-2/week3-5-nodes-topics-services',
        'module-2/week3-5-building-ros2-packages',
        'module-2/week3-5-launch-files'
      ],
      link: {
        type: 'generated-index',
        title: 'Module 2: ROS 2 Fundamentals',
        description: 'Weeks 3-5: ROS 2 architecture and core concepts',
        slug: '/module-2',
      },
    },
    {
      type: 'category',
      label: 'Module 3: Robot Simulation and NVIDIA Isaac Platform',
      items: [
        'module-3/intro',
        'module-3/week6-7-gazebo-intro',
        'module-3/week6-7-gazebo-setup',
        'module-3/week6-7-urdf-sdf',
        'module-3/week6-7-physics-simulation',
        'module-3/week6-7-unity-visualization',
        'module-3/week8-10-isaac-sdk',
        'module-3/week8-10-isaac-sim',
        'module-3/week8-10-ai-perception',
        'module-3/week8-10-reinforcement-learning',
        'module-3/week8-10-sim-to-real'
      ],
      link: {
        type: 'generated-index',
        title: 'Module 3: Robot Simulation and NVIDIA Isaac Platform',
        description: 'Weeks 6-10: Simulation environments and NVIDIA Isaac platform',
        slug: '/module-3',
      },
    },
    {
      type: 'category',
      label: 'Module 4: Humanoid Robot Development and Conversational Robotics',
      items: [
        'module-4/intro',
        'module-4/week11-12-humanoid-intro',
        'module-4/week11-12-kinematics-dynamics',
        'module-4/week11-12-bipedal-locomotion',
        'module-4/week11-12-manipulation-grasping',
        'module-4/week11-12-hri-design',
        'module-4/week13-conversational-robotics',
        'module-4/week13-gpt-integration',
        'module-4/week13-speech-recognition',
        'module-4/week13-multi-modal-interaction',
        'module-4/week13-advanced-topics'
      ],
      link: {
        type: 'generated-index',
        title: 'Module 4: Humanoid Robot Development and Conversational Robotics',
        description: 'Weeks 11-13: Humanoid robotics and conversational AI',
        slug: '/module-4',
      },
    },
    {
      type: 'category',
      label: 'Assessments & Capstone',
      items: [
        'assessments/ros-project',
        'assessments/gazebo-implementation',
        'assessments/isaac-pipeline',
        'capstone'
      ],
      link: {
        type: 'generated-index',
        title: 'Assessments & Capstone',
        description: 'Projects and capstone for the course',
        slug: '/assessments',
      },
    },
  ],
};

export default sidebars;
