// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Front Matter',
      items: [
        'front-matter/index',
        'front-matter/intro',
        'front-matter/motivation',
        'front-matter/outcomes',
        'front-matter/roadmap'
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 1: Robotic Nervous System (ROS 2)',
      items: [
        'module-1/index',
        {
          type: 'category',
          label: 'Chapter 1: ROS 2 Architecture',
          items: [
            'module-1/chapter-1/index',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 2: Nodes, Topics, Services, and Actions',
          items: [
            'module-1/chapter-2/index',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 3: Python rclpy, URDF Modeling, and Package & Launch Workflows',
          items: [
            'module-1/chapter-3/index',
          ],
        },
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 2: Digital Twin (Gazebo & Unity)',
      items: [
        'module-2/index',
        {
          type: 'category',
          label: 'Chapter 1: Physics Simulation',
          items: [
            'module-2/chapter-1/index',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 2: Sensor Simulation',
          items: [
            'module-2/chapter-2/index',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 3: Unity Visualization & HRI',
          items: [
            'module-2/chapter-3/index',
          ],
        },
      ],
      collapsed: true,
    },
    {
      type: 'category',
      label: 'Module 3: AI-Robot Brain (NVIDIA Isaac)',
      items: [
        'module-3/index',
        {
          type: 'category',
          label: 'Chapter 1: Isaac Sim Setup',
          items: [
            'module-3/chapter-1/index',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 2: ROS Pipelines for Perception/VSLAM',
          items: [
            'module-3/chapter-2/index',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 3: Hardware-Accelerated Navigation',
          items: [
            'module-3/chapter-3/index',
          ],
        },
      ],
      collapsed: true,
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA)',
      items: [
        'module-4/index',
        {
          type: 'category',
          label: 'Chapter 1: LLMs in Robotics',
          items: [
            'module-4/chapter-1/index',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 2: Voice-to-Action Systems',
          items: [
            'module-4/chapter-2/index',
          ],
        },
        {
          type: 'category',
          label: 'Chapter 3: Multi-modal Perception',
          items: [
            'module-4/chapter-3/index',
          ],
        },
      ],
      collapsed: true,
    },
    {
      type: 'doc',
      id: 'capstone/index',
      label: 'Capstone: Autonomous Humanoid Robot',
    },
  ],
};

export default sidebars;