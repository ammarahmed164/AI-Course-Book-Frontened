// @ts-check
// `@type` JSDoc annotations allow IDEs and type checkers to type-check this file
// even if they don't support TypeScript syntax.

import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Physical AI & Humanoid Robotics Textbook',
  tagline: 'An AI-native textbook for embodied intelligence and humanoid robotics',
  favicon: 'img/favicon.ico',

  // Set the production url of your site here
  url: 'https://your-organization.github.io',
  // Set the /<base> pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/physical-ai-textbook/',

  // GitHub Pages configuration
  organizationName: 'your-organization', // Usually your GitHub org/user name.
  projectName: 'physical-ai-textbook', // Usually your repo name.
  deploymentBranch: 'gh-pages', // Branch that GitHub Pages will deploy from.
  trailingSlash: false, // For compatibility with GitHub Pages

  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'ignore',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/your-organization/your-repo/edit/main/',
          showLastUpdateTime: true,
          showLastUpdateAuthor: true,
          exclude: ['**/node_modules/**', '**/my-website/**'],
          // Additional configuration to handle MDX issues would go here
        },
        blog: false, // Disable blog functionality
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themes: [
    // Add additional themes here if needed
  ],

  plugins: [
    // Add performance and SEO plugins
    [
      '@docusaurus/plugin-content-docs',
      {
        id: 'modules',
        path: 'docs',
        routeBasePath: 'docs',
        sidebarPath: './sidebars.js',
        exclude: ['**/node_modules/**', '**/my-website/**'],
      },
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/docusaurus-social-card.jpg',
      navbar: {
        title: 'Physical AI Textbook',
        logo: {
          alt: 'Physical AI & Humanoid Robotics Textbook Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Textbook',
          },
          {
            to: '/docs/front-matter/intro',
            label: 'Front Matter',
            position: 'left',
          },
          {
            href: 'https://github.com/facebook/docusaurus',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Textbook Modules',
            items: [
              {
                label: 'Module 1: Robotic Nervous System',
                to: '/docs/module-1/chapter-1',
              },
              {
                label: 'Module 2: Digital Twin',
                to: '/docs/module-2/chapter-1',
              },
              {
                label: 'Module 3: AI-Robot Brain',
                to: '/docs/module-3/chapter-1',
              },
              {
                label: 'Module 4: Vision-Language-Action',
                to: '/docs/module-4/chapter-1',
              },
            ],
          },
          {
            title: 'Resources',
            items: [
              {
                label: 'Capstone Project',
                to: '/docs/capstone',
              },
              {
                label: 'Learning Objectives',
                to: '/docs/front-matter/outcomes',
              },
              {
                label: 'Weekly Roadmap',
                to: '/docs/front-matter/roadmap',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'Stack Overflow',
                href: 'https://stackoverflow.com/questions/tagged/docusaurus',
              },
              {
                label: 'Discord',
                href: 'https://discordapp.com/invite/docusaurus',
              },
              {
                label: 'Twitter',
                href: 'https://twitter.com/docusaurus',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Textbook. Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
        additionalLanguages: ['python', 'bash', 'docker', 'json'],
      },
      algolia: {
        // The application ID provided by Algolia
        appId: 'YOUR_APP_ID',

        // Public API key: it is safe to commit it
        apiKey: 'YOUR_SEARCH_API_KEY',

        indexName: 'your-index-name',

        // Optional: see doc section below
        contextualSearch: true,

        // Optional: Specify domains where the navigation should occur through window.location instead on history.push. Useful when our Algolia config crawls multiple documentation sites and we want to navigate with window.location.href to them.
        externalUrlRegex: 'external\\.com|domain\\.com',

        // Optional: Replace parts of the item URLs from Algolia. Useful when using the same search index for multiple deployments using a different baseUrl. You can use regexp or string in the `from` param. For example: localhost:3000 vs myCompany.com/docs
        replaceSearchResultPathname: {
          from: '/docs/', // or as RegExp: /\/docs\//
          to: '/docs/',
        },

        // Optional: Algolia search parameters
        searchParameters: {},

        // Optional: path for search page that enabled by default (`false` to disable it)
        searchPagePath: 'search',
      },
      // Configuration for the textbook chatbot
      textbookChatbot: {
        enabled: true,
        apiUrl: process.env.CHATBOT_API_URL || 'http://localhost:8000/v1',
        maxTokens: 1000,
        temperature: 0.3,
      },
    }),
};

export default config;