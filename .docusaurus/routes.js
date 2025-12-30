import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/physical-ai-textbook/__docusaurus/debug',
    component: ComponentCreator('/physical-ai-textbook/__docusaurus/debug', 'f48'),
    exact: true
  },
  {
    path: '/physical-ai-textbook/__docusaurus/debug/config',
    component: ComponentCreator('/physical-ai-textbook/__docusaurus/debug/config', '27c'),
    exact: true
  },
  {
    path: '/physical-ai-textbook/__docusaurus/debug/content',
    component: ComponentCreator('/physical-ai-textbook/__docusaurus/debug/content', '23a'),
    exact: true
  },
  {
    path: '/physical-ai-textbook/__docusaurus/debug/globalData',
    component: ComponentCreator('/physical-ai-textbook/__docusaurus/debug/globalData', 'ea6'),
    exact: true
  },
  {
    path: '/physical-ai-textbook/__docusaurus/debug/metadata',
    component: ComponentCreator('/physical-ai-textbook/__docusaurus/debug/metadata', '342'),
    exact: true
  },
  {
    path: '/physical-ai-textbook/__docusaurus/debug/registry',
    component: ComponentCreator('/physical-ai-textbook/__docusaurus/debug/registry', 'a35'),
    exact: true
  },
  {
    path: '/physical-ai-textbook/__docusaurus/debug/routes',
    component: ComponentCreator('/physical-ai-textbook/__docusaurus/debug/routes', 'eb3'),
    exact: true
  },
  {
    path: '/physical-ai-textbook/search',
    component: ComponentCreator('/physical-ai-textbook/search', '34e'),
    exact: true
  },
  {
    path: '/physical-ai-textbook/docs',
    component: ComponentCreator('/physical-ai-textbook/docs', '28b'),
    routes: [
      {
        path: '/physical-ai-textbook/docs',
        component: ComponentCreator('/physical-ai-textbook/docs', 'd4b'),
        routes: [
          {
            path: '/physical-ai-textbook/docs',
            component: ComponentCreator('/physical-ai-textbook/docs', 'a00'),
            routes: [
              {
                path: '/physical-ai-textbook/docs/capstone',
                component: ComponentCreator('/physical-ai-textbook/docs/capstone', 'cff'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-textbook/docs/front-matter',
                component: ComponentCreator('/physical-ai-textbook/docs/front-matter', 'c60'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-textbook/docs/front-matter',
                component: ComponentCreator('/physical-ai-textbook/docs/front-matter', 'a4e'),
                exact: true
              },
              {
                path: '/physical-ai-textbook/docs/front-matter/intro',
                component: ComponentCreator('/physical-ai-textbook/docs/front-matter/intro', 'df4'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-textbook/docs/front-matter/motivation',
                component: ComponentCreator('/physical-ai-textbook/docs/front-matter/motivation', '72d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-textbook/docs/front-matter/outcomes',
                component: ComponentCreator('/physical-ai-textbook/docs/front-matter/outcomes', '92d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-textbook/docs/front-matter/roadmap',
                component: ComponentCreator('/physical-ai-textbook/docs/front-matter/roadmap', '620'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-textbook/docs/module-1',
                component: ComponentCreator('/physical-ai-textbook/docs/module-1', '87c'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-textbook/docs/module-1/chapter-1',
                component: ComponentCreator('/physical-ai-textbook/docs/module-1/chapter-1', 'b3b'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-textbook/docs/module-1/chapter-2',
                component: ComponentCreator('/physical-ai-textbook/docs/module-1/chapter-2', 'c39'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-textbook/docs/module-1/chapter-3',
                component: ComponentCreator('/physical-ai-textbook/docs/module-1/chapter-3', 'c4d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-textbook/docs/module-2',
                component: ComponentCreator('/physical-ai-textbook/docs/module-2', 'ac8'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-textbook/docs/module-2/chapter-1',
                component: ComponentCreator('/physical-ai-textbook/docs/module-2/chapter-1', 'ece'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-textbook/docs/module-2/chapter-2',
                component: ComponentCreator('/physical-ai-textbook/docs/module-2/chapter-2', '612'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-textbook/docs/module-2/chapter-3',
                component: ComponentCreator('/physical-ai-textbook/docs/module-2/chapter-3', '463'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-textbook/docs/module-3',
                component: ComponentCreator('/physical-ai-textbook/docs/module-3', '494'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-textbook/docs/module-3/chapter-1',
                component: ComponentCreator('/physical-ai-textbook/docs/module-3/chapter-1', 'b6d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-textbook/docs/module-3/chapter-2',
                component: ComponentCreator('/physical-ai-textbook/docs/module-3/chapter-2', 'bce'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-textbook/docs/module-3/chapter-3',
                component: ComponentCreator('/physical-ai-textbook/docs/module-3/chapter-3', 'cde'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-textbook/docs/module-4',
                component: ComponentCreator('/physical-ai-textbook/docs/module-4', 'e2d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-textbook/docs/module-4/chapter-1',
                component: ComponentCreator('/physical-ai-textbook/docs/module-4/chapter-1', '19b'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-textbook/docs/module-4/chapter-2',
                component: ComponentCreator('/physical-ai-textbook/docs/module-4/chapter-2', '9a2'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-textbook/docs/module-4/chapter-3',
                component: ComponentCreator('/physical-ai-textbook/docs/module-4/chapter-3', 'ecd'),
                exact: true,
                sidebar: "tutorialSidebar"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '/physical-ai-textbook/docs',
    component: ComponentCreator('/physical-ai-textbook/docs', 'd5d'),
    routes: [
      {
        path: '/physical-ai-textbook/docs',
        component: ComponentCreator('/physical-ai-textbook/docs', '403'),
        routes: [
          {
            path: '/physical-ai-textbook/docs',
            component: ComponentCreator('/physical-ai-textbook/docs', 'a00'),
            routes: [
              {
                path: '/physical-ai-textbook/docs/capstone',
                component: ComponentCreator('/physical-ai-textbook/docs/capstone', 'cff'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-textbook/docs/front-matter',
                component: ComponentCreator('/physical-ai-textbook/docs/front-matter', 'c60'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-textbook/docs/front-matter',
                component: ComponentCreator('/physical-ai-textbook/docs/front-matter', 'a4e'),
                exact: true
              },
              {
                path: '/physical-ai-textbook/docs/front-matter/intro',
                component: ComponentCreator('/physical-ai-textbook/docs/front-matter/intro', 'df4'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-textbook/docs/front-matter/motivation',
                component: ComponentCreator('/physical-ai-textbook/docs/front-matter/motivation', '72d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-textbook/docs/front-matter/outcomes',
                component: ComponentCreator('/physical-ai-textbook/docs/front-matter/outcomes', '92d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-textbook/docs/front-matter/roadmap',
                component: ComponentCreator('/physical-ai-textbook/docs/front-matter/roadmap', '620'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-textbook/docs/module-1',
                component: ComponentCreator('/physical-ai-textbook/docs/module-1', '87c'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-textbook/docs/module-1/chapter-1',
                component: ComponentCreator('/physical-ai-textbook/docs/module-1/chapter-1', 'b3b'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-textbook/docs/module-1/chapter-2',
                component: ComponentCreator('/physical-ai-textbook/docs/module-1/chapter-2', 'c39'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-textbook/docs/module-1/chapter-3',
                component: ComponentCreator('/physical-ai-textbook/docs/module-1/chapter-3', 'c4d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-textbook/docs/module-2',
                component: ComponentCreator('/physical-ai-textbook/docs/module-2', 'ac8'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-textbook/docs/module-2/chapter-1',
                component: ComponentCreator('/physical-ai-textbook/docs/module-2/chapter-1', 'ece'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-textbook/docs/module-2/chapter-2',
                component: ComponentCreator('/physical-ai-textbook/docs/module-2/chapter-2', '612'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-textbook/docs/module-2/chapter-3',
                component: ComponentCreator('/physical-ai-textbook/docs/module-2/chapter-3', '463'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-textbook/docs/module-3',
                component: ComponentCreator('/physical-ai-textbook/docs/module-3', '494'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-textbook/docs/module-3/chapter-1',
                component: ComponentCreator('/physical-ai-textbook/docs/module-3/chapter-1', 'b6d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-textbook/docs/module-3/chapter-2',
                component: ComponentCreator('/physical-ai-textbook/docs/module-3/chapter-2', 'bce'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-textbook/docs/module-3/chapter-3',
                component: ComponentCreator('/physical-ai-textbook/docs/module-3/chapter-3', 'cde'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-textbook/docs/module-4',
                component: ComponentCreator('/physical-ai-textbook/docs/module-4', 'e2d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-textbook/docs/module-4/chapter-1',
                component: ComponentCreator('/physical-ai-textbook/docs/module-4/chapter-1', '19b'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-textbook/docs/module-4/chapter-2',
                component: ComponentCreator('/physical-ai-textbook/docs/module-4/chapter-2', '9a2'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/physical-ai-textbook/docs/module-4/chapter-3',
                component: ComponentCreator('/physical-ai-textbook/docs/module-4/chapter-3', 'ecd'),
                exact: true,
                sidebar: "tutorialSidebar"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
