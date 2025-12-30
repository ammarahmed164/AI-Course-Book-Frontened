import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <div className="text--center">
          <Heading as="h1" className={clsx('hero__title', styles.mainTitle)}>
            {siteConfig.title}
          </Heading>
          <p className={clsx('hero__subtitle', styles.mainSubtitle)}>
            {siteConfig.tagline}
          </p>
        </div>
        <div className={styles.buttons}>
          <div className={styles.buttonGroup}>
            <Link
              className="button button--secondary button--lg rounded-pill margin-horiz--md"
              to="/docs/module-1/intro">
              Start Learning
            </Link>
            <Link
              className="button button--primary button--lg rounded-pill margin-horiz--md"
              to="/docs/module-1">
              Explore Modules
            </Link>
            <Link
              className="button button--info button--lg rounded-pill margin-horiz--md"
              to="/docs/syllabus">
              View Course Syllabus
            </Link>
          </div>
        </div>
      </div>
    </header>
  );
}

function CourseOverview() {
  return (
    <section className={styles.courseOverview}>
      <div className="container padding-horiz--md">
        <div className="row">
          <div className="col col--4">
            <div className="text--center padding-vert--md">
              <h3>Module 1</h3>
              <p>Introduction to Physical AI and embodied intelligence</p>
              <Link to="/docs/module-1" className="button button--outline button--primary button--sm">
                Learn More
              </Link>
            </div>
          </div>
          <div className="col col--4">
            <div className="text--center padding-vert--md">
              <h3>Module 2</h3>
              <p>ROS 2 Fundamentals and architecture</p>
              <Link to="/docs/module-2" className="button button--outline button--primary button--sm">
                Learn More
              </Link>
            </div>
          </div>
          <div className="col col--4">
            <div className="text--center padding-vert--md">
              <h3>Module 3</h3>
              <p>Robot Simulation and NVIDIA Isaac Platform</p>
              <Link to="/docs/module-3" className="button button--outline button--primary button--sm">
                Learn More
              </Link>
            </div>
          </div>
        </div>
        <div className="row padding-top--lg">
          <div className="col col--4 col--offset-2">
            <div className="text--center padding-vert--md">
              <h3>Module 4</h3>
              <p>Humanoid Robot Development and Conversational Robotics</p>
              <Link to="/docs/module-4" className="button button--outline button--primary button--sm">
                Learn More
              </Link>
            </div>
          </div>
          <div className="col col--4">
            <div className="text--center padding-vert--md">
              <h3>Capstone</h3>
              <p>Integrated project with conversational AI</p>
              <Link to="/docs/capstone" className="button button--outline button--primary button--sm">
                Start Project
              </Link>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Welcome to ${siteConfig.title}`}
      description="Physical AI & Humanoid Robotics Course - An AI-native textbook for embodied intelligence">
      <HomepageHeader />
      <main>
        <CourseOverview />
      </main>
    </Layout>
  );
}
