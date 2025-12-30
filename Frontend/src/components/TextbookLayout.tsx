import React from 'react';
import clsx from 'clsx';
import styles from './TextbookLayout.module.css';

// Define the props type for the component
type TextbookLayoutProps = {
  children?: React.ReactNode;
  className?: string;
  learningObjectives?: string[];
  title?: string;
};

// Create the layout component
const TextbookLayout: React.FC<TextbookLayoutProps> = ({
  children,
  className,
  learningObjectives,
  title,
}) => {
  return (
    <div className={clsx('container', styles.textbookContainer, className)}>
      {title && <h1 className={styles.textbookTitle}>{title}</h1>}
      {learningObjectives && learningObjectives.length > 0 && (
        <div className={styles.learningObjectives}>
          <h4>Learning Objectives</h4>
          <ul>
            {learningObjectives.map((objective, index) => (
              <li key={index}>{objective}</li>
            ))}
          </ul>
        </div>
      )}
      <div className={styles.textbookContent}>{children}</div>
    </div>
  );
};

export default TextbookLayout;

// Add styles as a string - in a real project, you'd use a separate CSS module file
// For now, we'll create a placeholder for the module styles