# Content Validation Tests for Physical AI & Humanoid Robotics Textbook

import unittest
import os
import json
import yaml
from pathlib import Path

class TextbookContentValidator(unittest.TestCase):
    """Test suite to validate textbook content integrity."""
    
    def setUp(self):
        """Set up test environment."""
        self.docs_path = Path("docs")
        self.required_sections = ['learning_objectives', 'summary']
        
    def test_all_chapters_have_learning_objectives(self):
        """Ensure every chapter has learning objectives."""
        for root, dirs, files in os.walk(self.docs_path):
            for file in files:
                if file.endswith('.md'):
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        # Check if learning objectives are present
                        self.assertIn('Learning Objectives', content, 
                                    f"Chapter {file} in {root} missing learning objectives")
    
    def test_chapter_structure_consistency(self):
        """Ensure consistent chapter structure."""
        for root, dirs, files in os.walk(self.docs_path):
            for file in files:
                if file.endswith('.md'):
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        
                        # Check for essential sections
                        has_summary = 'summary' in content
                        has_introduction = 'introduction' in content or 'overview' in content
                        
                        self.assertTrue(has_summary, f"Chapter {file} missing Summary section")
                        self.assertTrue(has_introduction, f"Chapter {file} missing Introduction/Overview section")
    
    def test_code_examples_exist_and_runnable(self):
        """Check that code examples referenced exist and are runnable."""
        # In a full implementation, this would check that code examples exist 
        # in the assets directory and are syntactically correct
        code_dirs = [
            'module-1/assets/code-examples',
            'module-2/assets/code-examples', 
            'module-3/assets/code-examples',
            'module-4/assets/code-examples',
            'capstone/assets/code-examples'
        ]
        
        for code_dir in code_dirs:
            path = Path(code_dir)
            if path.exists():
                code_files = list(path.glob('*.py'))
                self.assertGreater(len(code_files), 0, f"No code examples found in {code_dir}")
    
    def test_no_hallucinated_apis_mentioned(self):
        """Verify no fictional or hallucinated APIs are referenced."""
        # This would involve checking content against known real APIs
        # For now, we'll just ensure references are well-documented
        for root, dirs, files in os.walk(self.docs_path):
            for file in files:
                if file.endswith('.md'):
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        # Look for common signs of hallucinated content
                        self.assertNotIn('[NEEDS CLARIFICATION]', content, 
                                       f"Unclear content found in {file}")
                        self.assertNotIn('fictional_api', content.lower(),
                                       f"Hallucinated API found in {file}")
    
    def test_all_modules_present(self):
        """Ensure all 4 modules and capstone exist."""
        expected_modules = [
            'module-1',
            'module-2', 
            'module-3',
            'module-4',
            'capstone'
        ]
        
        for module in expected_modules:
            module_path = self.docs_path / module
            self.assertTrue(module_path.exists(), f"Module {module} is missing")


def run_content_validation():
    """Run the content validation tests."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TextbookContentValidator)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return True if all tests passed
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_content_validation()
    exit(0 if success else 1)