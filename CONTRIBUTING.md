# Contributing to RAG Server

Thank you for your interest in contributing to the RAG Server project! We welcome contributions from the community and are grateful for your help in making this project better.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)

## Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow. Please be respectful and professional in all interactions.

## How to Contribute

### Types of Contributions

We welcome several types of contributions:

1. **Bug Reports**: Help us identify and fix issues
2. **Feature Requests**: Suggest new features or improvements
3. **Code Contributions**: Submit bug fixes or new features
4. **Documentation**: Improve existing documentation or add new content
5. **Testing**: Add test cases or improve test coverage
6. **Performance Optimization**: Enhance system performance

### Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your contribution
4. Make your changes
5. Test your changes
6. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.11+
- uv package manager
- Git

### Setup Steps

```bash
# Clone your fork
git clone https://github.com/your-username/rag-server.git
cd rag-server

# Set up virtual environment
uv sync --group dev

# Install pre-commit hooks
uv run pre-commit install

# Set up development environment
cp .env.example .env
# Edit .env with your configuration
```

### Running the Development Server

```bash
# Start development server
uv run main.py

# Or use CLI
uv run rag-cli serve --reload
```

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

- Line length: 88 characters (Black default)
- Use type hints for all functions and methods
- Use docstrings for all public functions, classes, and modules
- Follow naming conventions: snake_case for variables/functions, PascalCase for classes

### Code Formatting

We use automated tools for code formatting:

```bash
# Format code
uv run black src/
uv run isort src/

# Check formatting
uv run black --check src/
uv run isort --check-only src/
```

### Linting

```bash
# Run linting
uv run flake8 src/
uv run mypy src/
```

### Pre-commit Hooks

Pre-commit hooks are automatically installed and will run before each commit:

- Black (code formatting)
- isort (import sorting)
- flake8 (linting)
- mypy (type checking)

## Testing

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test file
uv run pytest tests/unit/test_config.py

# Run specific test
uv run pytest tests/unit/test_config.py::test_config_loading
```

### Writing Tests

- Write unit tests for all new functions and methods
- Write integration tests for complex workflows
- Use pytest fixtures for common test setup
- Aim for high test coverage (>90%)

#### Test Structure

```python
# tests/unit/test_example.py
import pytest
from unittest.mock import Mock, patch

from src.core.example import ExampleClass


class TestExampleClass:
    def test_example_method(self):
        """Test example method behavior"""
        instance = ExampleClass()
        result = instance.example_method("test_input")
        assert result == "expected_output"

    @patch('src.core.example.external_dependency')
    def test_with_mock(self, mock_dependency):
        """Test with mocked dependency"""
        mock_dependency.return_value = "mocked_value"
        instance = ExampleClass()
        result = instance.method_with_dependency()
        assert result == "expected_output"
```

## Documentation

### Code Documentation

- Use docstrings for all public functions, classes, and modules
- Follow Google-style docstrings
- Include parameter types and return types
- Provide examples for complex functions

#### Docstring Example

```python
def process_query(query: str, mode: str = "single", k: int = 5) -> Dict[str, Any]:
    """Process a RAG query and return results.

    Args:
        query: The user's query string
        mode: RAG mode, either "single" or "multi"
        k: Number of results to retrieve

    Returns:
        Dictionary containing query results with keys:
            - "answer": Generated answer string
            - "sources": List of source documents
            - "metadata": Additional metadata

    Raises:
        ValueError: If mode is not "single" or "multi"
        LLMError: If LLM service is unavailable

    Example:
        >>> result = process_query("What is AI?", mode="single", k=3)
        >>> print(result["answer"])
        "Artificial Intelligence is..."
    """
```

### API Documentation

- All API endpoints must be documented with OpenAPI/Swagger
- Include request/response examples
- Document all parameters and possible error codes

### README Updates

- Update README.md when adding new features
- Keep installation and usage instructions current
- Add examples for new functionality

## Pull Request Process

### Before Submitting

1. Ensure your code passes all tests
2. Update documentation if needed
3. Add tests for new functionality
4. Update CHANGELOG.md if applicable

### PR Title and Description

- Use clear, descriptive PR titles
- Include a detailed description of changes
- Reference related issues using #issue-number
- Include screenshots for UI changes

### PR Template

```markdown
## Description
Brief description of the changes made.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

### Review Process

1. All PRs require at least one approval
2. CI/CD checks must pass
3. Code coverage should not decrease
4. Breaking changes require additional review

## Issue Reporting

### Bug Reports

When reporting bugs, please include:

1. **Environment**: OS, Python version, dependencies
2. **Steps to Reproduce**: Clear steps to reproduce the issue
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Error Messages**: Full error messages and stack traces
6. **Configuration**: Relevant configuration files (remove sensitive data)

### Feature Requests

When requesting features, please include:

1. **Problem Statement**: What problem does this solve?
2. **Proposed Solution**: Describe your ideal solution
3. **Alternatives**: Other solutions you've considered
4. **Use Cases**: How would this feature be used?

### Security Issues

Please report security vulnerabilities privately by emailing team@ragserver.com. Do not create public issues for security problems.

## Development Workflow

### Branch Naming

- Feature branches: `feature/description-of-feature`
- Bug fixes: `fix/description-of-bug`
- Documentation: `docs/description-of-change`
- Performance: `perf/description-of-optimization`

### Commit Messages

Follow conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test changes
- `chore`: Maintenance tasks

Examples:
```
feat(llm): add support for Anthropic Claude models

fix(database): resolve connection pool timeout issue

docs(api): update query endpoint documentation
```

## Additional Resources

- [Project Architecture](docs/architecture.md)
- [API Documentation](docs/api.md)
- [Deployment Guide](docs/deployment.md)
- [Performance Tuning](docs/performance.md)

## Questions?

If you have questions about contributing, please:

1. Check existing issues and documentation
2. Ask in GitHub Discussions
3. Contact maintainers via email

Thank you for contributing to RAG Server!