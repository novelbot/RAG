"""
Simple test to verify test infrastructure works.
"""
import pytest


class TestSimple:
    """Simple test class."""
    
    def test_basic_assertion(self):
        """Test basic assertion."""
        assert 1 + 1 == 2
    
    def test_string_operations(self):
        """Test string operations."""
        text = "Hello World"
        assert text.lower() == "hello world"
        assert len(text) == 11
    
    def test_list_operations(self):
        """Test list operations."""
        items = [1, 2, 3, 4, 5]
        assert len(items) == 5
        assert sum(items) == 15
        assert max(items) == 5


@pytest.mark.unit
class TestWithMarkers:
    """Test with markers."""
    
    def test_marked_unit_test(self):
        """Test with unit marker."""
        assert True
    
    @pytest.mark.slow
    def test_slow_operation(self):
        """Test marked as slow."""
        import time
        time.sleep(0.1)  # 100ms
        assert True


class TestExceptionHandling:
    """Test exception handling."""
    
    def test_exception_raising(self):
        """Test exception is raised."""
        with pytest.raises(ValueError):
            raise ValueError("Test error")
    
    def test_exception_message(self):
        """Test exception message."""
        with pytest.raises(ValueError, match="Test error"):
            raise ValueError("Test error")


class TestFixtures:
    """Test fixtures."""
    
    @pytest.fixture
    def sample_data(self):
        """Sample data fixture."""
        return {"key": "value", "number": 42}
    
    def test_fixture_usage(self, sample_data):
        """Test using fixture."""
        assert sample_data["key"] == "value"
        assert sample_data["number"] == 42