"""
Text Cleaning and Normalization System.

This module provides comprehensive text preprocessing capabilities including
cleaning unwanted characters, handling encoding issues, normalizing whitespace,
and standardizing text format for different document types.
"""

import re
import html
import unicodedata
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Union, Pattern
from urllib.parse import urlparse
import chardet

from src.core.logging import LoggerMixin
from .exceptions import CleaningError, UnsupportedEncodingError, TextValidationError


class CleaningMode(Enum):
    """Text cleaning modes for different document types."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"


class NormalizationForm(Enum):
    """Unicode normalization forms."""
    NFC = "NFC"  # Canonical decomposition, then canonical composition
    NFD = "NFD"  # Canonical decomposition
    NFKC = "NFKC"  # Compatibility decomposition, then canonical composition
    NFKD = "NFKD"  # Compatibility decomposition


@dataclass
class CleaningRule:
    """Configuration for a text cleaning rule."""
    name: str
    pattern: Union[str, Pattern]
    replacement: str
    enabled: bool = True
    description: Optional[str] = None
    priority: int = 0  # Higher priority rules are applied first
    
    def __post_init__(self):
        """Compile pattern if it's a string."""
        if isinstance(self.pattern, str):
            try:
                self.pattern = re.compile(self.pattern, re.MULTILINE | re.DOTALL)
            except re.error as e:
                raise CleaningError(f"Invalid regex pattern '{self.pattern}': {e}")


@dataclass
class CleaningConfig:
    """Configuration for text cleaning operations."""
    
    # Basic settings
    mode: CleaningMode = CleaningMode.STANDARD
    preserve_structure: bool = True
    preserve_urls: bool = False
    preserve_emails: bool = False
    preserve_numbers: bool = True
    
    # Encoding settings
    auto_detect_encoding: bool = True
    target_encoding: str = "utf-8"
    handle_encoding_errors: str = "replace"  # 'replace', 'ignore', 'strict'
    
    # Normalization settings
    unicode_normalization: Optional[NormalizationForm] = NormalizationForm.NFC
    normalize_whitespace: bool = True
    normalize_quotes: bool = True
    normalize_dashes: bool = True
    normalize_case: Optional[str] = None  # 'lower', 'upper', 'title', None
    
    # Content filtering
    remove_html_tags: bool = True
    remove_xml_tags: bool = False
    remove_markdown: bool = False
    remove_urls: bool = False
    remove_emails: bool = False
    remove_phone_numbers: bool = False
    remove_special_chars: bool = False
    remove_extra_newlines: bool = True
    remove_leading_trailing_whitespace: bool = True
    
    # Content replacement
    convert_html_entities: bool = True
    convert_unicode_quotes: bool = True
    convert_unicode_dashes: bool = True
    replace_tabs_with_spaces: bool = True
    tab_width: int = 4
    
    # Advanced settings
    min_line_length: int = 0
    max_line_length: int = 0  # 0 means no limit
    join_hyphenated_words: bool = False
    fix_encoding_issues: bool = True
    
    # Custom rules
    custom_rules: List[CleaningRule] = field(default_factory=list)
    
    # Document type specific settings
    document_type: Optional[str] = None  # 'pdf', 'html', 'markdown', 'plain'


@dataclass
class CleaningResult:
    """Result of text cleaning operation."""
    original_text: str
    cleaned_text: str
    encoding_detected: Optional[str] = None
    encoding_confidence: Optional[float] = None
    rules_applied: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    def get_reduction_ratio(self) -> float:
        """Calculate text reduction ratio."""
        if not self.original_text:
            return 0.0
        return 1.0 - (len(self.cleaned_text) / len(self.original_text))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "original_length": len(self.original_text),
            "cleaned_length": len(self.cleaned_text),
            "reduction_ratio": self.get_reduction_ratio(),
            "encoding_detected": self.encoding_detected,
            "encoding_confidence": self.encoding_confidence,
            "rules_applied": self.rules_applied,
            "warnings": self.warnings,
            "statistics": self.statistics
        }


class TextCleaner(LoggerMixin):
    """
    Comprehensive text cleaning and normalization system.
    
    Provides configurable text preprocessing capabilities for different
    document types with support for encoding detection, content filtering,
    normalization, and custom cleaning rules.
    """
    
    def __init__(self, config: Optional[CleaningConfig] = None):
        """
        Initialize the text cleaner.
        
        Args:
            config: Cleaning configuration
        """
        self.config = config or CleaningConfig()
        self._compiled_rules: List[CleaningRule] = []
        self._initialize_rules()
        
        self.logger.info(f"Text cleaner initialized with mode: {self.config.mode.value}")
    
    def clean_text(self, text: Union[str, bytes], **kwargs) -> CleaningResult:
        """
        Clean and normalize text according to configuration.
        
        Args:
            text: Text to clean (string or bytes)
            **kwargs: Override configuration parameters
            
        Returns:
            CleaningResult with cleaned text and metadata
        """
        try:
            # Create working config with overrides
            config = self._create_working_config(**kwargs)
            
            # Store original text
            original_text = text if isinstance(text, str) else ""
            
            # Handle encoding if input is bytes
            if isinstance(text, bytes):
                text, encoding_info = self._handle_encoding(text, config)
                original_text = text
            else:
                encoding_info = {"detected": None, "confidence": None}
            
            # Initialize result
            result = CleaningResult(
                original_text=original_text,
                cleaned_text=text,
                encoding_detected=encoding_info["detected"],
                encoding_confidence=encoding_info["confidence"]
            )
            
            # Validate input
            if not text or not text.strip():
                result.warnings.append("Input text is empty or contains only whitespace")
                return result
            
            # Apply cleaning pipeline
            cleaned_text = self._apply_cleaning_pipeline(text, config, result)
            result.cleaned_text = cleaned_text
            
            # Generate statistics
            result.statistics = self._calculate_statistics(original_text, cleaned_text)
            
            self.logger.debug(f"Text cleaning completed: {len(original_text)} -> {len(cleaned_text)} characters")
            return result
            
        except Exception as e:
            self.logger.error(f"Text cleaning failed: {e}")
            raise CleaningError(f"Text cleaning failed: {e}")
    
    def clean_file(self, file_path: Union[str, Path], **kwargs) -> CleaningResult:
        """
        Clean text from a file.
        
        Args:
            file_path: Path to the file
            **kwargs: Override configuration parameters
            
        Returns:
            CleaningResult with cleaned text and metadata
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise CleaningError(f"File does not exist: {file_path}")
            
            # Read file content
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # Clean the content
            result = self.clean_text(content, **kwargs)
            result.statistics["source_file"] = str(file_path)
            result.statistics["file_size"] = file_path.stat().st_size
            
            return result
            
        except Exception as e:
            self.logger.error(f"File cleaning failed for {file_path}: {e}")
            raise CleaningError(f"File cleaning failed for {file_path}: {e}")
    
    def add_custom_rule(self, rule: CleaningRule) -> None:
        """Add a custom cleaning rule."""
        self.config.custom_rules.append(rule)
        self._initialize_rules()
        self.logger.info(f"Added custom cleaning rule: {rule.name}")
    
    def remove_custom_rule(self, rule_name: str) -> bool:
        """Remove a custom cleaning rule by name."""
        for i, rule in enumerate(self.config.custom_rules):
            if rule.name == rule_name:
                del self.config.custom_rules[i]
                self._initialize_rules()
                self.logger.info(f"Removed custom cleaning rule: {rule_name}")
                return True
        return False
    
    def preview_cleaning(self, text: str, max_length: int = 500) -> Dict[str, str]:
        """
        Preview cleaning results on a text sample.
        
        Args:
            text: Text to preview
            max_length: Maximum length of preview text
            
        Returns:
            Dictionary with before and after text samples
        """
        sample = text[:max_length] if len(text) > max_length else text
        result = self.clean_text(sample)
        
        return {
            "original": sample,
            "cleaned": result.cleaned_text,
            "rules_applied": result.rules_applied,
            "reduction_ratio": result.get_reduction_ratio()
        }
    
    def validate_text(self, text: str) -> Dict[str, Any]:
        """
        Validate text for common issues.
        
        Args:
            text: Text to validate
            
        Returns:
            Validation results
        """
        issues = []
        
        # Check for common encoding issues
        if '�' in text:
            issues.append("Contains replacement characters (encoding issues)")
        
        # Check for excessive whitespace
        if re.search(r'\s{10,}', text):
            issues.append("Contains excessive whitespace")
        
        # Check for control characters
        control_chars = [c for c in text if unicodedata.category(c).startswith('C')]
        if control_chars:
            issues.append(f"Contains {len(control_chars)} control characters")
        
        # Check for mixed scripts (potential encoding issues)
        scripts = set(unicodedata.name(c, '').split()[0] for c in text if c.isalpha())
        if len(scripts) > 3:
            issues.append("Contains multiple writing scripts")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "length": len(text),
            "encoding": "utf-8" if text.isascii() else "unicode"
        }
    
    def _create_working_config(self, **kwargs) -> CleaningConfig:
        """Create a working configuration with overrides."""
        # Create a copy of the current config
        import copy
        config = copy.deepcopy(self.config)
        
        # Apply overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    def _handle_encoding(self, content: bytes, config: CleaningConfig) -> tuple[str, Dict[str, Any]]:
        """Handle encoding detection and conversion."""
        encoding_info = {"detected": None, "confidence": None}
        
        try:
            if config.auto_detect_encoding:
                # Detect encoding
                detection = chardet.detect(content)
                encoding_info["detected"] = detection.get("encoding")
                encoding_info["confidence"] = detection.get("confidence")
                
                if encoding_info["detected"]:
                    text = content.decode(encoding_info["detected"], errors=config.handle_encoding_errors)
                else:
                    # Fallback to target encoding
                    text = content.decode(config.target_encoding, errors=config.handle_encoding_errors)
            else:
                text = content.decode(config.target_encoding, errors=config.handle_encoding_errors)
            
            return text, encoding_info
            
        except Exception as e:
            raise UnsupportedEncodingError(f"Encoding handling failed: {e}")
    
    def _apply_cleaning_pipeline(self, text: str, config: CleaningConfig, result: CleaningResult) -> str:
        """Apply the complete cleaning pipeline."""
        cleaned = text
        
        # 1. Handle encoding issues
        if config.fix_encoding_issues:
            cleaned = self._fix_encoding_issues(cleaned, result)
        
        # 2. Convert HTML entities
        if config.convert_html_entities:
            cleaned = self._convert_html_entities(cleaned, result)
        
        # 3. Remove/convert content based on config
        if config.remove_html_tags:
            cleaned = self._remove_html_tags(cleaned, result)
        
        if config.remove_xml_tags:
            cleaned = self._remove_xml_tags(cleaned, result)
        
        if config.remove_markdown:
            cleaned = self._remove_markdown(cleaned, result)
        
        if config.remove_urls and not config.preserve_urls:
            cleaned = self._remove_urls(cleaned, result)
        
        if config.remove_emails and not config.preserve_emails:
            cleaned = self._remove_emails(cleaned, result)
        
        if config.remove_phone_numbers:
            cleaned = self._remove_phone_numbers(cleaned, result)
        
        # 4. Unicode normalization
        if config.unicode_normalization:
            cleaned = self._normalize_unicode(cleaned, config.unicode_normalization, result)
        
        # 5. Character conversions
        if config.convert_unicode_quotes:
            cleaned = self._convert_unicode_quotes(cleaned, result)
        
        if config.convert_unicode_dashes:
            cleaned = self._convert_unicode_dashes(cleaned, result)
        
        if config.replace_tabs_with_spaces:
            cleaned = self._replace_tabs_with_spaces(cleaned, config.tab_width, result)
        
        # 6. Whitespace normalization
        if config.normalize_whitespace:
            cleaned = self._normalize_whitespace(cleaned, result)
        
        if config.remove_extra_newlines:
            cleaned = self._remove_extra_newlines(cleaned, result)
        
        if config.remove_leading_trailing_whitespace:
            cleaned = cleaned.strip()
            result.rules_applied.append("remove_leading_trailing_whitespace")
        
        # 7. Apply custom rules
        cleaned = self._apply_custom_rules(cleaned, config, result)
        
        # 8. Case normalization (after other processing)
        if config.normalize_case:
            cleaned = self._normalize_case(cleaned, config.normalize_case, result)
        
        # 9. Line length handling
        if config.min_line_length > 0 or config.max_line_length > 0:
            cleaned = self._handle_line_lengths(cleaned, config, result)
        
        # 10. Final cleanup
        if config.join_hyphenated_words:
            cleaned = self._join_hyphenated_words(cleaned, result)
        
        return cleaned
    
    def _initialize_rules(self) -> None:
        """Initialize compiled cleaning rules based on configuration."""
        self._compiled_rules.clear()
        
        # Add built-in rules based on mode
        built_in_rules = self._get_built_in_rules()
        self._compiled_rules.extend(built_in_rules)
        
        # Add custom rules
        self._compiled_rules.extend(self.config.custom_rules)
        
        # Sort by priority (higher first)
        self._compiled_rules.sort(key=lambda r: r.priority, reverse=True)
    
    def _get_built_in_rules(self) -> List[CleaningRule]:
        """Get built-in cleaning rules based on mode."""
        rules = []
        
        if self.config.mode in [CleaningMode.STANDARD, CleaningMode.AGGRESSIVE]:
            # Common problematic patterns
            rules.append(CleaningRule(
                name="remove_zero_width_chars",
                pattern=r"[\u200b\u200c\u200d\ufeff]",
                replacement="",
                description="Remove zero-width characters"
            ))
            
            rules.append(CleaningRule(
                name="normalize_line_breaks",
                pattern=r"\r\n|\r",
                replacement="\n",
                description="Normalize line breaks to LF"
            ))
        
        if self.config.mode == CleaningMode.AGGRESSIVE:
            # More aggressive cleaning
            rules.append(CleaningRule(
                name="remove_extra_punctuation",
                pattern=r"[.]{3,}",
                replacement="...",
                description="Normalize excessive ellipsis"
            ))
            
            rules.append(CleaningRule(
                name="remove_excessive_exclamation",
                pattern=r"[!]{2,}",
                replacement="!",
                description="Normalize excessive exclamation marks"
            ))
        
        return rules
    
    def _fix_encoding_issues(self, text: str, result: CleaningResult) -> str:
        """Fix common encoding issues."""
        # Common encoding replacements
        replacements = {
            'â€™': "'",  # Right single quotation mark
            'â€œ': '"',  # Left double quotation mark  
            'â€\x9d': '"',  # Right double quotation mark
            'â€"': '—',  # Em dash
            'â€"': '–',  # En dash
            'Â': '',     # Non-breaking space artifacts
            'â\x80\x99': "'",  # Another apostrophe variant
            'â\x80\x9c': '"',  # Another left quote variant
            'â\x80\x9d': '"',  # Another right quote variant
        }
        
        original = text
        for wrong, right in replacements.items():
            text = text.replace(wrong, right)
        
        if text != original:
            result.rules_applied.append("fix_encoding_issues")
        
        return text
    
    def _convert_html_entities(self, text: str, result: CleaningResult) -> str:
        """Convert HTML entities to their Unicode equivalents."""
        original = text
        text = html.unescape(text)
        
        if text != original:
            result.rules_applied.append("convert_html_entities")
        
        return text
    
    def _remove_html_tags(self, text: str, result: CleaningResult) -> str:
        """Remove HTML tags while preserving content."""
        original = text
        
        # Remove script and style elements completely
        text = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove HTML comments
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
        
        # Remove all other HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        if text != original:
            result.rules_applied.append("remove_html_tags")
        
        return text
    
    def _remove_xml_tags(self, text: str, result: CleaningResult) -> str:
        """Remove XML tags while preserving content."""
        original = text
        text = re.sub(r'<[^>]+>', '', text)
        
        if text != original:
            result.rules_applied.append("remove_xml_tags")
        
        return text
    
    def _remove_markdown(self, text: str, result: CleaningResult) -> str:
        """Remove basic Markdown formatting."""
        original = text
        
        # Remove headers
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
        
        # Remove bold and italic
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'__(.+?)__', r'\1', text)
        text = re.sub(r'_(.+?)_', r'\1', text)
        
        # Remove links but keep link text
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        
        # Remove code blocks
        text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        if text != original:
            result.rules_applied.append("remove_markdown")
        
        return text
    
    def _remove_urls(self, text: str, result: CleaningResult) -> str:
        """Remove URLs from text."""
        original = text
        
        # URL pattern
        url_pattern = r'https?://[^\s<>"{}|\\^`[\]]+|www\.[^\s<>"{}|\\^`[\]]+|[^\s<>"{}|\\^`[\]]+\.(com|org|net|edu|gov|mil|int|co\.uk|de|fr|it|es|ru|jp|cn)[^\s<>"{}|\\^`[\]]*'
        text = re.sub(url_pattern, '', text, flags=re.IGNORECASE)
        
        if text != original:
            result.rules_applied.append("remove_urls")
        
        return text
    
    def _remove_emails(self, text: str, result: CleaningResult) -> str:
        """Remove email addresses from text."""
        original = text
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        text = re.sub(email_pattern, '', text)
        
        if text != original:
            result.rules_applied.append("remove_emails")
        
        return text
    
    def _remove_phone_numbers(self, text: str, result: CleaningResult) -> str:
        """Remove phone numbers from text."""
        original = text
        
        # Phone number patterns
        patterns = [
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # US format
            r'\b\(\d{3}\)\s?\d{3}[-.]?\d{4}\b',  # (123) 456-7890
            r'\b\d{3}\s\d{3}\s\d{4}\b',  # 123 456 7890
            r'\+\d{1,3}[-.\s]?\d{1,14}\b',  # International
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, '', text)
        
        if text != original:
            result.rules_applied.append("remove_phone_numbers")
        
        return text
    
    def _normalize_unicode(self, text: str, form: NormalizationForm, result: CleaningResult) -> str:
        """Normalize Unicode characters."""
        original = text
        text = unicodedata.normalize(form.value, text)
        
        if text != original:
            result.rules_applied.append(f"normalize_unicode_{form.value}")
        
        return text
    
    def _convert_unicode_quotes(self, text: str, result: CleaningResult) -> str:
        """Convert Unicode quotes to ASCII equivalents."""
        original = text
        
        # Quote replacements
        quotes = {
            '"': '"',  # Left double quotation mark
            '"': '"',  # Right double quotation mark
            ''': "'",  # Left single quotation mark
            ''': "'",  # Right single quotation mark
            '„': '"',  # Double low-9 quotation mark
            '‚': "'",  # Single low-9 quotation mark
        }
        
        for unicode_quote, ascii_quote in quotes.items():
            text = text.replace(unicode_quote, ascii_quote)
        
        if text != original:
            result.rules_applied.append("convert_unicode_quotes")
        
        return text
    
    def _convert_unicode_dashes(self, text: str, result: CleaningResult) -> str:
        """Convert Unicode dashes to ASCII equivalents."""
        original = text
        
        # Dash replacements
        dashes = {
            '—': '--',  # Em dash
            '–': '-',   # En dash
            '−': '-',   # Minus sign
            '‒': '-',   # Figure dash
            '―': '--',  # Horizontal bar
        }
        
        for unicode_dash, ascii_dash in dashes.items():
            text = text.replace(unicode_dash, ascii_dash)
        
        if text != original:
            result.rules_applied.append("convert_unicode_dashes")
        
        return text
    
    def _replace_tabs_with_spaces(self, text: str, tab_width: int, result: CleaningResult) -> str:
        """Replace tabs with spaces."""
        original = text
        text = text.expandtabs(tab_width)
        
        if text != original:
            result.rules_applied.append("replace_tabs_with_spaces")
        
        return text
    
    def _normalize_whitespace(self, text: str, result: CleaningResult) -> str:
        """Normalize whitespace characters."""
        original = text
        
        # Replace multiple consecutive spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Normalize other whitespace characters
        text = re.sub(r'[\t\f\v]+', ' ', text)
        
        if text != original:
            result.rules_applied.append("normalize_whitespace")
        
        return text
    
    def _remove_extra_newlines(self, text: str, result: CleaningResult) -> str:
        """Remove excessive newlines."""
        original = text
        
        # Replace multiple consecutive newlines with at most two
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        if text != original:
            result.rules_applied.append("remove_extra_newlines")
        
        return text
    
    def _normalize_case(self, text: str, case_type: str, result: CleaningResult) -> str:
        """Normalize text case."""
        original = text
        
        if case_type == "lower":
            text = text.lower()
        elif case_type == "upper":
            text = text.upper()
        elif case_type == "title":
            text = text.title()
        
        if text != original:
            result.rules_applied.append(f"normalize_case_{case_type}")
        
        return text
    
    def _handle_line_lengths(self, text: str, config: CleaningConfig, result: CleaningResult) -> str:
        """Handle line length constraints."""
        lines = text.split('\n')
        modified_lines = []
        
        for line in lines:
            # Remove lines that are too short
            if config.min_line_length > 0 and len(line.strip()) < config.min_line_length:
                continue
            
            # Split lines that are too long
            if config.max_line_length > 0 and len(line) > config.max_line_length:
                # Simple word-based splitting
                words = line.split()
                current_line = ""
                
                for word in words:
                    if len(current_line + " " + word) <= config.max_line_length:
                        current_line += " " + word if current_line else word
                    else:
                        if current_line:
                            modified_lines.append(current_line)
                        current_line = word
                
                if current_line:
                    modified_lines.append(current_line)
            else:
                modified_lines.append(line)
        
        new_text = '\n'.join(modified_lines)
        
        if new_text != text:
            result.rules_applied.append("handle_line_lengths")
        
        return new_text
    
    def _join_hyphenated_words(self, text: str, result: CleaningResult) -> str:
        """Join hyphenated words across line breaks."""
        original = text
        
        # Join words that are hyphenated across line breaks
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        if text != original:
            result.rules_applied.append("join_hyphenated_words")
        
        return text
    
    def _apply_custom_rules(self, text: str, config: CleaningConfig, result: CleaningResult) -> str:
        """Apply custom cleaning rules."""
        for rule in self._compiled_rules:
            if not rule.enabled:
                continue
            
            try:
                original = text
                text = rule.pattern.sub(rule.replacement, text)
                
                if text != original:
                    result.rules_applied.append(rule.name)
                    
            except Exception as e:
                result.warnings.append(f"Custom rule '{rule.name}' failed: {e}")
                self.logger.warning(f"Custom rule '{rule.name}' failed: {e}")
        
        return text
    
    def _calculate_statistics(self, original: str, cleaned: str) -> Dict[str, Any]:
        """Calculate cleaning statistics."""
        return {
            "original_chars": len(original),
            "cleaned_chars": len(cleaned),
            "chars_removed": len(original) - len(cleaned),
            "reduction_percentage": ((len(original) - len(cleaned)) / len(original) * 100) if original else 0,
            "original_lines": original.count('\n') + 1 if original else 0,
            "cleaned_lines": cleaned.count('\n') + 1 if cleaned else 0,
            "original_words": len(original.split()) if original else 0,
            "cleaned_words": len(cleaned.split()) if cleaned else 0,
        }