"""
Response Post-Processing and Formatting System.
"""

import re
import json
import html
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

from src.core.logging import LoggerMixin
from src.response_generation.base import ResponseResult, ResponseRequest, ResponseGeneratorConfig
from src.response_generation.exceptions import ResponseProcessingError


class OutputFormat(Enum):
    """Supported output formats."""
    PLAIN_TEXT = "plain_text"
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    STRUCTURED = "structured"
    CONVERSATIONAL = "conversational"


class ProcessingStrategy(Enum):
    """Post-processing strategies."""
    MINIMAL = "minimal"               # Basic cleaning only
    STANDARD = "standard"             # Standard formatting and validation
    ENHANCED = "enhanced"             # Enhanced formatting with structure
    CUSTOM = "custom"                 # Custom processing rules


@dataclass
class FormattingRules:
    """Rules for response formatting."""
    max_line_length: int = 80
    paragraph_spacing: int = 2
    list_indent: int = 2
    enable_auto_paragraphs: bool = True
    enable_auto_lists: bool = True
    enable_auto_headers: bool = True
    remove_redundancy: bool = True
    fix_grammar: bool = True
    normalize_punctuation: bool = True


@dataclass
class ValidationRules:
    """Rules for response validation."""
    min_length: int = 10
    max_length: int = 10000
    check_completeness: bool = True
    check_coherence: bool = True
    check_factual_consistency: bool = True
    remove_hallucinations: bool = False  # Conservative default
    validate_citations: bool = True


class ResponseProcessor(LoggerMixin):
    """
    Advanced Response Post-Processing and Formatting System.
    
    Features:
    - Multiple output format support (plain text, markdown, HTML, JSON)
    - Content validation and quality checks
    - Grammar and style corrections
    - Structured output formatting
    - Citation and reference handling
    - Response normalization and cleanup
    - Custom formatting rules
    """
    
    def __init__(self, config: ResponseGeneratorConfig):
        """
        Initialize Response Processor.
        
        Args:
            config: Response generator configuration
        """
        self.config = config
        
        # Initialize formatting and validation rules
        self.formatting_rules = FormattingRules()
        self.validation_rules = ValidationRules()
        
        # Processing strategy
        self.processing_strategy = ProcessingStrategy.STANDARD
        
        self.logger.info("ResponseProcessor initialized successfully")
    
    def process_response(
        self,
        response: ResponseResult,
        request: ResponseRequest,
        output_format: OutputFormat = OutputFormat.PLAIN_TEXT
    ) -> ResponseResult:
        """
        Process and format a response.
        
        Args:
            response: Response to process
            request: Original request for context
            output_format: Desired output format
            
        Returns:
            Processed response with formatted content
        """
        try:
            # Start with original response content
            processed_content = response.response
            
            # Apply pre-processing cleanup
            processed_content = self._pre_process_content(processed_content)
            
            # Apply validation
            if self.config.enable_post_processing:
                self._validate_content(processed_content, request)
            
            # Apply content corrections
            processed_content = self._apply_content_corrections(processed_content)
            
            # Apply format-specific processing
            processed_content = self._apply_format_processing(
                processed_content, output_format, request
            )
            
            # Apply post-processing enhancements
            processed_content = self._post_process_content(processed_content, request)
            
            # Update response with processed content
            response.response = processed_content
            
            # Add processing metadata
            response.processing_steps.append("Response post-processing")
            response.metadata.update({
                "output_format": output_format.value,
                "processing_strategy": self.processing_strategy.value
            })
            
            self.logger.debug(f"Processed response with {output_format.value} format")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Response processing failed: {e}")
            raise ResponseProcessingError(f"Response processing failed: {e}")
    
    def _pre_process_content(self, content: str) -> str:
        """Apply pre-processing cleanup to content."""
        
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove leading/trailing whitespace
        content = content.strip()
        
        # Fix common encoding issues
        content = html.unescape(content)
        
        # Remove control characters
        content = ''.join(char for char in content if ord(char) >= 32 or char in '\n\t')
        
        return content
    
    def _validate_content(self, content: str, request: ResponseRequest) -> None:
        """Validate content against validation rules."""
        
        # Check length constraints
        if len(content) < self.validation_rules.min_length:
            raise ResponseProcessingError(
                f"Response too short: {len(content)} < {self.validation_rules.min_length}"
            )
        
        if len(content) > self.validation_rules.max_length:
            raise ResponseProcessingError(
                f"Response too long: {len(content)} > {self.validation_rules.max_length}"
            )
        
        # Check for empty or meaningless content
        if not content.strip() or content.strip() in ['', 'None', 'null', 'undefined']:
            raise ResponseProcessingError("Response is empty or meaningless")
        
        # Check for potential hallucinations (basic patterns)
        if self.validation_rules.remove_hallucinations:
            self._check_for_hallucinations(content, request)
    
    def _check_for_hallucinations(self, content: str, request: ResponseRequest) -> None:
        """Check for potential hallucinations in content."""
        
        # Check for unrealistic claims
        unrealistic_patterns = [
            r'\b100% accurate\b', r'\bcompletely certain\b', r'\babsolutely guaranteed\b',
            r'\bnever fails\b', r'\balways works\b', r'\bimpossible to\b'
        ]
        
        for pattern in unrealistic_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                self.logger.warning(f"Potential overconfident claim detected: {pattern}")
    
    def _apply_content_corrections(self, content: str) -> str:
        """Apply content corrections and improvements."""
        
        if not self.formatting_rules.fix_grammar:
            return content
        
        # Fix common grammar issues
        corrections = [
            # Fix double spaces
            (r'\s{2,}', ' '),
            
            # Fix punctuation spacing
            (r'\s+([,.!?;:])', r'\1'),
            (r'([,.!?;:])\s*([a-zA-Z])', r'\1 \2'),
            
            # Fix capitalization after periods
            (r'(\.\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper()),
            
            # Fix common contractions
            (r'\bi am\b', "I'm"),
            (r'\bdo not\b', "don't"),
            (r'\bcannot\b', "can't"),
            (r'\bwill not\b', "won't"),
            
            # Fix sentence boundaries
            (r'([a-z])\.([A-Z])', r'\1. \2'),
        ]
        
        for pattern, replacement in corrections:
            if isinstance(replacement, str):
                content = re.sub(pattern, replacement, content)
            else:  # It's a function
                content = re.sub(pattern, replacement, content)
        
        return content
    
    def _apply_format_processing(
        self,
        content: str,
        output_format: OutputFormat,
        request: ResponseRequest
    ) -> str:
        """Apply format-specific processing."""
        
        if output_format == OutputFormat.PLAIN_TEXT:
            return self._format_plain_text(content)
        elif output_format == OutputFormat.MARKDOWN:
            return self._format_markdown(content)
        elif output_format == OutputFormat.HTML:
            return self._format_html(content)
        elif output_format == OutputFormat.JSON:
            return self._format_json(content, request)
        elif output_format == OutputFormat.STRUCTURED:
            return self._format_structured(content)
        elif output_format == OutputFormat.CONVERSATIONAL:
            return self._format_conversational(content, request)
        else:
            return content
    
    def _format_plain_text(self, content: str) -> str:
        """Format content as plain text."""
        
        # Apply paragraph formatting
        if self.formatting_rules.enable_auto_paragraphs:
            content = self._add_paragraph_breaks(content)
        
        # Apply list formatting
        if self.formatting_rules.enable_auto_lists:
            content = self._format_lists_plain(content)
        
        # Apply line length constraints
        if self.formatting_rules.max_line_length > 0:
            content = self._wrap_lines(content, self.formatting_rules.max_line_length)
        
        return content
    
    def _format_markdown(self, content: str) -> str:
        """Format content as Markdown."""
        
        # Convert structure to Markdown
        formatted_content = content
        
        # Auto-detect and format headers
        if self.formatting_rules.enable_auto_headers:
            formatted_content = self._add_markdown_headers(formatted_content)
        
        # Format lists
        if self.formatting_rules.enable_auto_lists:
            formatted_content = self._format_lists_markdown(formatted_content)
        
        # Format emphasis
        formatted_content = self._add_markdown_emphasis(formatted_content)
        
        # Add paragraph breaks
        if self.formatting_rules.enable_auto_paragraphs:
            formatted_content = self._add_paragraph_breaks(formatted_content)
        
        return formatted_content
    
    def _format_html(self, content: str) -> str:
        """Format content as HTML."""
        
        # Escape HTML entities
        formatted_content = html.escape(content)
        
        # Convert structure to HTML
        if self.formatting_rules.enable_auto_headers:
            formatted_content = self._add_html_headers(formatted_content)
        
        if self.formatting_rules.enable_auto_lists:
            formatted_content = self._format_lists_html(formatted_content)
        
        if self.formatting_rules.enable_auto_paragraphs:
            formatted_content = self._add_html_paragraphs(formatted_content)
        
        return formatted_content
    
    def _format_json(self, content: str, request: ResponseRequest) -> str:
        """Format content as structured JSON."""
        
        # Extract structured information
        structured_data = {
            "response": content,
            "query": request.query,
            "metadata": {
                "length": len(content),
                "timestamp": request.timestamp.isoformat(),
                "format": "json"
            }
        }
        
        # Try to extract additional structure
        try:
            # Look for lists, headers, etc.
            sections = self._extract_content_sections(content)
            if sections:
                structured_data["sections"] = sections
        except Exception:
            pass  # Fallback to simple structure
        
        return json.dumps(structured_data, indent=2, ensure_ascii=False)
    
    def _format_structured(self, content: str) -> str:
        """Format content with enhanced structure."""
        
        # Apply all structural enhancements
        formatted_content = content
        
        if self.formatting_rules.enable_auto_headers:
            formatted_content = self._add_section_headers(formatted_content)
        
        if self.formatting_rules.enable_auto_lists:
            formatted_content = self._enhance_lists(formatted_content)
        
        if self.formatting_rules.enable_auto_paragraphs:
            formatted_content = self._add_paragraph_breaks(formatted_content)
        
        # Add section numbering
        formatted_content = self._add_section_numbering(formatted_content)
        
        return formatted_content
    
    def _format_conversational(self, content: str, request: ResponseRequest) -> str:
        """Format content for conversational context."""
        
        # Make content more conversational
        formatted_content = content
        
        # Add conversational connectors
        formatted_content = self._add_conversational_elements(formatted_content)
        
        # Consider conversation history
        if request.conversation_history:
            formatted_content = self._adapt_to_conversation_style(
                formatted_content, request.conversation_history
            )
        
        return formatted_content
    
    def _post_process_content(self, content: str, request: ResponseRequest) -> str:
        """Apply final post-processing enhancements."""
        
        # Remove redundancy if enabled
        if self.formatting_rules.remove_redundancy:
            content = self._remove_redundancy(content)
        
        # Normalize punctuation
        if self.formatting_rules.normalize_punctuation:
            content = self._normalize_punctuation(content)
        
        # Final cleanup
        content = self._final_cleanup(content)
        
        return content
    
    def _add_paragraph_breaks(self, content: str) -> str:
        """Add appropriate paragraph breaks."""
        
        # Split by sentence patterns
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        paragraphs = []
        current_paragraph = []
        
        for sentence in sentences:
            current_paragraph.append(sentence)
            
            # Start new paragraph on topic changes (heuristic)
            if (len(current_paragraph) >= 3 and 
                any(word in sentence.lower() for word in 
                    ['however', 'meanwhile', 'furthermore', 'additionally', 'moreover'])):
                
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
        
        # Add remaining sentences
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        # Join with double line breaks
        spacing = '\n' * self.formatting_rules.paragraph_spacing
        return spacing.join(paragraphs)
    
    def _format_lists_plain(self, content: str) -> str:
        """Format lists in plain text."""
        
        # Detect list patterns
        list_patterns = [
            r'(\d+\.)\s+',  # Numbered lists
            r'([•\-\*])\s+',  # Bullet lists
            r'([a-z]\.)\s+',  # Lettered lists
        ]
        
        for pattern in list_patterns:
            content = re.sub(pattern, r'\n\1 ', content)
        
        return content
    
    def _format_lists_markdown(self, content: str) -> str:
        """Format lists in Markdown."""
        
        # Convert numbered lists
        content = re.sub(r'(\d+\.)\s+', r'\n\1 ', content)
        
        # Convert bullet lists
        content = re.sub(r'([•\-\*])\s+', r'\n- ', content)
        
        return content
    
    def _format_lists_html(self, content: str) -> str:
        """Format lists in HTML."""
        
        # This is a simplified implementation
        # In practice, you'd want more sophisticated list detection
        
        lines = content.split('\n')
        formatted_lines = []
        in_list = False
        list_type = None
        
        for line in lines:
            line = line.strip()
            
            # Check if this line is a list item
            if re.match(r'^\d+\.', line):
                if not in_list or list_type != 'ol':
                    if in_list:
                        formatted_lines.append(f'</{list_type}>')
                    formatted_lines.append('<ol>')
                    in_list = True
                    list_type = 'ol'
                
                item_content = re.sub(r'^\d+\.\s*', '', line)
                formatted_lines.append(f'  <li>{item_content}</li>')
                
            elif re.match(r'^[•\-\*]', line):
                if not in_list or list_type != 'ul':
                    if in_list:
                        formatted_lines.append(f'</{list_type}>')
                    formatted_lines.append('<ul>')
                    in_list = True
                    list_type = 'ul'
                
                item_content = re.sub(r'^[•\-\*]\s*', '', line)
                formatted_lines.append(f'  <li>{item_content}</li>')
                
            else:
                if in_list:
                    formatted_lines.append(f'</{list_type}>')
                    in_list = False
                    list_type = None
                
                if line:
                    formatted_lines.append(line)
        
        # Close any open list
        if in_list:
            formatted_lines.append(f'</{list_type}>')
        
        return '\n'.join(formatted_lines)
    
    def _add_markdown_headers(self, content: str) -> str:
        """Add Markdown headers based on content structure."""
        
        # Simple heuristic: capitalize sentences that look like headers
        lines = content.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                formatted_lines.append('')
                continue
            
            # Check if line looks like a header
            if (len(line) < 100 and 
                not line.endswith('.') and 
                not line.endswith('?') and 
                not line.endswith('!')):
                
                # Check for header indicators
                header_indicators = [
                    'introduction', 'conclusion', 'overview', 'summary',
                    'background', 'methodology', 'results', 'discussion'
                ]
                
                if any(indicator in line.lower() for indicator in header_indicators):
                    formatted_lines.append(f'## {line}')
                else:
                    formatted_lines.append(line)
            else:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def _add_markdown_emphasis(self, content: str) -> str:
        """Add Markdown emphasis to important terms."""
        
        # Emphasize important terms (simple heuristic)
        emphasis_patterns = [
            (r'\b(important|crucial|essential|key|significant)\b', r'**\1**'),
            (r'\b(note|warning|caution)\b', r'*\1*'),
        ]
        
        for pattern, replacement in emphasis_patterns:
            content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
        
        return content
    
    def _add_html_headers(self, content: str) -> str:
        """Add HTML headers based on content structure."""
        
        lines = content.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                formatted_lines.append('')
                continue
            
            # Simple header detection (similar to Markdown)
            if (len(line) < 100 and 
                not line.endswith('.') and 
                not line.endswith('?') and 
                not line.endswith('!')):
                
                header_indicators = [
                    'introduction', 'conclusion', 'overview', 'summary',
                    'background', 'methodology', 'results', 'discussion'
                ]
                
                if any(indicator in line.lower() for indicator in header_indicators):
                    formatted_lines.append(f'<h2>{line}</h2>')
                else:
                    formatted_lines.append(line)
            else:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def _add_html_paragraphs(self, content: str) -> str:
        """Add HTML paragraph tags."""
        
        paragraphs = content.split('\n\n')
        formatted_paragraphs = []
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if paragraph and not paragraph.startswith('<'):
                formatted_paragraphs.append(f'<p>{paragraph}</p>')
            else:
                formatted_paragraphs.append(paragraph)
        
        return '\n\n'.join(formatted_paragraphs)
    
    def _extract_content_sections(self, content: str) -> List[Dict[str, Any]]:
        """Extract structured sections from content."""
        
        sections = []
        
        # Split content into logical sections
        paragraphs = content.split('\n\n')
        
        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            section = {
                "id": i + 1,
                "content": paragraph,
                "type": "paragraph"
            }
            
            # Detect section type
            if len(paragraph) < 100 and not paragraph.endswith('.'):
                section["type"] = "header"
            elif any(pattern in paragraph for pattern in ['1.', '2.', '3.', '•', '-']):
                section["type"] = "list"
            
            sections.append(section)
        
        return sections
    
    def _wrap_lines(self, content: str, max_length: int) -> str:
        """Wrap lines to maximum length."""
        
        lines = content.split('\n')
        wrapped_lines = []
        
        for line in lines:
            if len(line) <= max_length:
                wrapped_lines.append(line)
            else:
                # Simple word wrapping
                words = line.split()
                current_line = []
                current_length = 0
                
                for word in words:
                    if current_length + len(word) + 1 <= max_length:
                        current_line.append(word)
                        current_length += len(word) + 1
                    else:
                        if current_line:
                            wrapped_lines.append(' '.join(current_line))
                        current_line = [word]
                        current_length = len(word)
                
                if current_line:
                    wrapped_lines.append(' '.join(current_line))
        
        return '\n'.join(wrapped_lines)
    
    def _remove_redundancy(self, content: str) -> str:
        """Remove redundant information from content."""
        
        # Remove repeated sentences
        sentences = re.split(r'[.!?]+', content)
        unique_sentences = []
        seen_sentences = set()
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence.lower() not in seen_sentences:
                unique_sentences.append(sentence)
                seen_sentences.add(sentence.lower())
        
        # Remove redundant phrases
        content = '. '.join(unique_sentences)
        
        redundant_patterns = [
            r'\bas mentioned above,?\s*',
            r'\bas stated previously,?\s*',
            r'\bin other words,?\s*',
            r'\bto reiterate,?\s*'
        ]
        
        for pattern in redundant_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        
        return content
    
    def _normalize_punctuation(self, content: str) -> str:
        """Normalize punctuation in content."""
        
        # Normalize quotes
        content = re.sub(r'["""]', '"', content)
        content = re.sub(r'[''']', "'", content)
        
        # Normalize dashes
        content = re.sub(r'[—–]', '-', content)
        
        # Fix spacing around punctuation
        content = re.sub(r'\s+([,.!?;:])', r'\1', content)
        content = re.sub(r'([,.!?;:])\s*([a-zA-Z])', r'\1 \2', content)
        
        return content
    
    def _final_cleanup(self, content: str) -> str:
        """Apply final cleanup to content."""
        
        # Remove excessive whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r'\s{2,}', ' ', content)
        
        # Trim whitespace
        content = content.strip()
        
        return content
    
    def _add_section_headers(self, content: str) -> str:
        """Add section headers to structured content."""
        # Implementation for adding section headers
        return content
    
    def _enhance_lists(self, content: str) -> str:
        """Enhance list formatting."""
        # Implementation for enhancing lists
        return content
    
    def _add_section_numbering(self, content: str) -> str:
        """Add section numbering."""
        # Implementation for section numbering
        return content
    
    def _add_conversational_elements(self, content: str) -> str:
        """Add conversational elements to content."""
        # Implementation for conversational formatting
        return content
    
    def _adapt_to_conversation_style(self, content: str, conversation_history: List) -> str:
        """Adapt content to conversation style."""
        # Implementation for conversation adaptation
        return content
    
    def set_formatting_rules(self, rules: FormattingRules) -> None:
        """Set custom formatting rules."""
        self.formatting_rules = rules
        self.logger.info("Updated formatting rules")
    
    def set_validation_rules(self, rules: ValidationRules) -> None:
        """Set custom validation rules."""
        self.validation_rules = rules
        self.logger.info("Updated validation rules")
    
    def get_processor_info(self) -> Dict[str, Any]:
        """Get processor configuration and statistics."""
        return {
            "processing_strategy": self.processing_strategy.value,
            "formatting_rules": {
                "max_line_length": self.formatting_rules.max_line_length,
                "paragraph_spacing": self.formatting_rules.paragraph_spacing,
                "enable_auto_paragraphs": self.formatting_rules.enable_auto_paragraphs,
                "enable_auto_lists": self.formatting_rules.enable_auto_lists,
                "enable_auto_headers": self.formatting_rules.enable_auto_headers,
                "remove_redundancy": self.formatting_rules.remove_redundancy,
                "fix_grammar": self.formatting_rules.fix_grammar,
                "normalize_punctuation": self.formatting_rules.normalize_punctuation
            },
            "validation_rules": {
                "min_length": self.validation_rules.min_length,
                "max_length": self.validation_rules.max_length,
                "check_completeness": self.validation_rules.check_completeness,
                "check_coherence": self.validation_rules.check_coherence,
                "remove_hallucinations": self.validation_rules.remove_hallucinations
            },
            "supported_formats": [fmt.value for fmt in OutputFormat]
        }