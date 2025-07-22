"""
Advanced Prompt Engineering with Dynamic Context Injection.
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from src.core.logging import LoggerMixin
from src.rag.context_retriever import RetrievalResult, DocumentContext
from src.response_generation.base import ResponseRequest, ResponseGeneratorConfig
from src.response_generation.exceptions import PromptEngineeringError, ContextTooLongError


class PromptStrategy(Enum):
    """Prompt engineering strategies."""
    SIMPLE = "simple"                    # Basic context concatenation
    STRUCTURED = "structured"            # Structured context organization
    HIERARCHICAL = "hierarchical"        # Hierarchical context arrangement
    CONTEXTUAL = "contextual"           # Context-aware prompt adaptation
    CONVERSATIONAL = "conversational"   # Conversation-optimized prompts
    DOMAIN_SPECIFIC = "domain_specific" # Domain-adapted prompts


class ContextInjectionMode(Enum):
    """Context injection modes."""
    PREPEND = "prepend"                 # Add context before query
    APPEND = "append"                   # Add context after query
    INTERLEAVED = "interleaved"         # Interleave context with query
    ADAPTIVE = "adaptive"               # Adaptively place context
    STRUCTURED_SECTIONS = "structured_sections"  # Use structured sections


@dataclass
class PromptTemplate:
    """Template for prompt generation."""
    name: str
    template: str
    required_variables: List[str] = field(default_factory=list)
    optional_variables: List[str] = field(default_factory=list)
    context_injection_points: List[str] = field(default_factory=list)
    max_context_length: Optional[int] = None
    domain: Optional[str] = None
    
    def validate_variables(self, variables: Dict[str, Any]) -> bool:
        """Validate that all required variables are provided."""
        return all(var in variables for var in self.required_variables)


@dataclass
class ContextRelevanceFilter:
    """Filter for context relevance."""
    min_similarity_threshold: float = 0.3
    max_contexts: int = 5
    enable_diversity_filtering: bool = True
    diversity_threshold: float = 0.8
    enable_length_filtering: bool = True
    max_context_length: int = 2000
    min_context_length: int = 50


@dataclass
class PromptOptimizationConfig:
    """Configuration for prompt optimization."""
    enable_context_compression: bool = True
    compression_ratio: float = 0.7
    enable_redundancy_removal: bool = True
    enable_context_ranking: bool = True
    enable_dynamic_templates: bool = True
    max_total_length: int = 8000
    preserve_query_context: bool = True


class PromptEngineer(LoggerMixin):
    """
    Advanced Prompt Engineering with Dynamic Context Injection.
    
    Features:
    - Multiple prompt strategies and templates
    - Dynamic context injection and optimization
    - Context relevance filtering and ranking
    - Prompt template management and variables
    - Context compression and redundancy removal
    - Domain-specific prompt adaptation
    """
    
    def __init__(self, config: ResponseGeneratorConfig):
        """
        Initialize Prompt Engineer.
        
        Args:
            config: Response generator configuration
        """
        self.config = config
        
        # Initialize prompt templates
        self.templates = self._initialize_default_templates()
        
        # Configuration for prompt optimization
        self.optimization_config = PromptOptimizationConfig(
            max_total_length=config.max_context_length,
            compression_ratio=config.context_compression_ratio
        )
        
        # Context filtering configuration
        self.context_filter = ContextRelevanceFilter(
            max_contexts=5,
            max_context_length=config.max_context_length // 4  # Allow room for multiple contexts
        )
        
        self.logger.info("PromptEngineer initialized successfully")
    
    def engineer_prompt(
        self,
        request: ResponseRequest,
        strategy: PromptStrategy = PromptStrategy.STRUCTURED,
        injection_mode: ContextInjectionMode = ContextInjectionMode.STRUCTURED_SECTIONS
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Engineer optimized prompt with context injection.
        
        Args:
            request: Response generation request
            strategy: Prompt engineering strategy
            injection_mode: Context injection mode
            
        Returns:
            Tuple of (engineered_prompt, metadata)
        """
        try:
            # Extract and filter contexts
            contexts = self._extract_and_filter_contexts(request)
            
            # Select appropriate template
            template = self._select_template(request, strategy)
            
            # Prepare template variables
            variables = self._prepare_template_variables(request, contexts)
            
            # Generate base prompt
            base_prompt = self._generate_base_prompt(template, variables)
            
            # Inject context using specified mode
            final_prompt = self._inject_context(
                base_prompt, contexts, injection_mode, request
            )
            
            # Optimize prompt length
            optimized_prompt = self._optimize_prompt_length(final_prompt)
            
            # Create metadata
            metadata = {
                "strategy": strategy.value,
                "injection_mode": injection_mode.value,
                "template_used": template.name,
                "contexts_used": len(contexts),
                "original_length": len(final_prompt),
                "optimized_length": len(optimized_prompt),
                "compression_applied": len(final_prompt) != len(optimized_prompt)
            }
            
            self.logger.debug(f"Engineered prompt with {len(contexts)} contexts using {strategy.value} strategy")
            
            return optimized_prompt, metadata
            
        except Exception as e:
            self.logger.error(f"Prompt engineering failed: {e}")
            raise PromptEngineeringError(f"Prompt engineering failed: {e}")
    
    def _extract_and_filter_contexts(self, request: ResponseRequest) -> List[DocumentContext]:
        """Extract and filter relevant contexts from request."""
        contexts = []
        
        # Extract from retrieval result
        if request.retrieval_result and request.retrieval_result.contexts:
            contexts.extend(request.retrieval_result.contexts)
        
        # Filter contexts by relevance
        filtered_contexts = self._filter_contexts_by_relevance(contexts, request.query)
        
        # Apply diversity filtering
        if self.context_filter.enable_diversity_filtering:
            filtered_contexts = self._apply_diversity_filtering(filtered_contexts)
        
        # Apply length filtering
        if self.context_filter.enable_length_filtering:
            filtered_contexts = self._apply_length_filtering(filtered_contexts)
        
        # Limit number of contexts
        final_contexts = filtered_contexts[:self.context_filter.max_contexts]
        
        return final_contexts
    
    def _filter_contexts_by_relevance(
        self, 
        contexts: List[DocumentContext], 
        query: str
    ) -> List[DocumentContext]:
        """Filter contexts by relevance to query."""
        if not contexts:
            return []
        
        # Sort by similarity score
        sorted_contexts = sorted(
            contexts, 
            key=lambda ctx: ctx.similarity_score, 
            reverse=True
        )
        
        # Filter by minimum threshold
        relevant_contexts = [
            ctx for ctx in sorted_contexts
            if ctx.similarity_score >= self.context_filter.min_similarity_threshold
        ]
        
        return relevant_contexts
    
    def _apply_diversity_filtering(self, contexts: List[DocumentContext]) -> List[DocumentContext]:
        """Apply diversity filtering to avoid redundant contexts."""
        if len(contexts) <= 1:
            return contexts
        
        selected_contexts = [contexts[0]]  # Start with highest ranked
        
        for candidate in contexts[1:]:
            is_diverse = True
            
            for selected in selected_contexts:
                similarity = self._calculate_content_similarity(
                    candidate.content, selected.content
                )
                
                if similarity > self.context_filter.diversity_threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                selected_contexts.append(candidate)
        
        return selected_contexts
    
    def _apply_length_filtering(self, contexts: List[DocumentContext]) -> List[DocumentContext]:
        """Apply length filtering to contexts."""
        filtered_contexts = []
        
        for ctx in contexts:
            content_length = len(ctx.content)
            
            if (self.context_filter.min_context_length <= content_length <= 
                self.context_filter.max_context_length):
                filtered_contexts.append(ctx)
            elif content_length > self.context_filter.max_context_length:
                # Truncate long context
                truncated_ctx = DocumentContext(
                    id=ctx.id,
                    content=ctx.content[:self.context_filter.max_context_length] + "...",
                    metadata=ctx.metadata,
                    similarity_score=ctx.similarity_score,
                    source_info=ctx.source_info
                )
                filtered_contexts.append(truncated_ctx)
        
        return filtered_contexts
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two content strings."""
        # Simple word-based similarity
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _select_template(self, request: ResponseRequest, strategy: PromptStrategy) -> PromptTemplate:
        """Select appropriate template based on request and strategy."""
        
        # Use custom system prompt if provided
        if request.system_prompt:
            return PromptTemplate(
                name="custom",
                template=request.system_prompt + "\n\n{context}\n\nQuery: {query}",
                required_variables=["query"],
                optional_variables=["context"],
                context_injection_points=["context"]
            )
        
        # Select based on strategy
        template_mapping = {
            PromptStrategy.SIMPLE: "simple_qa",
            PromptStrategy.STRUCTURED: "structured_response",
            PromptStrategy.HIERARCHICAL: "hierarchical_context",
            PromptStrategy.CONTEXTUAL: "contextual_adaptive",
            PromptStrategy.CONVERSATIONAL: "conversational",
            PromptStrategy.DOMAIN_SPECIFIC: "domain_specific"
        }
        
        template_name = template_mapping.get(strategy, "structured_response")
        
        if template_name not in self.templates:
            self.logger.warning(f"Template {template_name} not found, using default")
            template_name = "structured_response"
        
        return self.templates[template_name]
    
    def _prepare_template_variables(
        self, 
        request: ResponseRequest, 
        contexts: List[DocumentContext]
    ) -> Dict[str, Any]:
        """Prepare variables for template substitution."""
        variables = {
            "query": request.query,
            "context": self._format_contexts(contexts),
            "conversation_history": self._format_conversation_history(request),
            "user_context": self._format_user_context(request.user_context),
            "custom_instructions": request.custom_instructions or "",
            "response_format": request.response_format or ""
        }
        
        return variables
    
    def _format_contexts(self, contexts: List[DocumentContext]) -> str:
        """Format contexts for inclusion in prompt."""
        if not contexts:
            return ""
        
        formatted_contexts = []
        
        for i, ctx in enumerate(contexts, 1):
            # Include source info if available
            source_info = ""
            if ctx.source_info and "title" in ctx.source_info:
                source_info = f" (Source: {ctx.source_info['title']})"
            
            formatted_context = f"Context {i}{source_info}:\n{ctx.content}"
            formatted_contexts.append(formatted_context)
        
        return "\n\n".join(formatted_contexts)
    
    def _format_conversation_history(self, request: ResponseRequest) -> str:
        """Format conversation history for prompt."""
        if not request.conversation_history:
            return ""
        
        history_parts = []
        for msg in request.conversation_history[-5:]:  # Last 5 messages
            role = msg.role.value.title()
            content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            history_parts.append(f"{role}: {content}")
        
        return "\n".join(history_parts)
    
    def _format_user_context(self, user_context: Optional[Dict[str, Any]]) -> str:
        """Format user context for prompt."""
        if not user_context:
            return ""
        
        context_parts = []
        for key, value in user_context.items():
            context_parts.append(f"{key}: {value}")
        
        return ", ".join(context_parts)
    
    def _generate_base_prompt(self, template: PromptTemplate, variables: Dict[str, Any]) -> str:
        """Generate base prompt from template and variables."""
        # Validate required variables
        if not template.validate_variables(variables):
            missing_vars = [var for var in template.required_variables if var not in variables]
            raise PromptEngineeringError(f"Missing required variables: {missing_vars}")
        
        # Substitute variables in template
        try:
            prompt = template.template.format(**variables)
        except KeyError as e:
            raise PromptEngineeringError(f"Template variable substitution failed: {e}")
        
        return prompt
    
    def _inject_context(
        self,
        base_prompt: str,
        contexts: List[DocumentContext],
        injection_mode: ContextInjectionMode,
        request: ResponseRequest
    ) -> str:
        """Inject context into prompt using specified mode."""
        
        if not contexts:
            return base_prompt
        
        context_text = self._format_contexts(contexts)
        
        if injection_mode == ContextInjectionMode.PREPEND:
            return f"{context_text}\n\n{base_prompt}"
        
        elif injection_mode == ContextInjectionMode.APPEND:
            return f"{base_prompt}\n\n{context_text}"
        
        elif injection_mode == ContextInjectionMode.INTERLEAVED:
            return self._interleave_context(base_prompt, contexts, request)
        
        elif injection_mode == ContextInjectionMode.STRUCTURED_SECTIONS:
            return self._inject_structured_sections(base_prompt, contexts, request)
        
        elif injection_mode == ContextInjectionMode.ADAPTIVE:
            return self._inject_adaptive(base_prompt, contexts, request)
        
        else:
            # Default to prepend
            return f"{context_text}\n\n{base_prompt}"
    
    def _interleave_context(
        self,
        base_prompt: str,
        contexts: List[DocumentContext],
        request: ResponseRequest
    ) -> str:
        """Interleave context with prompt components."""
        # Split prompt into sections
        sections = base_prompt.split("\n\n")
        
        # Interleave contexts between sections
        interleaved_parts = []
        context_index = 0
        
        for i, section in enumerate(sections):
            interleaved_parts.append(section)
            
            # Add context after each section (except the last)
            if i < len(sections) - 1 and context_index < len(contexts):
                ctx = contexts[context_index]
                interleaved_parts.append(f"Relevant Information: {ctx.content}")
                context_index += 1
        
        return "\n\n".join(interleaved_parts)
    
    def _inject_structured_sections(
        self,
        base_prompt: str,
        contexts: List[DocumentContext],
        request: ResponseRequest
    ) -> str:
        """Inject context using structured sections."""
        sections = []
        
        # Add system instructions
        sections.append(base_prompt)
        
        # Add context section
        if contexts:
            sections.append("## Relevant Information")
            for i, ctx in enumerate(contexts, 1):
                source_info = ""
                if ctx.source_info and "title" in ctx.source_info:
                    source_info = f" - {ctx.source_info['title']}"
                
                sections.append(f"### Source {i}{source_info}")
                sections.append(ctx.content)
        
        # Add conversation history if available
        if request.conversation_history:
            sections.append("## Conversation History")
            sections.append(self._format_conversation_history(request))
        
        # Add user query section
        sections.append("## User Query")
        sections.append(request.query)
        
        return "\n\n".join(sections)
    
    def _inject_adaptive(
        self,
        base_prompt: str,
        contexts: List[DocumentContext],
        request: ResponseRequest
    ) -> str:
        """Adaptively inject context based on query and context characteristics."""
        
        # Analyze query to determine best injection strategy
        query_length = len(request.query)
        context_total_length = sum(len(ctx.content) for ctx in contexts)
        
        # For short queries with long context, use structured sections
        if query_length < 100 and context_total_length > 1000:
            return self._inject_structured_sections(base_prompt, contexts, request)
        
        # For long queries with short context, prepend context
        elif query_length > 500 and context_total_length < 500:
            context_text = self._format_contexts(contexts)
            return f"{context_text}\n\n{base_prompt}"
        
        # Default to interleaved for balanced cases
        else:
            return self._interleave_context(base_prompt, contexts, request)
    
    def _optimize_prompt_length(self, prompt: str) -> str:
        """Optimize prompt length while preserving important information."""
        
        if len(prompt) <= self.optimization_config.max_total_length:
            return prompt
        
        self.logger.warning(f"Prompt length {len(prompt)} exceeds maximum {self.optimization_config.max_total_length}, applying compression")
        
        # Apply compression strategies
        optimized_prompt = prompt
        
        if self.optimization_config.enable_redundancy_removal:
            optimized_prompt = self._remove_redundancy(optimized_prompt)
        
        if self.optimization_config.enable_context_compression:
            optimized_prompt = self._compress_content(optimized_prompt)
        
        # Final truncation if still too long
        if len(optimized_prompt) > self.optimization_config.max_total_length:
            # Preserve query at the end
            truncation_point = int(self.optimization_config.max_total_length * 0.9)
            optimized_prompt = optimized_prompt[:truncation_point] + "\n\n[Content truncated for length]"
        
        return optimized_prompt
    
    def _remove_redundancy(self, prompt: str) -> str:
        """Remove redundant information from prompt."""
        # Split into sentences
        sentences = re.split(r'[.!?]+', prompt)
        
        # Remove duplicate sentences
        unique_sentences = []
        seen_sentences = set()
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence.lower() not in seen_sentences:
                unique_sentences.append(sentence)
                seen_sentences.add(sentence.lower())
        
        return '. '.join(unique_sentences)
    
    def _compress_content(self, prompt: str) -> str:
        """Compress content while preserving meaning."""
        # Simple compression: remove excessive whitespace and redundant phrases
        
        # Remove excessive whitespace
        compressed = re.sub(r'\s+', ' ', prompt)
        
        # Remove redundant phrases
        redundant_patterns = [
            r'\bAs mentioned above,?\s*',
            r'\bAs stated previously,?\s*',
            r'\bIn other words,?\s*',
            r'\bTo summarize,?\s*',
            r'\bIn summary,?\s*'
        ]
        
        for pattern in redundant_patterns:
            compressed = re.sub(pattern, '', compressed, flags=re.IGNORECASE)
        
        return compressed.strip()
    
    def _initialize_default_templates(self) -> Dict[str, PromptTemplate]:
        """Initialize default prompt templates."""
        templates = {}
        
        # Simple Q&A template
        templates["simple_qa"] = PromptTemplate(
            name="simple_qa",
            template="Answer the following question based on the provided context.\n\nContext: {context}\n\nQuestion: {query}",
            required_variables=["query"],
            optional_variables=["context"],
            context_injection_points=["context"]
        )
        
        # Structured response template
        templates["structured_response"] = PromptTemplate(
            name="structured_response",
            template="""You are a helpful AI assistant. Provide accurate and helpful responses based on the given context and conversation history.

{context}

{conversation_history}

Please provide a comprehensive answer to the following query: {query}

{custom_instructions}""",
            required_variables=["query"],
            optional_variables=["context", "conversation_history", "custom_instructions"],
            context_injection_points=["context"]
        )
        
        # Hierarchical context template
        templates["hierarchical_context"] = PromptTemplate(
            name="hierarchical_context",
            template="""You are an expert assistant. Use the following information hierarchy to answer the user's question:

## Primary Context
{context}

## User Background
{user_context}

## Previous Conversation
{conversation_history}

## Current Query
{query}

Provide a detailed and accurate response considering all available information.""",
            required_variables=["query"],
            optional_variables=["context", "user_context", "conversation_history"],
            context_injection_points=["context"]
        )
        
        # Contextual adaptive template
        templates["contextual_adaptive"] = PromptTemplate(
            name="contextual_adaptive",
            template="""Based on the following context and user query, provide the most relevant and helpful response:

Context Information:
{context}

User Query: {query}

Response Format: {response_format}

Additional Instructions: {custom_instructions}""",
            required_variables=["query"],
            optional_variables=["context", "response_format", "custom_instructions"],
            context_injection_points=["context"]
        )
        
        # Conversational template
        templates["conversational"] = PromptTemplate(
            name="conversational",
            template="""Continue this conversation naturally and helpfully:

Previous Messages:
{conversation_history}

Current Context:
{context}

Current Message: {query}

Respond in a conversational and engaging manner.""",
            required_variables=["query"],
            optional_variables=["conversation_history", "context"],
            context_injection_points=["context"]
        )
        
        # Domain-specific template
        templates["domain_specific"] = PromptTemplate(
            name="domain_specific",
            template="""You are a specialized expert in the relevant domain. Use your expertise and the provided context to answer:

Expert Knowledge Base:
{context}

User Context: {user_context}

Question: {query}

Provide an expert-level response with appropriate technical depth.""",
            required_variables=["query"],
            optional_variables=["context", "user_context"],
            context_injection_points=["context"]
        )
        
        return templates
    
    def add_custom_template(self, template: PromptTemplate) -> None:
        """Add a custom prompt template."""
        self.templates[template.name] = template
        self.logger.info(f"Added custom template: {template.name}")
    
    def get_available_templates(self) -> List[str]:
        """Get list of available template names."""
        return list(self.templates.keys())
    
    def get_template_info(self, template_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific template."""
        if template_name not in self.templates:
            return None
        
        template = self.templates[template_name]
        return {
            "name": template.name,
            "required_variables": template.required_variables,
            "optional_variables": template.optional_variables,
            "context_injection_points": template.context_injection_points,
            "max_context_length": template.max_context_length,
            "domain": template.domain
        }