"""
Enhanced Metadata Enrichment System with Access Control Integration.
"""

import asyncio
import hashlib
import json
import time
from typing import Dict, List, Any, Optional, Set, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

from src.core.logging import LoggerMixin
from src.core.exceptions import PipelineError
from src.text_processing.metadata_manager import MetadataManager, ChunkMetadata
from src.access_control.access_control_manager import AccessControlManager


class EnrichmentLevel(Enum):
    """Levels of metadata enrichment."""
    BASIC = "basic"  # Only essential metadata
    STANDARD = "standard"  # Standard enrichment
    COMPREHENSIVE = "comprehensive"  # Full enrichment with analysis
    CUSTOM = "custom"  # Custom enrichment rules


class SecurityClassification(Enum):
    """Security classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal" 
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


@dataclass
class ContentAnalysis:
    """Content analysis results."""
    language: Optional[str] = None
    readability_score: Optional[float] = None
    sentiment_score: Optional[float] = None
    topic_keywords: List[str] = field(default_factory=list)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    pii_detected: bool = False
    sensitive_patterns: List[str] = field(default_factory=list)
    classification_confidence: float = 0.0


@dataclass
class AccessControlMetadata:
    """Access control specific metadata."""
    security_classification: SecurityClassification = SecurityClassification.PUBLIC
    access_groups: List[str] = field(default_factory=list)
    data_subject_tags: List[str] = field(default_factory=list)
    retention_policy: Optional[str] = None
    encryption_required: bool = False
    audit_required: bool = False
    geographic_restrictions: List[str] = field(default_factory=list)
    compliance_tags: List[str] = field(default_factory=list)


@dataclass
class EnrichmentConfig:
    """Configuration for metadata enrichment."""
    # Analysis settings
    enable_language_detection: bool = True
    enable_sentiment_analysis: bool = False
    enable_entity_extraction: bool = True
    enable_pii_detection: bool = True
    enable_topic_analysis: bool = True
    enable_readability_analysis: bool = False
    
    # Security settings
    enable_security_classification: bool = True
    enable_compliance_tagging: bool = True
    auto_classify_sensitivity: bool = True
    
    # Performance settings
    max_analysis_time: float = 30.0
    batch_size: int = 50
    parallel_processing: bool = True
    
    # Custom enrichers
    custom_enrichers: Dict[str, Callable] = field(default_factory=dict)
    
    # Content patterns for detection
    pii_patterns: Dict[str, str] = field(default_factory=lambda: {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"
    })
    
    sensitivity_keywords: List[str] = field(default_factory=lambda: [
        "confidential", "secret", "private", "restricted", "classified",
        "password", "token", "key", "credential", "api_key"
    ])


class MetadataEnricher(LoggerMixin):
    """
    Enhanced metadata enrichment system with access control integration.
    
    Provides comprehensive metadata enrichment including:
    - Content analysis (language, sentiment, entities)
    - Security classification
    - PII detection
    - Access control tag generation
    - Compliance metadata
    """
    
    def __init__(
        self,
        config: EnrichmentConfig,
        metadata_manager: MetadataManager,
        access_control_manager: Optional[AccessControlManager] = None
    ):
        """
        Initialize metadata enricher.
        
        Args:
            config: Enrichment configuration
            metadata_manager: Base metadata manager
            access_control_manager: Access control manager (optional)
        """
        self.config = config
        self.metadata_manager = metadata_manager
        self.access_control_manager = access_control_manager
        
        # Initialize analysis components
        self._init_analysis_components()
        
        # Performance tracking
        self.enrichment_stats = {
            "total_enriched": 0,
            "successful": 0,
            "failed": 0,
            "pii_detected": 0,
            "classified_items": 0,
            "total_time": 0.0
        }
    
    def _init_analysis_components(self):
        """Initialize analysis components."""
        # Language detection
        if self.config.enable_language_detection:
            try:
                import langdetect
                self._langdetect_available = True
            except ImportError:
                self.logger.warning("langdetect not available for language detection")
                self._langdetect_available = False
        
        # Sentiment analysis
        if self.config.enable_sentiment_analysis:
            try:
                # Try to import a sentiment analysis library
                # This is a placeholder - you'd use actual libraries like transformers, vaderSentiment, etc.
                self._sentiment_available = True
            except ImportError:
                self.logger.warning("Sentiment analysis library not available")
                self._sentiment_available = False
        
        # Entity extraction
        if self.config.enable_entity_extraction:
            try:
                # Try to import NER libraries like spacy, transformers, etc.
                # This is a placeholder
                self._ner_available = True
            except ImportError:
                self.logger.warning("Named entity recognition library not available")
                self._ner_available = False
    
    async def enrich_metadata(
        self,
        content: str,
        base_metadata: Dict[str, Any],
        user_id: Optional[str] = None,
        source_path: Optional[str] = None,
        enrichment_level: EnrichmentLevel = EnrichmentLevel.STANDARD
    ) -> Dict[str, Any]:
        """
        Enrich metadata for content with comprehensive analysis.
        
        Args:
            content: Text content to analyze
            base_metadata: Base metadata to enrich
            user_id: User ID for access control
            source_path: Source file path
            enrichment_level: Level of enrichment to apply
            
        Returns:
            Enriched metadata dictionary
        """
        start_time = time.time()
        
        try:
            self.enrichment_stats["total_enriched"] += 1
            
            # Start with base metadata
            enriched_metadata = base_metadata.copy()
            
            # Add processing metadata
            enriched_metadata.update({
                "enrichment_timestamp": datetime.utcnow().isoformat(),
                "enrichment_level": enrichment_level.value,
                "enricher_version": "1.0"
            })
            
            # Perform content analysis
            if enrichment_level in [EnrichmentLevel.STANDARD, EnrichmentLevel.COMPREHENSIVE]:
                content_analysis = await self._analyze_content(content)
                enriched_metadata["content_analysis"] = self._serialize_content_analysis(content_analysis)
            
            # Perform security analysis
            if self.config.enable_security_classification:
                security_metadata = await self._analyze_security(content, source_path, user_id)
                enriched_metadata["access_control"] = self._serialize_access_control_metadata(security_metadata)
            
            # Apply access control tags
            if self.access_control_manager:
                access_tags = await self._generate_access_control_tags(content, enriched_metadata, user_id)
                enriched_metadata["access_tags"] = access_tags
            
            # Add source-based metadata
            if source_path:
                source_metadata = self._extract_source_metadata(source_path)
                enriched_metadata.update(source_metadata)
            
            # Apply custom enrichers
            if self.config.custom_enrichers:
                custom_metadata = await self._apply_custom_enrichers(content, enriched_metadata)
                enriched_metadata.update(custom_metadata)
            
            # Add compliance metadata
            if self.config.enable_compliance_tagging:
                compliance_metadata = self._generate_compliance_metadata(enriched_metadata)
                enriched_metadata["compliance"] = compliance_metadata
            
            # Calculate content fingerprint
            enriched_metadata["content_fingerprint"] = self._calculate_content_fingerprint(content)
            
            # Update statistics
            processing_time = time.time() - start_time
            self.enrichment_stats["successful"] += 1
            self.enrichment_stats["total_time"] += processing_time
            
            if enriched_metadata.get("content_analysis", {}).get("pii_detected"):
                self.enrichment_stats["pii_detected"] += 1
            
            if enriched_metadata.get("access_control", {}).get("security_classification"):
                self.enrichment_stats["classified_items"] += 1
            
            enriched_metadata["processing_time"] = processing_time
            
            self.logger.debug(f"Enriched metadata in {processing_time:.3f}s")
            return enriched_metadata
            
        except Exception as e:
            self.enrichment_stats["failed"] += 1
            self.logger.error(f"Failed to enrich metadata: {e}")
            
            # Return base metadata with error information
            base_metadata["enrichment_error"] = str(e)
            base_metadata["enrichment_timestamp"] = datetime.utcnow().isoformat()
            return base_metadata
    
    async def _analyze_content(self, content: str) -> ContentAnalysis:
        """Perform comprehensive content analysis."""
        analysis = ContentAnalysis()
        
        # Language detection
        if self.config.enable_language_detection and self._langdetect_available:
            try:
                import langdetect
                analysis.language = langdetect.detect(content)
            except Exception as e:
                self.logger.warning(f"Language detection failed: {e}")
        
        # PII detection
        if self.config.enable_pii_detection:
            analysis.pii_detected, analysis.sensitive_patterns = self._detect_pii(content)
        
        # Topic keyword extraction (simple implementation)
        if self.config.enable_topic_analysis:
            analysis.topic_keywords = self._extract_keywords(content)
        
        # Sentiment analysis (placeholder)
        if self.config.enable_sentiment_analysis and self._sentiment_available:
            analysis.sentiment_score = self._analyze_sentiment(content)
        
        # Entity extraction (placeholder)
        if self.config.enable_entity_extraction and self._ner_available:
            analysis.entities = self._extract_entities(content)
        
        # Readability analysis (simple implementation)
        if self.config.enable_readability_analysis:
            analysis.readability_score = self._calculate_readability(content)
        
        return analysis
    
    def _detect_pii(self, content: str) -> tuple[bool, List[str]]:
        """Detect personally identifiable information in content."""
        import re
        
        detected_patterns = []
        
        for pattern_name, pattern in self.config.pii_patterns.items():
            if re.search(pattern, content, re.IGNORECASE):
                detected_patterns.append(pattern_name)
        
        return len(detected_patterns) > 0, detected_patterns
    
    def _extract_keywords(self, content: str, max_keywords: int = 10) -> List[str]:
        """Extract topic keywords from content (simple implementation)."""
        # Simple keyword extraction based on word frequency
        words = content.lower().split()
        
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this', 'that', 'these', 'those'}
        
        # Count word frequencies
        word_freq = {}
        for word in words:
            # Clean the word
            word = ''.join(c for c in word if c.isalnum())
            if len(word) > 3 and word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:max_keywords]]
    
    def _analyze_sentiment(self, content: str) -> float:
        """Analyze sentiment of content (placeholder implementation)."""
        # This is a placeholder - in practice you'd use a real sentiment analysis library
        # Return a score between -1 (negative) and 1 (positive)
        
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like', 'happy', 'pleased']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'angry', 'sad', 'disappointed', 'frustrated', 'annoyed']
        
        words = content.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words == 0:
            return 0.0
        
        return (positive_count - negative_count) / total_sentiment_words
    
    def _extract_entities(self, content: str) -> List[Dict[str, Any]]:
        """Extract named entities from content (placeholder implementation)."""
        # This is a placeholder - in practice you'd use a real NER library like spacy
        entities = []
        
        # Simple pattern-based entity extraction
        import re
        
        # Date patterns
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b'
        dates = re.findall(date_pattern, content)
        for date in dates:
            entities.append({"text": date, "label": "DATE", "confidence": 0.8})
        
        # Email patterns (already in PII patterns)
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, content)
        for email in emails:
            entities.append({"text": email, "label": "EMAIL", "confidence": 0.9})
        
        return entities
    
    def _calculate_readability(self, content: str) -> float:
        """Calculate readability score (Flesch Reading Ease approximation)."""
        import re
        
        # Count sentences (approximate)
        sentences = len(re.split(r'[.!?]+', content))
        if sentences == 0:
            return 0.0
        
        # Count words
        words = len(content.split())
        if words == 0:
            return 0.0
        
        # Count syllables (very rough approximation)
        syllables = sum(max(1, len(re.findall(r'[aeiouAEIOU]', word))) for word in content.split())
        
        # Flesch Reading Ease formula (simplified)
        avg_sentence_length = words / sentences
        avg_syllables_per_word = syllables / words
        
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        return max(0.0, min(100.0, score))  # Clamp between 0 and 100
    
    async def _analyze_security(
        self,
        content: str,
        source_path: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> AccessControlMetadata:
        """Analyze security aspects of content."""
        security_metadata = AccessControlMetadata()
        
        # Auto-classify sensitivity based on content
        if self.config.auto_classify_sensitivity:
            security_metadata.security_classification = self._classify_sensitivity(content)
        
        # Detect if encryption is required
        security_metadata.encryption_required = self._requires_encryption(content)
        
        # Determine if audit is required
        security_metadata.audit_required = self._requires_audit(content, source_path)
        
        # Generate compliance tags
        security_metadata.compliance_tags = self._generate_compliance_tags(content)
        
        # Set retention policy based on content type
        security_metadata.retention_policy = self._determine_retention_policy(content, source_path)
        
        return security_metadata
    
    def _classify_sensitivity(self, content: str) -> SecurityClassification:
        """Classify content sensitivity level."""
        content_lower = content.lower()
        
        # Check for top secret indicators
        top_secret_keywords = ['top secret', 'classified', 'national security', 'state secret']
        if any(keyword in content_lower for keyword in top_secret_keywords):
            return SecurityClassification.TOP_SECRET
        
        # Check for restricted indicators
        restricted_keywords = ['restricted', 'highly confidential', 'executive only']
        if any(keyword in content_lower for keyword in restricted_keywords):
            return SecurityClassification.RESTRICTED
        
        # Check for confidential indicators
        confidential_keywords = ['confidential', 'private', 'internal only', 'proprietary']
        if any(keyword in content_lower for keyword in confidential_keywords):
            return SecurityClassification.CONFIDENTIAL
        
        # Check for internal indicators
        internal_keywords = ['internal', 'company only', 'employee only', 'staff only']
        if any(keyword in content_lower for keyword in internal_keywords):
            return SecurityClassification.INTERNAL
        
        # Default to public
        return SecurityClassification.PUBLIC
    
    def _requires_encryption(self, content: str) -> bool:
        """Determine if content requires encryption."""
        # Check for PII
        pii_detected, _ = self._detect_pii(content)
        if pii_detected:
            return True
        
        # Check for sensitive keywords
        sensitive_keywords = self.config.sensitivity_keywords
        content_lower = content.lower()
        
        return any(keyword in content_lower for keyword in sensitive_keywords)
    
    def _requires_audit(self, content: str, source_path: Optional[str] = None) -> bool:
        """Determine if content requires audit logging."""
        # Always audit if PII is detected
        pii_detected, _ = self._detect_pii(content)
        if pii_detected:
            return True
        
        # Audit based on source path patterns
        if source_path:
            audit_paths = ['financial', 'legal', 'hr', 'confidential', 'restricted']
            source_lower = source_path.lower()
            if any(pattern in source_lower for pattern in audit_paths):
                return True
        
        return False
    
    def _generate_compliance_tags(self, content: str) -> List[str]:
        """Generate compliance-related tags."""
        tags = []
        
        # GDPR compliance
        pii_detected, patterns = self._detect_pii(content)
        if pii_detected:
            tags.append("GDPR_APPLICABLE")
            if "email" in patterns:
                tags.append("PERSONAL_DATA")
        
        # Financial compliance
        financial_keywords = ['payment', 'credit card', 'bank account', 'financial', 'transaction']
        if any(keyword in content.lower() for keyword in financial_keywords):
            tags.append("PCI_DSS_APPLICABLE")
        
        # Healthcare compliance
        health_keywords = ['medical', 'health', 'patient', 'diagnosis', 'treatment']
        if any(keyword in content.lower() for keyword in health_keywords):
            tags.append("HIPAA_APPLICABLE")
        
        return tags
    
    def _determine_retention_policy(self, content: str, source_path: Optional[str] = None) -> str:
        """Determine retention policy based on content and source."""
        # PII data - shorter retention
        pii_detected, _ = self._detect_pii(content)
        if pii_detected:
            return "3_YEARS"
        
        # Financial data
        if any(keyword in content.lower() for keyword in ['financial', 'payment', 'transaction']):
            return "7_YEARS"
        
        # Legal documents
        if source_path and 'legal' in source_path.lower():
            return "10_YEARS"
        
        # Default retention
        return "5_YEARS"
    
    async def _generate_access_control_tags(
        self,
        content: str,
        metadata: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> List[str]:
        """Generate access control tags using access control manager."""
        tags = []
        
        if not self.access_control_manager:
            return tags
        
        try:
            # Generate tags based on content classification
            classification = metadata.get("access_control", {}).get("security_classification")
            if classification:
                tags.append(f"classification:{classification}")
            
            # Add user-based tags
            if user_id:
                tags.append(f"creator:{user_id}")
            
            # Add department/group tags based on user
            if user_id and self.access_control_manager:
                # This would integrate with the access control manager to get user groups
                # For now, we'll use a placeholder
                tags.append("department:unknown")
            
            # Add compliance-based tags
            compliance_tags = metadata.get("compliance", [])
            for compliance_tag in compliance_tags:
                tags.append(f"compliance:{compliance_tag}")
            
        except Exception as e:
            self.logger.warning(f"Failed to generate access control tags: {e}")
        
        return tags
    
    def _extract_source_metadata(self, source_path: str) -> Dict[str, Any]:
        """Extract metadata from source path."""
        path = Path(source_path)
        
        return {
            "source_filename": path.name,
            "source_extension": path.suffix,
            "source_directory": str(path.parent),
            "source_size": path.stat().st_size if path.exists() else None,
            "source_modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat() if path.exists() else None
        }
    
    async def _apply_custom_enrichers(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Apply custom enrichers."""
        custom_metadata = {}
        
        for enricher_name, enricher_func in self.config.custom_enrichers.items():
            try:
                result = await self._run_enricher(enricher_func, content, metadata)
                if result:
                    custom_metadata[f"custom_{enricher_name}"] = result
            except Exception as e:
                self.logger.warning(f"Custom enricher '{enricher_name}' failed: {e}")
        
        return custom_metadata
    
    async def _run_enricher(self, enricher_func: Callable, content: str, metadata: Dict[str, Any]) -> Any:
        """Run a custom enricher with timeout."""
        try:
            # Run with timeout
            return await asyncio.wait_for(
                asyncio.create_task(enricher_func(content, metadata)),
                timeout=self.config.max_analysis_time
            )
        except asyncio.TimeoutError:
            self.logger.warning(f"Custom enricher timed out after {self.config.max_analysis_time}s")
            return None
    
    def _generate_compliance_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate compliance metadata based on enriched metadata."""
        compliance = {}
        
        # GDPR compliance metadata
        if metadata.get("content_analysis", {}).get("pii_detected"):
            compliance["gdpr"] = {
                "applicable": True,
                "data_subject_rights": ["access", "rectification", "erasure", "portability"],
                "lawful_basis": "legitimate_interest",  # This would be determined by business logic
                "retention_period": metadata.get("access_control", {}).get("retention_policy", "5_YEARS")
            }
        
        # Data protection metadata
        if metadata.get("access_control", {}).get("encryption_required"):
            compliance["data_protection"] = {
                "encryption_required": True,
                "encryption_standard": "AES-256",
                "key_management": "required"
            }
        
        return compliance
    
    def _calculate_content_fingerprint(self, content: str) -> str:
        """Calculate a unique fingerprint for the content."""
        # Create a SHA-256 hash of the normalized content
        normalized_content = ' '.join(content.split())  # Normalize whitespace
        return hashlib.sha256(normalized_content.encode('utf-8')).hexdigest()
    
    def _serialize_content_analysis(self, analysis: ContentAnalysis) -> Dict[str, Any]:
        """Serialize content analysis to dictionary."""
        return {
            "language": analysis.language,
            "readability_score": analysis.readability_score,
            "sentiment_score": analysis.sentiment_score,
            "topic_keywords": analysis.topic_keywords,
            "entities": analysis.entities,
            "pii_detected": analysis.pii_detected,
            "sensitive_patterns": analysis.sensitive_patterns,
            "classification_confidence": analysis.classification_confidence
        }
    
    def _serialize_access_control_metadata(self, metadata: AccessControlMetadata) -> Dict[str, Any]:
        """Serialize access control metadata to dictionary."""
        return {
            "security_classification": metadata.security_classification.value,
            "access_groups": metadata.access_groups,
            "data_subject_tags": metadata.data_subject_tags,
            "retention_policy": metadata.retention_policy,
            "encryption_required": metadata.encryption_required,
            "audit_required": metadata.audit_required,
            "geographic_restrictions": metadata.geographic_restrictions,
            "compliance_tags": metadata.compliance_tags
        }
    
    def get_enrichment_statistics(self) -> Dict[str, Any]:
        """Get enrichment statistics."""
        total = self.enrichment_stats["total_enriched"]
        avg_time = self.enrichment_stats["total_time"] / max(total, 1)
        
        return {
            "total_enriched": total,
            "successful": self.enrichment_stats["successful"],
            "failed": self.enrichment_stats["failed"],
            "success_rate": (self.enrichment_stats["successful"] / max(total, 1)) * 100,
            "pii_detected": self.enrichment_stats["pii_detected"],
            "pii_detection_rate": (self.enrichment_stats["pii_detected"] / max(total, 1)) * 100,
            "classified_items": self.enrichment_stats["classified_items"],
            "classification_rate": (self.enrichment_stats["classified_items"] / max(total, 1)) * 100,
            "average_processing_time": avg_time,
            "total_processing_time": self.enrichment_stats["total_time"]
        }
    
    def reset_statistics(self) -> None:
        """Reset enrichment statistics."""
        self.enrichment_stats = {
            "total_enriched": 0,
            "successful": 0,
            "failed": 0,
            "pii_detected": 0,
            "classified_items": 0,
            "total_time": 0.0
        }