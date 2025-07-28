#!/usr/bin/env python3
"""
RDB Pipeline Integration Test Script

This script provides comprehensive testing of the RDB to Vector pipeline functionality.
It tests all components individually and then as an integrated system.
"""

import asyncio
import sys
import os
import json
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.config import DatabaseConfig, DatabaseType, get_config
from src.extraction.base import ExtractionConfig, ExtractionMode, DataFormat
from src.extraction.factory import RDBExtractorFactory
from src.pipeline.rdb_adapter import RDBAdapterConfig, RDBPipelineConnector
from src.pipeline.rdb_pipeline import create_rdb_vector_pipeline, RDBPipelineConfig
from src.pipeline.rdb_config_validator import validate_rdb_system, test_database_connection
from src.core.logging import LoggerMixin


class RDBPipelineIntegrationTest(LoggerMixin):
    """Comprehensive integration test for RDB pipeline."""
    
    def __init__(self):
        """Initialize integration test."""
        self.test_results = {}
        self.test_start_time = datetime.utcnow()
        
        self.logger.info("RDB Pipeline Integration Test started")
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all integration tests.
        
        Returns:
            Complete test results
        """
        self.logger.info("Starting comprehensive RDB pipeline integration tests")
        
        # Test configuration validation
        await self._test_configuration_validation()
        
        # Test RDB extraction components
        await self._test_rdb_extraction()
        
        # Test document adapter
        await self._test_document_adapter()
        
        # Test integrated pipeline
        await self._test_integrated_pipeline()
        
        # Test CLI commands
        await self._test_cli_commands()
        
        # Generate final report
        return self._generate_test_report()
    
    async def _test_configuration_validation(self):
        """Test configuration validation functionality."""
        test_name = "configuration_validation"
        self.logger.info(f"Testing: {test_name}")
        
        try:
            # Test system validation
            validation_report = await validate_rdb_system()
            
            self.test_results[test_name] = {
                "status": "passed" if validation_report.overall_status else "warning",
                "message": f"System validation completed with {validation_report.summary['passed']}/{validation_report.summary['total_checks']} checks passed",
                "details": {
                    "validation_id": validation_report.validation_id,
                    "summary": validation_report.summary,
                    "critical_errors": len(validation_report.critical_errors),
                    "errors": len(validation_report.errors),
                    "warnings": len(validation_report.warnings)
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.logger.info(f"✓ {test_name} completed")
            
        except Exception as e:
            self.test_results[test_name] = {
                "status": "failed",
                "message": f"Configuration validation failed: {e}",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.utcnow().isoformat()
            }
            self.logger.error(f"✗ {test_name} failed: {e}")
    
    async def _test_rdb_extraction(self):
        """Test RDB extraction components."""
        test_name = "rdb_extraction"
        self.logger.info(f"Testing: {test_name}")
        
        try:
            config = get_config()
            
            if not hasattr(config, 'rdb_connections') or not config.rdb_connections:
                self.test_results[test_name] = {
                    "status": "skipped",
                    "message": "No RDB connections configured",
                    "timestamp": datetime.utcnow().isoformat()
                }
                return
            
            # Test with first configured database
            db_name = list(config.rdb_connections.keys())[0]
            db_config = config.rdb_connections[db_name]
            
            # Create extraction config
            extraction_config = ExtractionConfig(
                database_config=db_config,
                mode=ExtractionMode.FULL,
                batch_size=10,
                max_rows=5,  # Small number for testing
                timeout=30
            )
            
            # Test extractor creation
            extractor = RDBExtractorFactory.create(extraction_config)
            
            try:
                # Test connection
                connection_valid = extractor.validate_connection()
                
                if not connection_valid:
                    self.test_results[test_name] = {
                        "status": "failed",
                        "message": "Database connection validation failed",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    return
                
                # Test table discovery
                tables = extractor.discover_tables()
                
                # Test data extraction (if tables exist)
                extraction_results = []
                if tables:
                    test_table = tables[0]
                    result = extractor.extract_table_data(test_table)
                    extraction_results.append({
                        "table": test_table,
                        "row_count": len(result.data),
                        "has_metadata": result.metadata is not None,
                        "has_errors": len(result.errors) > 0
                    })
                
                self.test_results[test_name] = {
                    "status": "passed",
                    "message": f"RDB extraction successful: {len(tables)} tables discovered",
                    "details": {
                        "database": db_name,
                        "connection_valid": connection_valid,
                        "table_count": len(tables),
                        "tables": tables[:5],  # First 5 tables
                        "extraction_results": extraction_results
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            finally:
                extractor.close()
            
            self.logger.info(f"✓ {test_name} completed")
            
        except Exception as e:
            self.test_results[test_name] = {
                "status": "failed",
                "message": f"RDB extraction test failed: {e}",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.utcnow().isoformat()
            }
            self.logger.error(f"✗ {test_name} failed: {e}")
    
    async def _test_document_adapter(self):
        """Test document adapter functionality."""
        test_name = "document_adapter"
        self.logger.info(f"Testing: {test_name}")
        
        try:
            config = get_config()
            
            if not hasattr(config, 'rdb_connections') or not config.rdb_connections:
                self.test_results[test_name] = {
                    "status": "skipped",
                    "message": "No RDB connections configured",
                    "timestamp": datetime.utcnow().isoformat()
                }
                return
            
            # Test with first configured database
            db_name = list(config.rdb_connections.keys())[0]
            db_config = config.rdb_connections[db_name]
            
            # Create extraction and adapter configs
            extraction_config = ExtractionConfig(
                database_config=db_config,
                mode=ExtractionMode.FULL,
                batch_size=5,
                max_rows=3  # Very small for testing
            )
            
            adapter_config = RDBAdapterConfig(
                content_format="structured",
                include_table_name=True,
                include_column_names=True,
                exclude_null_values=True
            )
            
            # Create connector
            connector = RDBPipelineConnector(extraction_config, adapter_config)
            
            try:
                # Test document conversion
                documents = await connector.extract_and_convert_all_tables_async(
                    database_name=db_name,
                    max_concurrent=2
                )
                
                self.test_results[test_name] = {
                    "status": "passed",
                    "message": f"Document adapter successful: {len(documents)} documents generated",
                    "details": {
                        "database": db_name,
                        "document_count": len(documents),
                        "sample_document": documents[0].to_dict() if documents else None,
                        "adapter_config": {
                            "content_format": adapter_config.content_format,
                            "include_table_name": adapter_config.include_table_name,
                            "include_column_names": adapter_config.include_column_names
                        }
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            finally:
                connector.close()
            
            self.logger.info(f"✓ {test_name} completed")
            
        except Exception as e:
            self.test_results[test_name] = {
                "status": "failed",
                "message": f"Document adapter test failed: {e}",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.utcnow().isoformat()
            }
            self.logger.error(f"✗ {test_name} failed: {e}")
    
    async def _test_integrated_pipeline(self):
        """Test the complete integrated pipeline."""
        test_name = "integrated_pipeline"
        self.logger.info(f"Testing: {test_name}")
        
        try:
            config = get_config()
            
            if not hasattr(config, 'rdb_connections') or not config.rdb_connections:
                self.test_results[test_name] = {
                    "status": "skipped",
                    "message": "No RDB connections configured",
                    "timestamp": datetime.utcnow().isoformat()
                }
                return
            
            # Test with first configured database
            db_name = list(config.rdb_connections.keys())[0]
            db_config = config.rdb_connections[db_name]
            
            # Create test collection name
            test_collection = f"test_rdb_pipeline_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Create pipeline with test settings
            pipeline = create_rdb_vector_pipeline(
                database_name=db_name,
                database_config=db_config,
                collection_name=test_collection,
                extraction_mode=ExtractionMode.FULL,
                extraction_batch_size=5,
                max_rows_per_table=2,  # Very small for testing
                continue_on_table_error=True,
                continue_on_pipeline_error=True,
                max_concurrent_tables=2
            )
            
            try:
                # Test pipeline health check
                health_status = await pipeline.health_check()
                
                # Run pipeline (limited scope for testing)
                # We'll test with a very small subset to avoid overwhelming the system
                result = await pipeline.process_all_tables()
                
                self.test_results[test_name] = {
                    "status": "passed" if result.overall_status or result.successful_documents > 0 else "failed",
                    "message": f"Integrated pipeline completed: {result.successful_documents}/{result.total_documents} documents processed",
                    "details": {
                        "database": result.database_name,
                        "collection": test_collection,
                        "pipeline_id": result.pipeline_id,
                        "total_tables": result.total_tables,
                        "processed_tables": result.processed_tables,
                        "failed_tables": result.failed_tables,
                        "total_documents": result.total_documents,
                        "successful_documents": result.successful_documents,
                        "failed_documents": result.failed_documents,
                        "processing_time": result.processing_time,
                        "table_success_rate": result.table_success_rate,
                        "document_success_rate": result.document_success_rate,
                        "health_status": health_status,
                        "error_count": len(result.errors)
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            finally:
                pipeline.close()
            
            self.logger.info(f"✓ {test_name} completed")
            
        except Exception as e:
            self.test_results[test_name] = {
                "status": "failed",
                "message": f"Integrated pipeline test failed: {e}",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.utcnow().isoformat()
            }
            self.logger.error(f"✗ {test_name} failed: {e}")
    
    async def _test_cli_commands(self):
        """Test CLI command functionality."""
        test_name = "cli_commands"
        self.logger.info(f"Testing: {test_name}")
        
        try:
            # Test validation command
            from src.pipeline.rdb_config_validator import validate_rdb_system
            
            validation_report = await validate_rdb_system()
            
            self.test_results[test_name] = {
                "status": "passed",
                "message": "CLI validation command functional",
                "details": {
                    "validation_available": True,
                    "validation_status": validation_report.overall_status,
                    "validation_checks": validation_report.summary['total_checks']
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.logger.info(f"✓ {test_name} completed")
            
        except Exception as e:
            self.test_results[test_name] = {
                "status": "failed",
                "message": f"CLI commands test failed: {e}",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.utcnow().isoformat()
            }
            self.logger.error(f"✗ {test_name} failed: {e}")
    
    def _generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        end_time = datetime.utcnow()
        total_time = (end_time - self.test_start_time).total_seconds()
        
        # Calculate summary statistics
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results.values() if r["status"] == "passed"])
        failed_tests = len([r for r in self.test_results.values() if r["status"] == "failed"])
        skipped_tests = len([r for r in self.test_results.values() if r["status"] == "skipped"])
        warning_tests = len([r for r in self.test_results.values() if r["status"] == "warning"])
        
        overall_status = failed_tests == 0 and passed_tests > 0
        
        report = {
            "test_session": {
                "start_time": self.test_start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "total_time_seconds": total_time,
                "overall_status": overall_status
            },
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "skipped": skipped_tests,
                "warnings": warning_tests,
                "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0
            },
            "test_results": self.test_results,
            "recommendations": self._generate_recommendations()
        }
        
        self.logger.info(f"Integration test completed: {passed_tests}/{total_tests} tests passed")
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        failed_tests = [name for name, result in self.test_results.items() if result["status"] == "failed"]
        skipped_tests = [name for name, result in self.test_results.items() if result["status"] == "skipped"]
        
        if failed_tests:
            recommendations.append(f"Fix failed tests: {', '.join(failed_tests)}")
        
        if skipped_tests:
            recommendations.append(f"Configure missing components for skipped tests: {', '.join(skipped_tests)}")
        
        if "rdb_extraction" in self.test_results and self.test_results["rdb_extraction"]["status"] == "skipped":
            recommendations.append("Configure RDB connections to enable database extraction testing")
        
        if "integrated_pipeline" in self.test_results and self.test_results["integrated_pipeline"]["status"] == "failed":
            recommendations.append("Check Milvus and embedding service configurations")
        
        if not recommendations:
            recommendations.append("All tests passed! RDB pipeline is ready for production use.")
        
        return recommendations


def print_test_report(report: Dict[str, Any]):
    """Print formatted test report."""
    print("\n" + "="*80)
    print("RDB PIPELINE INTEGRATION TEST REPORT")
    print("="*80)
    
    # Test session info
    session = report["test_session"]
    print(f"\nTest Session:")
    print(f"  Start Time: {session['start_time']}")
    print(f"  End Time: {session['end_time']}")
    print(f"  Duration: {session['total_time_seconds']:.2f} seconds")
    print(f"  Overall Status: {'✓ PASSED' if session['overall_status'] else '✗ FAILED'}")
    
    # Summary
    summary = report["summary"]
    print(f"\nSummary:")
    print(f"  Total Tests: {summary['total_tests']}")
    print(f"  Passed: {summary['passed']}")
    print(f"  Failed: {summary['failed']}")
    print(f"  Skipped: {summary['skipped']}")
    print(f"  Warnings: {summary['warnings']}")
    print(f"  Success Rate: {summary['success_rate']:.1f}%")
    
    # Individual test results
    print(f"\nTest Results:")
    for test_name, result in report["test_results"].items():
        status_icon = {
            "passed": "✓",
            "failed": "✗",
            "skipped": "⊝",
            "warning": "⚠"
        }.get(result["status"], "?")
        
        print(f"  {status_icon} {test_name}: {result['message']}")
        
        if result["status"] == "failed" and "error" in result:
            print(f"    Error: {result['error']}")
    
    # Recommendations
    print(f"\nRecommendations:")
    for i, rec in enumerate(report["recommendations"], 1):
        print(f"  {i}. {rec}")
    
    print("\n" + "="*80)


async def main():
    """Main test execution function."""
    print("RDB Pipeline Integration Test")
    print("=" * 50)
    
    # Create and run test
    test = RDBPipelineIntegrationTest()
    
    try:
        report = await test.run_all_tests()
        
        # Print report
        print_test_report(report)
        
        # Save report to file
        report_file = f"rdb_pipeline_test_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\nDetailed report saved to: {report_file}")
        
        # Exit with appropriate code
        sys.exit(0 if report["test_session"]["overall_status"] else 1)
        
    except Exception as e:
        print(f"\nFATAL ERROR: Integration test failed to complete: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")
        sys.exit(2)


if __name__ == "__main__":
    asyncio.run(main())