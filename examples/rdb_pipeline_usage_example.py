#!/usr/bin/env python3
"""
RDB Pipeline Usage Example

This example demonstrates how to use the RDB to Vector pipeline functionality
for extracting data from relational databases and converting to vector embeddings.
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.config import DatabaseConfig, DatabaseType, get_config
from src.extraction.base import ExtractionMode
from src.pipeline.rdb_pipeline import create_rdb_vector_pipeline, RDBPipelineConfig
from src.pipeline.rdb_adapter import RDBAdapterConfig
from src.pipeline.rdb_config_validator import validate_rdb_system, test_database_connection


async def example_1_system_validation():
    """Example 1: Validate RDB pipeline system configuration."""
    print("="*60)
    print("Example 1: System Validation")
    print("="*60)
    
    # Validate complete system
    print("Validating RDB pipeline system...")
    validation_report = await validate_rdb_system()
    
    print(f"Overall Status: {'✓ PASSED' if validation_report.overall_status else '✗ FAILED'}")
    print(f"Total Checks: {validation_report.summary['total_checks']}")
    print(f"Passed: {validation_report.summary['passed']}")
    print(f"Failed: {validation_report.summary['failed']}")
    
    if validation_report.critical_errors:
        print("\nCritical Errors:")
        for error in validation_report.critical_errors:
            print(f"  • {error.message}")
    
    if validation_report.errors:
        print("\nErrors:")
        for error in validation_report.errors[:3]:  # Show first 3
            print(f"  • {error.message}")
    
    if validation_report.warnings:
        print("\nWarnings:")
        for warning in validation_report.warnings[:3]:  # Show first 3
            print(f"  • {warning.message}")
    
    return validation_report.overall_status


async def example_2_test_database_connection():
    """Example 2: Test specific database connection."""
    print("\n" + "="*60)
    print("Example 2: Database Connection Test")
    print("="*60)
    
    try:
        config = get_config()
        
        if not hasattr(config, 'rdb_connections') or not config.rdb_connections:
            print("No RDB connections configured. Please configure database connections first.")
            return False
        
        # Test first configured database
        db_name = list(config.rdb_connections.keys())[0]
        db_config = config.rdb_connections[db_name]
        
        print(f"Testing database connection: {db_name}")
        print(f"Host: {db_config.host}")
        print(f"Database: {db_config.database}")
        
        # Test connection
        test_result = await test_database_connection(db_config)
        
        print(f"Connection Status: {'✓ PASSED' if test_result.overall_status else '✗ FAILED'}")
        
        for result in test_result.results:
            status_icon = "✓" if result.status else "✗"
            print(f"  {status_icon} {result.check_name}: {result.message}")
        
        return test_result.overall_status
        
    except Exception as e:
        print(f"Database connection test failed: {e}")
        return False


async def example_3_simple_extraction():
    """Example 3: Simple RDB data extraction."""
    print("\n" + "="*60)
    print("Example 3: Simple RDB Data Extraction")
    print("="*60)
    
    try:
        config = get_config()
        
        if not hasattr(config, 'rdb_connections') or not config.rdb_connections:
            print("No RDB connections configured. Skipping extraction example.")
            return False
        
        # Use first configured database
        db_name = list(config.rdb_connections.keys())[0]
        db_config = config.rdb_connections[db_name]
        
        print(f"Extracting data from database: {db_name}")
        
        # Create adapter config for simple extraction
        adapter_config = RDBAdapterConfig(
            content_format="structured",
            include_table_name=True,
            include_column_names=True,
            exclude_null_values=True,
            max_content_length=1000
        )
        
        # Create RDB pipeline
        pipeline = create_rdb_vector_pipeline(
            database_name=db_name,
            database_config=db_config,
            collection_name="example_documents",
            adapter_config=adapter_config,
            extraction_mode=ExtractionMode.FULL,
            extraction_batch_size=10,
            max_rows_per_table=5,  # Limit for example
            continue_on_table_error=True,
            enable_detailed_logging=True
        )
        
        try:
            # Check pipeline health
            health_status = await pipeline.health_check()
            print(f"Pipeline Health: {'Healthy' if health_status.get('pipeline_healthy', False) else 'Unhealthy'}")
            
            # Process database
            print("Processing database through RDB pipeline...")
            result = await pipeline.process_all_tables()
            
            print(f"\nExtraction Results:")
            print(f"  Database: {result.database_name}")
            print(f"  Tables processed: {result.processed_tables}/{result.total_tables}")
            print(f"  Documents processed: {result.successful_documents}/{result.total_documents}")
            print(f"  Processing time: {result.processing_time:.2f} seconds")
            print(f"  Table success rate: {result.table_success_rate:.1f}%")
            print(f"  Document success rate: {result.document_success_rate:.1f}%")
            
            if result.errors:
                print(f"  Errors: {len(result.errors)}")
                for error in result.errors[:3]:  # Show first 3
                    print(f"    • {error.get('error', 'Unknown error')}")
            
            return result.successful_documents > 0
            
        finally:
            pipeline.close()
            
    except Exception as e:
        print(f"Simple extraction failed: {e}")
        import traceback
        print(f"Error details: {traceback.format_exc()}")
        return False


async def example_4_custom_configuration():
    """Example 4: Custom pipeline configuration."""
    print("\n" + "="*60)
    print("Example 4: Custom Pipeline Configuration")
    print("="*60)
    
    try:
        config = get_config()
        
        if not hasattr(config, 'rdb_connections') or not config.rdb_connections:
            print("No RDB connections configured. Skipping custom configuration example.")
            return False
        
        # Use first configured database
        db_name = list(config.rdb_connections.keys())[0]
        db_config = config.rdb_connections[db_name]
        
        print("Creating pipeline with custom configuration...")
        
        # Custom adapter configuration
        custom_adapter_config = RDBAdapterConfig(
            content_format="json",  # JSON format instead of structured
            include_table_name=False,
            include_column_names=True,
            exclude_null_values=True,
            exclude_empty_strings=True,
            exclude_columns=["id", "created_at", "updated_at"],  # Exclude common system columns
            max_content_length=5000,
            truncate_long_content=True,
            id_prefix="custom_rdb",
            include_timestamp_in_id=True
        )
        
        # Create pipeline configuration
        pipeline_config = RDBPipelineConfig(
            database_name=db_name,
            database_config=db_config,
            extraction_mode=ExtractionMode.FULL,
            adapter_config=custom_adapter_config,
            extraction_batch_size=20,
            max_rows_per_table=10,  # Small limit for example
            max_concurrent_tables=2,
            collection_name="custom_rdb_documents",
            continue_on_table_error=True,
            continue_on_pipeline_error=True,
            enable_detailed_logging=True
        )
        
        # Create pipeline with custom config
        from src.pipeline.rdb_pipeline import RDBVectorPipeline
        pipeline = RDBVectorPipeline(pipeline_config)
        
        try:
            # Get pipeline status
            status = pipeline.get_status()
            print(f"Pipeline Status:")
            print(f"  Database: {status['database_name']}")
            print(f"  Collection: {status['collection_name']}")
            print(f"  Extraction Mode: {status['extraction_mode']}")
            print(f"  Max Concurrent Tables: {status['pipeline_config']['max_concurrent_tables']}")
            
            # Process with custom configuration
            print("\nProcessing with custom configuration...")
            result = await pipeline.process_all_tables()
            
            print(f"\nCustom Processing Results:")
            print(f"  Tables processed: {result.processed_tables}/{result.total_tables}")
            print(f"  Documents created: {result.successful_documents}")
            print(f"  Processing time: {result.processing_time:.2f}s")
            
            # Show table-specific results
            if result.table_results:
                print(f"\nTable Results:")
                for table_name, table_result in list(result.table_results.items())[:3]:  # First 3 tables
                    status = "✓" if table_result.get("success", False) else "✗"
                    doc_count = table_result.get("document_count", 0)
                    print(f"  {status} {table_name}: {doc_count} documents")
            
            return result.successful_documents > 0
            
        finally:
            pipeline.close()
            
    except Exception as e:
        print(f"Custom configuration example failed: {e}")
        import traceback
        print(f"Error details: {traceback.format_exc()}")
        return False


async def example_5_cli_simulation():
    """Example 5: Simulate CLI command usage."""
    print("\n" + "="*60)
    print("Example 5: CLI Command Simulation")
    print("="*60)
    
    print("This example shows how the CLI commands would work:")
    print()
    
    print("1. Validate system configuration:")
    print("   $ rag-cli data validate")
    print("   $ rag-cli data validate --database mydb --detailed")
    print()
    
    print("2. Ingest data from database:")
    print("   $ rag-cli data ingest --path /path/to/database")
    print("   $ rag-cli data ingest --path . --batch-size 100 --force")
    print()
    
    print("3. Check data status:")
    print("   $ rag-cli data status")
    print()
    
    print("4. Sync data sources:")
    print("   $ rag-cli data sync --source database")
    print("   $ rag-cli data sync --source all --full")
    print()
    
    # Demonstrate validation command programmatically
    print("Running validation command simulation...")
    try:
        validation_report = await validate_rdb_system()
        print(f"✓ Validation command functional: {validation_report.summary['passed']}/{validation_report.summary['total_checks']} checks passed")
        return True
    except Exception as e:
        print(f"✗ Validation command failed: {e}")
        return False


async def main():
    """Main example execution function."""
    print("RDB Pipeline Usage Examples")
    print("=" * 50)
    print("This script demonstrates various ways to use the RDB to Vector pipeline.")
    print()
    
    examples = [
        ("System Validation", example_1_system_validation),
        ("Database Connection Test", example_2_test_database_connection),
        ("Simple Data Extraction", example_3_simple_extraction),
        ("Custom Configuration", example_4_custom_configuration),
        ("CLI Commands", example_5_cli_simulation)
    ]
    
    results = {}
    
    for name, example_func in examples:
        try:
            print(f"\nRunning: {name}")
            success = await example_func()
            results[name] = success
            print(f"Result: {'✓ SUCCESS' if success else '✗ FAILED'}")
        except Exception as e:
            print(f"Result: ✗ ERROR - {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*60)
    print("EXAMPLE EXECUTION SUMMARY")
    print("="*60)
    
    total_examples = len(results)
    successful_examples = sum(results.values())
    
    print(f"Total Examples: {total_examples}")
    print(f"Successful: {successful_examples}")
    print(f"Failed: {total_examples - successful_examples}")
    print(f"Success Rate: {(successful_examples / total_examples * 100):.1f}%")
    
    print("\nDetailed Results:")
    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {name}")
    
    if successful_examples == total_examples:
        print("\n🎉 All examples completed successfully!")
        print("The RDB pipeline is ready for use.")
    else:
        print(f"\n⚠️  {total_examples - successful_examples} examples failed.")
        print("Please check the configuration and try again.")
    
    print("\nNext Steps:")
    print("1. Configure your database connections in the application config")
    print("2. Run 'rag-cli data validate' to check your setup")
    print("3. Use 'rag-cli data ingest' to process your databases")
    print("4. Monitor the vector database for your embedded documents")


if __name__ == "__main__":
    asyncio.run(main())