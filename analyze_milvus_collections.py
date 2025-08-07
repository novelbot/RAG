#!/usr/bin/env python3
"""
Milvus Collection Analyzer
Analyzes all collections in the Milvus database and displays their entry counts and schemas.
"""

import os
from typing import Dict, List, Any
from pymilvus import (
    connections, 
    Collection, 
    utility,
    DataType
)
from dotenv import load_dotenv
from tabulate import tabulate
import json

# Load environment variables
load_dotenv()

def connect_to_milvus():
    """Connect to Milvus database using credentials from .env"""
    host = os.getenv('MILVUS_HOST', 'localhost')
    port = os.getenv('MILVUS_PORT', '19530')
    user = os.getenv('MILVUS_USER', '')
    password = os.getenv('MILVUS_PASSWORD', '')
    
    print(f"Connecting to Milvus at {host}:{port}...")
    
    connection_args = {
        "alias": "default",
        "host": host,
        "port": port
    }
    
    if user and password:
        connection_args["user"] = user
        connection_args["password"] = password
    
    connections.connect(**connection_args)
    print("✅ Connected to Milvus successfully\n")

def get_data_type_name(dtype: DataType) -> str:
    """Convert DataType enum to readable string"""
    type_map = {
        DataType.BOOL: "BOOL",
        DataType.INT8: "INT8",
        DataType.INT16: "INT16",
        DataType.INT32: "INT32",
        DataType.INT64: "INT64",
        DataType.FLOAT: "FLOAT",
        DataType.DOUBLE: "DOUBLE",
        DataType.STRING: "STRING",
        DataType.VARCHAR: "VARCHAR",
        DataType.JSON: "JSON",
        DataType.ARRAY: "ARRAY",
        DataType.FLOAT_VECTOR: "FLOAT_VECTOR",
        DataType.BINARY_VECTOR: "BINARY_VECTOR",
        DataType.FLOAT16_VECTOR: "FLOAT16_VECTOR",
        DataType.BFLOAT16_VECTOR: "BFLOAT16_VECTOR",
        DataType.SPARSE_FLOAT_VECTOR: "SPARSE_FLOAT_VECTOR"
    }
    return type_map.get(dtype, str(dtype))

def analyze_collection(collection_name: str) -> Dict[str, Any]:
    """Analyze a single collection and return its details"""
    try:
        collection = Collection(collection_name)
        collection.load()
        
        # Get collection schema
        schema = collection.schema
        fields_info = []
        
        for field in schema.fields:
            field_info = {
                "name": field.name,
                "type": get_data_type_name(field.dtype),
                "is_primary": field.is_primary,
                "auto_id": field.auto_id if hasattr(field, 'auto_id') else False,
                "description": field.description if field.description else ""
            }
            
            # Add dimension info for vector fields
            if hasattr(field, 'params') and field.params:
                if 'dim' in field.params:
                    field_info["dimension"] = field.params['dim']
                if 'max_length' in field.params:
                    field_info["max_length"] = field.params['max_length']
            
            fields_info.append(field_info)
        
        # Get collection statistics
        stats = collection.num_entities
        
        # Get index information
        indexes = []
        for field in schema.fields:
            if field.dtype in [DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR, 
                              DataType.FLOAT16_VECTOR, DataType.BFLOAT16_VECTOR]:
                index_info = collection.index(field_name=field.name)
                if index_info:
                    indexes.append({
                        "field": field.name,
                        "index_type": index_info.params.get('index_type', 'Unknown'),
                        "metric_type": index_info.params.get('metric_type', 'Unknown')
                    })
        
        # Get partitions
        partitions = collection.partitions
        partition_info = []
        for partition in partitions:
            partition_info.append({
                "name": partition.name,
                "num_entities": partition.num_entities
            })
        
        return {
            "name": collection_name,
            "description": schema.description if schema.description else "",
            "num_entities": stats,
            "fields": fields_info,
            "indexes": indexes,
            "partitions": partition_info,
            "auto_id": schema.auto_id if hasattr(schema, 'auto_id') else False
        }
    
    except Exception as e:
        return {
            "name": collection_name,
            "error": str(e)
        }

def display_collection_info(info: Dict[str, Any]):
    """Display collection information in a formatted way"""
    if "error" in info:
        print(f"❌ Error analyzing collection '{info['name']}': {info['error']}")
        return
    
    print(f"📊 Collection: {info['name']}")
    print(f"   Description: {info['description'] if info['description'] else 'No description'}")
    print(f"   Total Entries: {info['num_entities']:,}")
    print(f"   Auto ID: {info['auto_id']}")
    
    # Display schema
    print("\n   Schema:")
    field_table = []
    for field in info['fields']:
        row = [
            field['name'],
            field['type'],
            "✓" if field['is_primary'] else "",
            "✓" if field['auto_id'] else "",
            field.get('dimension', ''),
            field.get('max_length', ''),
            field['description'][:30] if field['description'] else ''
        ]
        field_table.append(row)
    
    headers = ["Field", "Type", "Primary", "Auto ID", "Dimension", "Max Length", "Description"]
    print(tabulate(field_table, headers=headers, tablefmt="grid", colalign=("left",)*7))
    
    # Display indexes
    if info['indexes']:
        print("\n   Indexes:")
        index_table = []
        for idx in info['indexes']:
            index_table.append([idx['field'], idx['index_type'], idx['metric_type']])
        print(tabulate(index_table, headers=["Field", "Index Type", "Metric Type"], tablefmt="grid"))
    
    # Display partitions
    if len(info['partitions']) > 1:  # More than just the default partition
        print("\n   Partitions:")
        partition_table = []
        for part in info['partitions']:
            partition_table.append([part['name'], f"{part['num_entities']:,}"])
        print(tabulate(partition_table, headers=["Name", "Entities"], tablefmt="grid"))
    
    print("\n" + "="*80 + "\n")

def main():
    """Main function to analyze all Milvus collections"""
    try:
        # Connect to Milvus
        connect_to_milvus()
        
        # Get all collections
        collections = utility.list_collections()
        
        if not collections:
            print("📭 No collections found in the Milvus database.")
            return
        
        print(f"Found {len(collections)} collection(s) in Milvus:\n")
        print("="*80 + "\n")
        
        # Summary statistics
        total_entries = 0
        collection_summaries = []
        
        # Analyze each collection
        for collection_name in collections:
            info = analyze_collection(collection_name)
            display_collection_info(info)
            
            if "error" not in info:
                total_entries += info['num_entities']
                collection_summaries.append([
                    collection_name,
                    f"{info['num_entities']:,}",
                    len(info['fields']),
                    len(info['partitions'])
                ])
        
        # Display summary
        print("\n" + "="*80)
        print("\n📈 SUMMARY")
        print("="*80 + "\n")
        
        if collection_summaries:
            print(tabulate(
                collection_summaries,
                headers=["Collection", "Entries", "Fields", "Partitions"],
                tablefmt="grid"
            ))
            
            print(f"\n📊 Total entries across all collections: {total_entries:,}")
        
        # Export to JSON (optional)
        export_json = input("\n💾 Export detailed analysis to JSON? (y/n): ").strip().lower()
        if export_json == 'y':
            all_collections = []
            for collection_name in collections:
                info = analyze_collection(collection_name)
                all_collections.append(info)
            
            output_file = "milvus_collections_analysis.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_collections, f, indent=2, ensure_ascii=False, default=str)
            print(f"✅ Analysis exported to {output_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Disconnect from Milvus
        if connections.has_connection("default"):
            connections.disconnect("default")
            print("\n👋 Disconnected from Milvus")

if __name__ == "__main__":
    main()