"""
Document management page for RAG Server Web UI
Handles file uploads, document library, and processing status
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import json
from webui.api_client import get_api_client
from webui.auth import require_auth
from webui.config import config
import io

@require_auth
def show():
    """Display the documents management page"""
    st.title("📄 Document Management")
    
    api_client = get_api_client()
    
    # Create tabs for different document operations
    tab1, tab2, tab3 = st.tabs(["📤 Upload", "📚 Library", "⚙️ Processing Status"])
    
    with tab1:
        show_upload_section(api_client)
    
    with tab2:
        show_document_library(api_client)
    
    with tab3:
        show_processing_status(api_client)

def show_upload_section(api_client):
    """Display the document upload section"""
    st.subheader("📤 Upload Documents")
    
    # Upload area
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=['txt', 'pdf', 'docx', 'xlsx', 'md'],
        accept_multiple_files=True,
        help="Supported formats: TXT, PDF, Word, Excel, Markdown"
    )
    
    if uploaded_files:
        st.write(f"Selected {len(uploaded_files)} file(s)")
        
        # Display selected files
        for file in uploaded_files:
            st.write(f"📄 {file.name} ({file.size:,} bytes)")
        
        # Upload configuration
        st.subheader("Upload Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Get document categories from config
            available_categories = [cat['name'] for cat in config.get_document_categories()]
            if not available_categories:
                available_categories = ["General", "Technical", "Financial", "Legal", "Marketing", "HR"]
            
            category = st.selectbox(
                "Document Category",
                available_categories
            )
            
            tags = st.text_input(
                "Tags (comma-separated)",
                placeholder="project, quarterly, analysis"
            )
        
        with col2:
            access_level = st.selectbox(
                "Access Level",
                ["Public", "Internal", "Restricted", "Confidential"]
            )
            
            department = st.selectbox(
                "Department",
                ["IT", "Finance", "HR", "Marketing", "Legal", "Operations"]
            )
        
        # Additional metadata
        with st.expander("Additional Metadata (Optional)"):
            author = st.text_input("Author")
            description = st.text_area("Description")
            custom_fields = st.text_area(
                "Custom Fields (JSON format)",
                placeholder='{"project": "Q1-2024", "version": "1.0"}'
            )
        
        # Upload button
        if st.button("🚀 Upload Documents", type="primary", use_container_width=True):
            upload_documents(api_client, uploaded_files, {
                "category": category,
                "tags": [tag.strip() for tag in tags.split(",") if tag.strip()],
                "access_level": access_level,
                "department": department,
                "author": author,
                "description": description,
                "custom_fields": custom_fields
            })

def upload_documents(api_client, files, metadata):
    """Handle document upload process"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    uploaded_count = 0
    failed_uploads = []
    
    for i, file in enumerate(files):
        try:
            status_text.text(f"Uploading {file.name}...")
            
            # Prepare metadata for this file
            file_metadata = metadata.copy()
            file_metadata["filename"] = file.name
            file_metadata["file_size"] = file.size
            file_metadata["upload_timestamp"] = datetime.now().isoformat()
            
            # Parse custom fields JSON if provided
            if file_metadata.get("custom_fields"):
                try:
                    file_metadata["custom_fields"] = json.loads(file_metadata["custom_fields"])
                except json.JSONDecodeError:
                    file_metadata["custom_fields"] = {}
            
            # Read file content
            file_content = file.read()
            
            # Upload via API (mock implementation for demo)
            # result = api_client.upload_document(file_content, file.name, file_metadata)
            
            # Mock success for demo
            uploaded_count += 1
            
        except Exception as e:
            failed_uploads.append(f"{file.name}: {str(e)}")
        
        # Update progress
        progress_bar.progress((i + 1) / len(files))
    
    # Show results
    status_text.empty()
    progress_bar.empty()
    
    if uploaded_count > 0:
        st.success(f"✅ Successfully uploaded {uploaded_count} document(s)")
    
    if failed_uploads:
        st.error("❌ Failed uploads:")
        for error in failed_uploads:
            st.write(f"• {error}")

def show_document_library(api_client):
    """Display the document library"""
    st.subheader("📚 Document Library")
    
    # Search and filter controls
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        search_query = st.text_input("🔍 Search documents", placeholder="Enter keywords...")
    
    with col2:
        # Get document categories from config for filter
        available_categories = [cat['name'] for cat in config.get_document_categories()]
        if not available_categories:
            available_categories = ["General", "Technical", "Financial", "Legal", "Marketing", "HR"]
        category_filter_options = ["All"] + available_categories
        
        category_filter = st.selectbox(
            "Category",
            category_filter_options
        )
    
    with col3:
        sort_by = st.selectbox(
            "Sort by",
            ["Upload Date", "File Name", "File Size", "Category"]
        )
    
    # Get documents (mock data for demo)
    documents = get_mock_documents()
    
    # Apply filters
    filtered_docs = documents
    if search_query:
        filtered_docs = [doc for doc in documents 
                        if search_query.lower() in doc["name"].lower() 
                        or search_query.lower() in doc["description"].lower()]
    
    if category_filter != "All":
        filtered_docs = [doc for doc in filtered_docs if doc["category"] == category_filter]
    
    # Display documents
    if not filtered_docs:
        st.info("No documents found matching your criteria.")
        return
    
    st.write(f"Found {len(filtered_docs)} document(s)")
    
    # Documents table
    for i, doc in enumerate(filtered_docs):
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 1, 1, 2])
            
            with col1:
                st.write(f"**{doc['name']}**")
                st.write(doc["description"])
                st.write(f"Tags: {', '.join(doc['tags'])}")
            
            with col2:
                st.write(f"**Category:**")
                st.write(doc["category"])
                st.write(f"**Size:**")
                st.write(f"{doc['size']:,} bytes")
            
            with col3:
                st.write(f"**Status:**")
                status_color = {"Processed": "🟢", "Processing": "🟡", "Failed": "🔴"}
                st.write(f"{status_color.get(doc['status'], '⚪')} {doc['status']}")
                st.write(f"**Uploaded:**")
                st.write(doc["upload_date"])
            
            with col4:
                action_col1, action_col2 = st.columns(2)
                
                with action_col1:
                    if st.button("👁️ View", key=f"view_{i}", use_container_width=True):
                        show_document_details(doc)
                
                with action_col2:
                    if st.button("🗑️ Delete", key=f"delete_{i}", use_container_width=True):
                        delete_document(api_client, doc)
            
            st.markdown("---")

def show_processing_status(api_client):
    """Display processing status of documents"""
    st.subheader("⚙️ Processing Status")
    
    # Get processing queue (mock data for demo)
    processing_queue = get_mock_processing_queue()
    
    if not processing_queue:
        st.info("No documents currently processing.")
        return
    
    st.write(f"{len(processing_queue)} document(s) in processing queue")
    
    # Processing queue table
    df = pd.DataFrame(processing_queue)
    
    # Add progress bars for processing items
    for i, item in enumerate(processing_queue):
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"**{item['filename']}**")
                if item['status'] == 'Processing':
                    st.progress(item['progress'] / 100)
                    st.write(f"Stage: {item['current_stage']}")
                else:
                    st.write(f"Status: {item['status']}")
            
            with col2:
                st.write(f"**Started:**")
                st.write(item['start_time'])
                if item['status'] == 'Processing':
                    st.write(f"**Progress:**")
                    st.write(f"{item['progress']}%")
            
            with col3:
                st.write(f"**Queue Position:**")
                st.write(f"#{i+1}")
                if item['status'] == 'Failed':
                    st.error(f"Error: {item.get('error', 'Unknown error')}")
            
            # Retry button for failed items
            if item['status'] == 'Failed':
                if st.button(f"🔄 Retry", key=f"retry_{i}"):
                    retry_processing(api_client, item['id'])
            
            st.markdown("---")
    
    # Refresh button
    if st.button("🔄 Refresh Status", use_container_width=True):
        st.rerun()

def show_document_details(document):
    """Show detailed document information in a modal-like display"""
    st.subheader(f"📄 {document['name']}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**File Information:**")
        st.write(f"• **Name:** {document['name']}")
        st.write(f"• **Category:** {document['category']}")
        st.write(f"• **Size:** {document['size']:,} bytes")
        st.write(f"• **Status:** {document['status']}")
        st.write(f"• **Upload Date:** {document['upload_date']}")
        
        st.write("**Tags:**")
        for tag in document['tags']:
            st.write(f"• {tag}")
    
    with col2:
        st.write("**Metadata:**")
        st.write(f"• **Author:** {document.get('author', 'N/A')}")
        st.write(f"• **Department:** {document.get('department', 'N/A')}")
        st.write(f"• **Access Level:** {document.get('access_level', 'N/A')}")
        st.write(f"• **Description:** {document['description']}")
        
        if document.get('custom_fields'):
            st.write("**Custom Fields:**")
            for key, value in document['custom_fields'].items():
                st.write(f"• **{key}:** {value}")

def delete_document(api_client, document):
    """Handle document deletion"""
    if st.button(f"Confirm delete {document['name']}?", key=f"confirm_delete_{document['id']}"):
        try:
            # api_client.delete_document(document['id'])
            st.success(f"✅ Deleted {document['name']}")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Failed to delete document: {str(e)}")

def retry_processing(api_client, document_id):
    """Retry processing a failed document"""
    try:
        # Mock retry functionality
        st.success("✅ Document queued for reprocessing")
        st.rerun()
    except Exception as e:
        st.error(f"❌ Failed to retry processing: {str(e)}")

def get_mock_documents():
    """Generate mock document data for demo"""
    return [
        {
            "id": "doc_001",
            "name": "quarterly_report_q1_2024.pdf",
            "description": "Q1 2024 financial quarterly report with revenue analysis",
            "category": "Financial",
            "size": 2587463,
            "status": "Processed",
            "upload_date": "2024-01-15 14:30",
            "tags": ["quarterly", "finance", "2024", "revenue"],
            "author": "John Smith",
            "department": "Finance",
            "access_level": "Internal",
            "custom_fields": {"quarter": "Q1", "year": "2024"}
        },
        {
            "id": "doc_002",
            "name": "employee_handbook.docx",
            "description": "Updated employee handbook with new policies",
            "category": "HR",
            "size": 1234567,
            "status": "Processing",
            "upload_date": "2024-01-16 09:15",
            "tags": ["hr", "policies", "handbook", "employees"],
            "author": "Sarah Wilson",
            "department": "HR",
            "access_level": "Internal",
            "custom_fields": {"version": "2024.1"}
        },
        {
            "id": "doc_003",
            "name": "api_documentation.md",
            "description": "Technical documentation for REST API endpoints",
            "category": "Technical",
            "size": 567890,
            "status": "Processed",
            "upload_date": "2024-01-16 11:45",
            "tags": ["api", "documentation", "technical", "endpoints"],
            "author": "Mike Jones",
            "department": "IT",
            "access_level": "Public",
            "custom_fields": {"api_version": "v2.1"}
        },
        {
            "id": "doc_004",
            "name": "marketing_campaign_analysis.xlsx",
            "description": "Analysis of marketing campaign performance metrics",
            "category": "Marketing",
            "size": 987654,
            "status": "Failed",
            "upload_date": "2024-01-16 16:20",
            "tags": ["marketing", "campaign", "analysis", "metrics"],
            "author": "Lisa Brown",
            "department": "Marketing",
            "access_level": "Internal",
            "custom_fields": {"campaign": "Winter2024"}
        }
    ]

def get_mock_processing_queue():
    """Generate mock processing queue data"""
    return [
        {
            "id": "proc_001",
            "filename": "large_dataset.pdf",
            "status": "Processing",
            "progress": 67,
            "current_stage": "Text Extraction",
            "start_time": "2024-01-16 17:30",
            "estimated_completion": "2024-01-16 17:45"
        },
        {
            "id": "proc_002", 
            "filename": "contract_template.docx",
            "status": "Processing",
            "progress": 23,
            "current_stage": "Document Parsing",
            "start_time": "2024-01-16 17:35",
            "estimated_completion": "2024-01-16 17:40"
        },
        {
            "id": "proc_003",
            "filename": "corrupted_file.pdf",
            "status": "Failed",
            "progress": 0,
            "current_stage": "Validation",
            "start_time": "2024-01-16 17:20",
            "error": "File appears to be corrupted or unsupported format"
        }
    ]