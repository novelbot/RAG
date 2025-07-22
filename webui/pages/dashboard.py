"""
Dashboard page for RAG Server Web UI
Displays system overview, statistics, and health monitoring
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
from webui.api_client import get_api_client
from webui.auth import require_auth

@require_auth
def show():
    """Display the dashboard page"""
    st.title("📊 Dashboard")
    
    api_client = get_api_client()
    
    # Get system status and metrics
    try:
        system_status = api_client.get_system_status()
        health_check = api_client.get_health_check()
        metrics = api_client.get_metrics()
    except Exception as e:
        st.error(f"Failed to load dashboard data: {str(e)}")
        system_status = {"status": "unknown"}
        health_check = {"healthy": False}
        metrics = {}
    
    # System Status Overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_color = "🟢" if health_check.get("healthy", False) else "🔴"
        st.metric(
            label="System Status",
            value=f"{status_color} {system_status.get('status', 'Unknown').title()}"
        )
    
    with col2:
        # Mock data for demo - replace with real metrics
        total_documents = metrics.get("total_documents", 1247)
        st.metric(
            label="Total Documents",
            value=f"{total_documents:,}",
            delta="12 new today"
        )
    
    with col3:
        total_queries = metrics.get("total_queries", 3456)
        st.metric(
            label="Total Queries",
            value=f"{total_queries:,}",
            delta="156 today"
        )
    
    with col4:
        active_users = metrics.get("active_users", 23)
        st.metric(
            label="Active Users",
            value=active_users,
            delta="3 online now"
        )
    
    st.markdown("---")
    
    # Charts and Analytics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Query Volume (Last 7 Days)")
        
        # Mock data for query volume chart
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=6),
            end=datetime.now(),
            freq='D'
        )
        query_counts = [45, 67, 89, 123, 98, 134, 156]  # Mock data
        
        fig_queries = px.line(
            x=dates,
            y=query_counts,
            title="Daily Query Count",
            labels={"x": "Date", "y": "Queries"}
        )
        fig_queries.update_layout(showlegend=False)
        st.plotly_chart(fig_queries, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Query Success Rate")
        
        # Mock data for success rate
        success_rate = metrics.get("query_success_rate", 0.94)
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=success_rate * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Success Rate (%)"},
            delta={'reference': 95, 'suffix': '%'},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 80], 'color': "lightgray"},
                    {'range': [80, 95], 'color': "yellow"},
                    {'range': [95, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    # System Health Details
    st.subheader("🩺 System Health")
    
    health_col1, health_col2, health_col3 = st.columns(3)
    
    with health_col1:
        st.markdown("**Database Status**")
        db_status = health_check.get("database", {})
        db_healthy = db_status.get("healthy", False)
        status_icon = "✅" if db_healthy else "❌"
        st.write(f"{status_icon} {db_status.get('status', 'Unknown')}")
        
        if db_healthy:
            st.success(f"Response time: {db_status.get('response_time', 'N/A')}ms")
        else:
            st.error(f"Error: {db_status.get('error', 'Connection failed')}")
    
    with health_col2:
        st.markdown("**Vector Database**")
        vector_status = health_check.get("milvus", {})
        vector_healthy = vector_status.get("healthy", False)
        status_icon = "✅" if vector_healthy else "❌"
        st.write(f"{status_icon} {vector_status.get('status', 'Unknown')}")
        
        if vector_healthy:
            st.success(f"Collections: {vector_status.get('collections', 0)}")
        else:
            st.error(f"Error: {vector_status.get('error', 'Connection failed')}")
    
    with health_col3:
        st.markdown("**LLM Services**")
        llm_status = health_check.get("llm", {})
        llm_healthy = llm_status.get("healthy", False)
        status_icon = "✅" if llm_healthy else "❌"
        st.write(f"{status_icon} {llm_status.get('status', 'Unknown')}")
        
        if llm_healthy:
            st.success(f"Providers: {len(llm_status.get('providers', []))}")
        else:
            st.error(f"Error: {llm_status.get('error', 'Service unavailable')}")
    
    # Recent Activity
    st.markdown("---")
    st.subheader("📝 Recent Activity")
    
    # Mock recent activity data
    recent_activities = [
        {
            "time": "2 minutes ago",
            "user": "john.doe",
            "action": "Uploaded document",
            "details": "financial_report_2024.pdf"
        },
        {
            "time": "5 minutes ago", 
            "user": "jane.smith",
            "action": "Performed query",
            "details": "What is the revenue trend?"
        },
        {
            "time": "8 minutes ago",
            "user": "admin",
            "action": "Created user",
            "details": "new.user@company.com"
        },
        {
            "time": "12 minutes ago",
            "user": "mike.jones",
            "action": "Deleted document",
            "details": "old_manual.docx"
        },
        {
            "time": "15 minutes ago",
            "user": "sarah.wilson",
            "action": "Updated settings",
            "details": "Changed LLM provider to Claude"
        }
    ]
    
    for activity in recent_activities:
        col1, col2, col3, col4 = st.columns([2, 2, 3, 3])
        with col1:
            st.text(activity["time"])
        with col2:
            st.text(activity["user"])
        with col3:
            st.text(activity["action"])
        with col4:
            st.text(activity["details"])
    
    # Resource Usage
    st.markdown("---")
    st.subheader("💻 Resource Usage")
    
    resource_col1, resource_col2, resource_col3 = st.columns(3)
    
    with resource_col1:
        st.markdown("**Memory Usage**")
        memory_usage = metrics.get("memory_usage_percent", 67)
        st.progress(memory_usage / 100)
        st.text(f"{memory_usage}% used")
    
    with resource_col2:
        st.markdown("**CPU Usage**")
        cpu_usage = metrics.get("cpu_usage_percent", 45)
        st.progress(cpu_usage / 100)
        st.text(f"{cpu_usage}% used")
    
    with resource_col3:
        st.markdown("**Storage Usage**")
        storage_usage = metrics.get("storage_usage_percent", 78)
        st.progress(storage_usage / 100)
        st.text(f"{storage_usage}% used")
    
    # Quick Actions
    st.markdown("---")
    st.subheader("⚡ Quick Actions")
    
    action_col1, action_col2, action_col3, action_col4 = st.columns(4)
    
    with action_col1:
        if st.button("📄 Upload Documents", use_container_width=True):
            st.session_state.current_page = "documents"
            st.rerun()
    
    with action_col2:
        if st.button("🔍 Query System", use_container_width=True):
            st.session_state.current_page = "query"
            st.rerun()
    
    with action_col3:
        if st.button("⚙️ Settings", use_container_width=True):
            st.session_state.current_page = "settings"
            st.rerun()
    
    with action_col4:
        user_info = st.session_state.get("user_info", {})
        if user_info.get("role") == "admin":
            if st.button("👥 Admin Panel", use_container_width=True):
                st.session_state.current_page = "admin"
                st.rerun()
        else:
            st.button("👥 Admin Panel", disabled=True, use_container_width=True, help="Admin access required")
    
    # Auto-refresh option
    st.markdown("---")
    auto_refresh = st.checkbox("🔄 Auto-refresh (30 seconds)", value=False)
    
    if auto_refresh:
        import time
        time.sleep(30)
        st.rerun()