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
        with st.spinner("Loading dashboard data..."):
            system_status = api_client.get_system_status()
            st.write("✅ System status loaded")
            
            health_check = api_client.get_health_check()
            st.write("✅ Health check loaded")
            
            metrics = api_client.get_metrics()
            st.write("✅ Metrics loaded")
            
    except Exception as e:
        st.error(f"Failed to load dashboard data: {str(e)}")
        st.write(f"Exception type: {type(e).__name__}")
        st.write(f"Exception details: {e}")
        import traceback
        st.code(traceback.format_exc())
        
        # Fallback data
        system_status = {"status": "unknown", "services": {}}
        health_check = {"healthy": False}
        metrics = {"application_metrics": {}, "resource_usage": {}}
    
    # System Status Overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Fix: Use 'overall_status' field from system status API response
        overall_status = system_status.get('overall_status', 'unknown')
        status_color = "🟢" if overall_status == "operational" else "🟡" if overall_status == "degraded" else "🔴"
        st.metric(
            label="System Status",
            value=f"{status_color} {overall_status.title()}"
        )
    
    with col2:
        # Get data from real metrics
        app_metrics = metrics.get("application_metrics", {})
        total_documents = app_metrics.get("total_documents", 0)
        
        # Calculate delta from query trends if available
        try:
            query_trends = api_client.get_query_trends(days=2)
            daily_trends = query_trends.get("daily_trends", [])
            if len(daily_trends) >= 2:
                today_docs = daily_trends[-1].get("query_count", 0) if daily_trends else 0
                yesterday_docs = daily_trends[-2].get("query_count", 0) if len(daily_trends) > 1 else 0
                doc_delta = today_docs - yesterday_docs
                delta_text = f"{doc_delta:+d} from yesterday" if doc_delta != 0 else "No change"
            else:
                delta_text = "No historical data"
        except:
            delta_text = None
        
        st.metric(
            label="Total Documents",
            value=f"{total_documents:,}",
            delta=delta_text
        )
    
    with col3:
        total_queries = app_metrics.get("total_queries", 0)
        
        # Calculate query delta from trends
        try:
            query_trends = api_client.get_query_trends(days=2)
            daily_trends = query_trends.get("daily_trends", [])
            if len(daily_trends) >= 1:
                today_queries = daily_trends[-1].get("query_count", 0) if daily_trends else 0
                query_delta_text = f"{today_queries} today"
            else:
                query_delta_text = "No data today"
        except:
            query_delta_text = None
        
        st.metric(
            label="Total Queries",
            value=f"{total_queries:,}",
            delta=query_delta_text
        )
    
    with col4:
        active_users = app_metrics.get("active_users", 0)
        
        # Get real user activity stats
        try:
            user_activity = api_client.get_user_activity_stats()
            active_now = user_activity.get("active_users", {}).get("last_30_minutes", 0)
            user_delta_text = f"{active_now} active now"
        except:
            user_delta_text = None
        
        st.metric(
            label="Active Users",
            value=active_users,
            delta=user_delta_text
        )
    
    st.markdown("---")
    
    # Charts and Analytics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Query Volume (Last 7 Days)")
        
        # Get real query trend data
        try:
            query_trends = api_client.get_query_trends(days=7)
            daily_trends = query_trends.get("daily_trends", [])
            
            if daily_trends:
                # Extract dates and counts from real data
                dates = [trend['date'] for trend in daily_trends]
                query_counts = [trend['query_count'] for trend in daily_trends]
                
                # Convert dates to datetime for proper plotting
                dates = pd.to_datetime(dates)
            else:
                # Fallback to empty chart if no data
                dates = pd.date_range(
                    start=datetime.now() - timedelta(days=6),
                    end=datetime.now(),
                    freq='D'
                )
                query_counts = [0] * 7
        except Exception as e:
            st.error(f"Failed to load query trends: {e}")
            # Fallback data
            dates = pd.date_range(
                start=datetime.now() - timedelta(days=6),
                end=datetime.now(),
                freq='D'
            )
            query_counts = [0] * 7
        
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
        
        # Get success rate from real metrics
        app_metrics = metrics.get("application_metrics", {})
        success_rate = app_metrics.get("query_success_rate", 0.94)
        
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
    
    # Get detailed service status from system_status
    services = system_status.get("services", {})
    
    with health_col1:
        st.markdown("**Database Status**")
        db_status = services.get("database", {})
        db_healthy = db_status.get("status") == "connected"
        status_icon = "✅" if db_healthy else "❌"
        st.write(f"{status_icon} {db_status.get('status', 'Unknown').title()}")
        
        if db_healthy:
            st.success(f"Response time: {db_status.get('response_time_ms', 'N/A')}ms")
        else:
            st.error(f"Error: {db_status.get('error', 'Connection failed')}")
    
    with health_col2:
        st.markdown("**Vector Database**")
        vector_status = services.get("vector_database", {})
        vector_healthy = vector_status.get("status") == "connected"
        status_icon = "✅" if vector_healthy else "❌"
        st.write(f"{status_icon} {vector_status.get('status', 'Unknown').title()}")
        
        if vector_healthy:
            st.success(f"Collections: {vector_status.get('collection_count', 0)}")
        else:
            st.error(f"Error: {vector_status.get('error', 'Connection failed')}")
    
    with health_col3:
        st.markdown("**LLM Services**")
        llm_providers = services.get("llm_providers", {})
        llm_healthy = any(provider.get("status") == "available" for provider in llm_providers.values())
        status_icon = "✅" if llm_healthy else "❌"
        st.write(f"{status_icon} {'Available' if llm_healthy else 'Unavailable'}")
        
        if llm_healthy:
            st.success(f"Providers: {len([p for p in llm_providers.values() if p.get('status') == 'available'])}")
        else:
            st.error(f"Error: Service unavailable")
    
    # Recent Activity
    st.markdown("---")
    st.subheader("📝 Recent Activity")
    
    # Get real recent activity data
    try:
        recent_activities = api_client.get_recent_activity(limit=20)
        
        if recent_activities:
            # Display header
            col1, col2, col3, col4 = st.columns([2, 2, 3, 3])
            with col1:
                st.markdown("**Time**")
            with col2:
                st.markdown("**User**") 
            with col3:
                st.markdown("**Action**")
            with col4:
                st.markdown("**Details**")
            
            st.markdown("---")
            
            # Display activities
            for activity in recent_activities[:10]:  # Show top 10
                col1, col2, col3, col4 = st.columns([2, 2, 3, 3])
                with col1:
                    st.text(activity.get("time", "Unknown"))
                with col2:
                    st.text(activity.get("user", "Unknown"))
                with col3:
                    st.text(activity.get("action", "Unknown"))
                with col4:
                    st.text(activity.get("details", ""))
        else:
            st.info("No recent activity data available. Activities will appear here as users interact with the system.")
            
    except Exception as e:
        st.error(f"Failed to load recent activity: {e}")
        st.info("Unable to load recent activity. Please check system connectivity.")
    
    # Resource Usage
    st.markdown("---")
    st.subheader("💻 Resource Usage")
    
    resource_col1, resource_col2, resource_col3 = st.columns(3)
    
    resource_metrics = metrics.get("resource_usage", {})
    
    with resource_col1:
        st.markdown("**Memory Usage**")
        memory_usage = resource_metrics.get("memory_usage_percent", 67)
        st.progress(memory_usage / 100)
        st.text(f"{memory_usage}% used")
    
    with resource_col2:
        st.markdown("**CPU Usage**")
        cpu_usage = resource_metrics.get("cpu_usage_percent", 45)
        st.progress(cpu_usage / 100)
        st.text(f"{cpu_usage}% used")
    
    with resource_col3:
        st.markdown("**Storage Usage**")
        storage_usage = resource_metrics.get("storage_usage_percent", 78)
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