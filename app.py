"""
Refactored ONA Dashboard - Main Streamlit Application

A modular organizational network analysis dashboard.
"""

import streamlit as st
import pandas as pd
import networkx as nx
import time
import json
from datetime import datetime
from typing import Dict, List

# Import our modular components
from src.ona_dashboard import (
    DashboardConfig,
    ONASimulator,
    calculate_ona_metrics,
    create_ona_network_plot,
    create_centrality_analysis,
    create_department_analysis,
    create_network_health_dashboard,
    create_talent_risk_analysis,
    create_engagement_dashboard,
    create_influence_network_analysis
)
from src.ona_dashboard.simulation.ergm_simulator import get_default_simulation_params, get_quick_sample_params


def initialize_app():
    """Initialize the Streamlit application."""
    DashboardConfig.configure_streamlit_page()
    
    # Initialize simulator
    if 'ona_simulator' not in st.session_state:
        st.session_state.ona_simulator = ONASimulator(DashboardConfig.DEFAULT_N_EMPLOYEES)


def render_sidebar():
    """Render the sidebar navigation."""
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox("Select View", DashboardConfig.NAVIGATION_PAGES)
    return page


def render_executive_summary(simulator: ONASimulator, current_network: nx.Graph, metrics: Dict):
    """Render the executive summary page."""
    st.header("📊 Executive Summary")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Employees", len(simulator.employees))
    
    with col2:
        st.metric("Active Connections", current_network.number_of_edges())
    
    with col3:
        density = nx.density(current_network)
        st.metric("Network Density", f"{density:.3f}")
    
    with col4:
        if current_network.number_of_edges() > 0:
            components = nx.number_connected_components(current_network)
            st.metric("Network Fragmentation", f"{components} groups")
        else:
            st.metric("Network Fragmentation", "No connections")
    
    # Quick insights
    st.subheader("🔍 Key Insights")
    
    if current_network.number_of_edges() > 0:
        # Most connected employee
        most_connected = max(metrics['degree_centrality'].items(), key=lambda x: x[1])
        most_connected_emp = simulator.employees[most_connected[0]]
        
        # Most influential (betweenness)
        most_influential = max(metrics['betweenness_centrality'].items(), key=lambda x: x[1])
        most_influential_emp = simulator.employees[most_influential[0]]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Most Connected Employee:** {most_connected_emp.name} ({most_connected_emp.department})")
        
        with col2:
            st.info(f"**Most Influential Employee:** {most_influential_emp.name} ({most_influential_emp.department})")
    
    # Department connectivity overview
    if 'department_analysis' in metrics:
        st.subheader("🏬 Department Connectivity")
        dept_fig = create_department_analysis(metrics)
        if dept_fig:
            st.plotly_chart(dept_fig, use_container_width=True)


def render_network_visualization(simulator: ONASimulator, current_network: nx.Graph):
    """Render the network visualization page."""
    st.header("🌐 Interactive Network Visualization")
    
    # Visualization controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        color_by = st.selectbox("Color nodes by:", DashboardConfig.COLOR_SCHEMES)
    
    with col2:
        layout_type = st.selectbox("Layout algorithm:", DashboardConfig.LAYOUT_ALGORITHMS)
    
    with col3:
        if st.button("Generate Sample Network"):
            params = get_default_simulation_params()
            for _ in range(10):
                simulator.simulate_step(params)
            st.rerun()
    
    # Filtering controls
    st.subheader("🔍 Filters")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        departments = list(set(emp.department for emp in simulator.employees))
        filter_department = st.selectbox("Filter by department:", ["All"] + departments, index=0)
        if filter_department == "All":
            filter_department = None
    
    with col2:
        roles = list(set(emp.role for emp in simulator.employees))
        filter_role = st.selectbox("Filter by role:", ["All"] + roles, index=0)
        if filter_role == "All":
            filter_role = None
    
    with col3:
        engagement_filter = st.selectbox("Filter by engagement:", DashboardConfig.ENGAGEMENT_FILTERS, index=0)
        if engagement_filter == "All":
            engagement_filter = None
    
    with col4:
        search_term = st.text_input("Search employees:", placeholder="Enter name or keyword...")
        matching_employees = None
        if search_term:
            matching_employees = [
                i for i, emp in enumerate(simulator.employees)
                if search_term.lower() in emp.name.lower() or 
                   search_term.lower() in emp.department.lower() or
                   search_term.lower() in emp.role.lower()
            ]
            if matching_employees:
                st.info(f"Found {len(matching_employees)} matching employees")
            else:
                st.warning("No matching employees found")
    
    # Interactive network plot
    fig = create_ona_network_plot(
        current_network, 
        simulator.employees, 
        color_by=color_by,
        layout_type=layout_type,
        selected_nodes=matching_employees,
        filter_department=filter_department,
        filter_role=filter_role,
        engagement_filter=engagement_filter
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Visualization guide
    with st.expander("🎯 Visualization Guide"):
        st.markdown(DashboardConfig.get_visualization_help_text())


def render_centrality_analysis(simulator: ONASimulator, current_network: nx.Graph, metrics: Dict):
    """Render the centrality analysis page."""
    st.header("⭐ Centrality Analysis")
    
    if current_network.number_of_edges() > 0:
        fig_degree, fig_betweenness = create_centrality_analysis(metrics, simulator.employees)
        
        if fig_degree and fig_betweenness:
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(fig_degree, use_container_width=True)
            
            with col2:
                st.plotly_chart(fig_betweenness, use_container_width=True)
            
            # Detailed centrality table
            st.subheader("📈 Centrality Metrics")
            
            centrality_df = pd.DataFrame([
                {
                    'Employee': emp.name,
                    'Department': emp.department,
                    'Role': emp.role,
                    'Degree Centrality': metrics['degree_centrality'][i],
                    'Betweenness Centrality': metrics['betweenness_centrality'][i],
                    'Closeness Centrality': metrics['closeness_centrality'][i],
                    'Eigenvector Centrality': metrics['eigenvector_centrality'][i]
                }
                for i, emp in enumerate(simulator.employees)
            ])
            
            st.dataframe(centrality_df.round(4))
    else:
        st.info("Generate a network first to see centrality analysis.")


def render_department_analysis(metrics: Dict):
    """Render the department analysis page."""
    st.header("🏢 Department Analysis")
    
    if 'department_analysis' in metrics:
        # Department connectivity
        dept_fig = create_department_analysis(metrics)
        if dept_fig:
            st.plotly_chart(dept_fig, use_container_width=True)
        
        # Department details table
        st.subheader("📊 Department Details")
        
        dept_data = []
        for dept, data in metrics['department_analysis'].items():
            dept_data.append({
                'Department': dept,
                'Employees': data['nodes'],
                'Internal Connections': data['internal_edges'],
                'Internal Density': f"{data['density']:.3f}",
                'Avg Connections per Employee': f"{data['internal_edges'] / data['nodes']:.1f}" if data['nodes'] > 0 else "0"
            })
        
        dept_df = pd.DataFrame(dept_data)
        st.dataframe(dept_df)
    else:
        st.info("Generate a network first to see department analysis.")


def render_talent_risk(simulator: ONASimulator, current_network: nx.Graph, metrics: Dict):
    """Render the talent risk analysis page."""
    st.header("⚠️ Talent Risk Analysis")
    
    if current_network.number_of_edges() > 0:
        risk_fig, risk_df = create_talent_risk_analysis(current_network, simulator.employees, metrics)
        
        if risk_fig is not None:
            st.plotly_chart(risk_fig, use_container_width=True)
            
            # High risk employees
            st.subheader("🚨 High Risk Employees")
            high_risk = risk_df[risk_df['Risk Score'] > 0.4].sort_values('Risk Score', ascending=False)
            
            if not high_risk.empty:
                st.dataframe(high_risk[['Employee', 'Department', 'Role', 'Risk Score', 'Performance', 'Connections']])
            else:
                st.success("No high-risk employees identified.")
            
            # Key talent
            st.subheader("💎 Key Talent")
            key_talent = risk_df[risk_df['Key Talent'] == True].sort_values('Centrality', ascending=False)
            
            if not key_talent.empty:
                st.dataframe(key_talent[['Employee', 'Department', 'Role', 'Centrality', 'Performance', 'Connections']])
            else:
                st.info("Generate more connections to identify key talent.")
    else:
        st.info("Generate a network first to see talent risk analysis.")


def render_network_health(metrics: Dict):
    """Render the network health page."""
    st.header("🏥 Network Health")
    
    health_fig = create_network_health_dashboard(metrics)
    if health_fig:
        st.plotly_chart(health_fig, use_container_width=True)
    
    # Health recommendations
    st.subheader("💡 Recommendations")
    
    health = metrics.get('network_health', {})
    density = health.get('overall_density', 0)
    clustering = health.get('clustering_coefficient', 0)
    components = health.get('connected_components', 0)
    
    if density < DashboardConfig.LOW_DENSITY_THRESHOLD:
        st.warning("**Low Network Density:** Consider team-building activities or cross-departmental projects to increase connections.")
    
    if clustering < DashboardConfig.LOW_CLUSTERING_THRESHOLD:
        st.warning("**Low Clustering:** Encourage team formation and collaborative projects to build stronger working relationships.")
    
    if components > DashboardConfig.HIGH_FRAGMENTATION_THRESHOLD:
        st.warning("**Network Fragmentation:** Multiple disconnected groups detected. Consider initiatives to bridge these groups.")
    
    if (density > DashboardConfig.LOW_DENSITY_THRESHOLD and 
        clustering > DashboardConfig.LOW_CLUSTERING_THRESHOLD and 
        components < 3):
        st.success("**Healthy Network:** Good balance of connections and collaboration patterns.")


def render_employee_engagement(simulator: ONASimulator, metrics: Dict):
    """Render the employee engagement analysis page."""
    st.header("💡 Employee Engagement Analysis")
    
    engagement_metrics = metrics.get('engagement_metrics', {})
    
    if engagement_metrics:
        # Key engagement metrics
        st.subheader("📊 Key Engagement Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        overall_engagement = engagement_metrics.get('overall_engagement', {})
        risk_analysis = engagement_metrics.get('risk_analysis', {})
        
        with col1:
            mean_engagement = overall_engagement.get('mean', 0)
            st.metric("Average Engagement", f"{mean_engagement:.2f}")
        
        with col2:
            high_burnout = risk_analysis.get('high_burnout_risk', 0)
            st.metric("High Burnout Risk", high_burnout)
        
        with col3:
            low_retention = risk_analysis.get('low_retention_risk', 0)
            st.metric("Retention Risk", low_retention)
        
        with col4:
            high_influence = risk_analysis.get('high_influence_potential', 0)
            st.metric("High Influence Potential", high_influence)
        
        # Engagement visualizations
        fig_dist, fig_dept_box, fig_perf_eng, fig_risk, dept_df, risk_df = create_engagement_dashboard(simulator.employees, metrics)
        
        # Display charts
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_dist, use_container_width=True)
            st.plotly_chart(fig_perf_eng, use_container_width=True)
        
        with col2:
            st.plotly_chart(fig_dept_box, use_container_width=True)
            st.plotly_chart(fig_risk, use_container_width=True)
    else:
        st.info("Generate a network first to see engagement analysis.")


def render_influence_analysis(simulator: ONASimulator, current_network: nx.Graph, metrics: Dict):
    """Render the influence analysis page."""
    st.header("🎯 Organizational Influence Analysis")
    
    if current_network.number_of_edges() > 0:
        fig_influence, influence_df = create_influence_network_analysis(current_network, simulator.employees)
        
        if fig_influence is not None:
            st.plotly_chart(fig_influence, use_container_width=True)
            
            # Influence metrics explanation
            st.subheader("📈 Understanding Influence Scores")
            st.markdown("""
            **Influence Score Calculation:**
            - 40% Network Centrality (connections and position)
            - 30% Employee Engagement Level
            - 30% Calculated Influence Potential
            
            **Key Insights:**
            - High influence employees can drive organizational change
            - Consider these employees for leadership development
            - Monitor engagement levels of top influencers
            """)
            
            # Detailed influence table
            st.subheader("📅 Detailed Influence Analysis")
            st.dataframe(influence_df, use_container_width=True)
    else:
        st.info("Generate a network first to see influence analysis.")


def render_simulation(simulator: ONASimulator, current_network: nx.Graph):
    """Render the simulation page."""
    st.header("🔬 Interactive Network Simulation")
    
    # Simulation parameter controls
    st.sidebar.subheader("ONA Parameters")
    param_ranges = DashboardConfig.get_simulation_param_ranges()
    
    # Basic parameters
    edge_param = st.sidebar.slider("Baseline Connection Probability", **param_ranges['edges'])
    dept_param = st.sidebar.slider("Same Department Bonus", **param_ranges['same_department'])
    hierarchy_param = st.sidebar.slider("Hierarchy Effect", **param_ranges['hierarchy'])
    performance_param = st.sidebar.slider("Performance Similarity", **param_ranges['performance_similarity'])
    tenure_param = st.sidebar.slider("Tenure Similarity", **param_ranges['tenure_similarity'])
    
    # Engagement parameters
    st.sidebar.subheader("Engagement Parameters")
    engagement_param = st.sidebar.slider("Engagement Similarity", **param_ranges['engagement_similarity'])
    collaboration_param = st.sidebar.slider("Collaboration Boost", **param_ranges['collaboration_boost'])
    communication_param = st.sidebar.slider("Communication Affinity", **param_ranges['communication_affinity'])
    
    # Network structure parameters
    st.sidebar.subheader("Network Structure")
    triangle_param = st.sidebar.slider("Triangle Closure", **param_ranges['triangles'])
    preferential_param = st.sidebar.slider("Preferential Attachment", **param_ranges['preferential_attachment'])
    dissolution_param = st.sidebar.slider("Connection Dissolution", **param_ranges['dissolution'])
    
    # Simulation controls
    st.sidebar.subheader("Simulation Controls")
    time_steps = st.sidebar.slider("Time Steps", 1, DashboardConfig.MAX_SIMULATION_STEPS, DashboardConfig.DEFAULT_SIMULATION_STEPS)
    animation_speed = st.sidebar.slider("Animation Speed (seconds)", 
                                      DashboardConfig.MIN_ANIMATION_SPEED, 
                                      DashboardConfig.MAX_ANIMATION_SPEED, 
                                      DashboardConfig.DEFAULT_ANIMATION_SPEED, 0.1)
    
    params = {
        'edges': edge_param,
        'same_department': dept_param,
        'hierarchy': hierarchy_param,
        'performance_similarity': performance_param,
        'tenure_similarity': tenure_param,
        'engagement_similarity': engagement_param,
        'collaboration_boost': collaboration_param,
        'communication_affinity': communication_param,
        'triangles': triangle_param,
        'preferential_attachment': preferential_param,
        'dissolution': dissolution_param
    }
    
    # Control buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Reset Network"):
            simulator.reset_network()
            if 'simulation_history' in st.session_state:
                del st.session_state.simulation_history
            st.rerun()
    
    with col2:
        if st.button("⏭️ Single Step"):
            simulator.simulate_step(params)
            st.rerun()
    
    with col3:
        if st.button("▶️ Run Animated Simulation"):
            run_animated_simulation(simulator, params, time_steps, animation_speed)
    
    with col4:
        if st.button("📈 Quick Sample"):
            simulator.reset_network()
            sample_params = get_quick_sample_params()
            for _ in range(15):
                simulator.simulate_step(sample_params)
            st.rerun()
    
    # Current network visualization
    if simulator.networks:
        current_step = len(simulator.networks)
        st.subheader(f"🌐 Current Network State (Step {current_step})")
    else:
        st.subheader("🌐 Initial Network State")
    
    fig = create_ona_network_plot(current_network, simulator.employees, 
                                color_by="department", layout_type="spring")
    fig.update_layout(height=600, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)
    
    # Help section
    with st.expander("ℹ️ Simulation Help"):
        st.markdown(DashboardConfig.get_simulation_help_text())


def run_animated_simulation(simulator: ONASimulator, params: Dict, time_steps: int, animation_speed: float):
    """Run an animated simulation with real-time updates."""
    simulator.reset_network()
    
    st.session_state.simulation_history = []
    st.write("---")
    st.markdown("### 🎬 Running Simulation Animation")
    
    progress_placeholder = st.empty()
    metrics_placeholder = st.empty()
    network_placeholder = st.empty()
    
    for step in range(time_steps):
        simulator.simulate_step(params)
        current_net = simulator.networks[-1]
        
        step_metrics = {
            'step': step + 1,
            'nodes': current_net.number_of_nodes(),
            'edges': current_net.number_of_edges(),
            'density': nx.density(current_net),
            'clustering': nx.average_clustering(current_net),
            'components': nx.number_connected_components(current_net)
        }
        st.session_state.simulation_history.append(step_metrics)
        
        progress_placeholder.progress((step + 1) / time_steps, 
                                    f"Step {step + 1}/{time_steps}: {current_net.number_of_edges()} connections")
        
        with metrics_placeholder.container():
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Connections", step_metrics['edges'])
            with col2:
                st.metric("Density", f"{step_metrics['density']:.3f}")
            with col3:
                st.metric("Clustering", f"{step_metrics['clustering']:.3f}")
            with col4:
                st.metric("Components", step_metrics['components'])
        
        with network_placeholder:
            st.write(f"**Network at Step {step + 1}**")
            fig = create_ona_network_plot(current_net, simulator.employees, 
                                        color_by="department", layout_type="spring")
            st.plotly_chart(fig, use_container_width=True)
        
        time.sleep(animation_speed)
    
    progress_placeholder.empty()
    st.success(f"✅ Simulation completed! Final network has {current_net.number_of_edges()} connections.")
    st.write("---")


def render_export_controls(simulator: ONASimulator, current_network: nx.Graph):
    """Render export functionality in sidebar."""
    st.sidebar.subheader("Export Data")
    if st.sidebar.button("Export Network Data"):
        export_data = simulator.export_data()
        
        st.sidebar.download_button(
            label="Download JSON",
            data=json.dumps(export_data, indent=2),
            file_name=f"ona_data_{datetime.now().strftime(DashboardConfig.EXPORT_DATETIME_FORMAT)}.json",
            mime="application/json"
        )


def main():
    """Main application function."""
    # Initialize the app
    initialize_app()
    
    # Get simulator and current network
    simulator = st.session_state.ona_simulator
    current_network = simulator.get_current_network()
    
    # Calculate metrics
    metrics = calculate_ona_metrics(current_network, simulator.employees)
    
    # Render title and description
    st.title("🏢 Organizational Network Analysis Dashboard")
    st.markdown("Advanced ONA insights for HR professionals")
    
    # Render sidebar and get selected page
    page = render_sidebar()
    
    # Route to appropriate page renderer
    if page == "Executive Summary":
        render_executive_summary(simulator, current_network, metrics)
    elif page == "Network Visualization":
        render_network_visualization(simulator, current_network)
    elif page == "Centrality Analysis":
        render_centrality_analysis(simulator, current_network, metrics)
    elif page == "Department Analysis":
        render_department_analysis(metrics)
    elif page == "Talent Risk":
        render_talent_risk(simulator, current_network, metrics)
    elif page == "Network Health":
        render_network_health(metrics)
    elif page == "Employee Engagement":
        render_employee_engagement(simulator, metrics)
    elif page == "Influence Analysis":
        render_influence_analysis(simulator, current_network, metrics)
    elif page == "Simulation":
        render_simulation(simulator, current_network)
    
    # Export controls
    render_export_controls(simulator, current_network)


if __name__ == "__main__":
    main()