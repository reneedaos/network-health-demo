"""
Analysis visualization components for ONA dashboard.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Optional
from ..models.employee import Employee
from ..utils.metrics import get_top_employees_by_metric


def create_centrality_analysis(metrics: Dict, employees: List[Employee]) -> Tuple[Optional[go.Figure], Optional[go.Figure]]:
    """Create centrality analysis visualizations."""
    if not metrics:
        return None, None
    
    # Get top employees by centrality metrics
    top_degree = get_top_employees_by_metric(metrics, employees, 'degree_centrality', 10)
    top_betweenness = get_top_employees_by_metric(metrics, employees, 'betweenness_centrality', 10)
    
    if not top_degree or not top_betweenness:
        return None, None
    
    # Create dataframes
    degree_df = pd.DataFrame(top_degree)
    betweenness_df = pd.DataFrame(top_betweenness)
    
    # Create plots
    fig_degree = px.bar(degree_df, x='Score', y='Employee', 
                       color='Department', title='Top 10 Most Connected Employees',
                       orientation='h', height=400)
    
    fig_betweenness = px.bar(betweenness_df, x='Score', y='Employee',
                            color='Department', title='Top 10 Bridge Builders',
                            orientation='h', height=400)
    
    return fig_degree, fig_betweenness


def create_department_analysis(metrics: Dict) -> Optional[go.Figure]:
    """Create department-level analysis."""
    if 'department_analysis' not in metrics:
        return None
    
    dept_data = []
    for dept, data in metrics['department_analysis'].items():
        dept_data.append({
            'Department': dept,
            'Employees': data['nodes'],
            'Internal Connections': data['internal_edges'],
            'Internal Density': data['density']
        })
    
    df = pd.DataFrame(dept_data)
    
    # Create subplot figure
    fig = go.Figure()
    
    # Add bars for internal connections
    fig.add_trace(go.Bar(
        name='Internal Connections',
        x=df['Department'],
        y=df['Internal Connections'],
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title='Department Internal Connectivity',
        xaxis_title='Department',
        yaxis_title='Number of Internal Connections',
        height=400
    )
    
    return fig


def create_network_health_dashboard(metrics: Dict) -> Optional[go.Figure]:
    """Create network health indicators."""
    if 'network_health' not in metrics:
        return None
    
    health = metrics['network_health']
    
    # Create gauge charts
    fig = go.Figure()
    
    # Overall density gauge
    fig.add_trace(go.Indicator(
        mode = "gauge+number+delta",
        value = health['overall_density'],
        domain = {'x': [0, 0.5], 'y': [0.5, 1]},
        title = {'text': "Network Density"},
        gauge = {
            'axis': {'range': [None, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.3], 'color': "lightgray"},
                {'range': [0.3, 0.7], 'color': "gray"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.9}}))
    
    # Clustering coefficient gauge
    fig.add_trace(go.Indicator(
        mode = "gauge+number+delta",
        value = health['clustering_coefficient'],
        domain = {'x': [0.5, 1], 'y': [0.5, 1]},
        title = {'text': "Clustering Coefficient"},
        gauge = {
            'axis': {'range': [None, 1]},
            'bar': {'color': "darkgreen"},
            'steps': [
                {'range': [0, 0.3], 'color': "lightgray"},
                {'range': [0.3, 0.7], 'color': "gray"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.9}}))
    
    fig.update_layout(
        title="Network Health Indicators",
        height=400
    )
    
    return fig


def create_talent_risk_analysis(G, employees: List[Employee], metrics: Dict) -> Tuple[Optional[go.Figure], Optional[pd.DataFrame]]:
    """Identify employees at risk of leaving or key talent."""
    if not metrics:
        return None, None
    
    risk_data = []
    for i, emp in enumerate(employees):
        connections = G.degree(i)
        centrality = metrics['degree_centrality'][i]
        
        # Risk factors (simplified)
        risk_score = 0
        if emp.tenure < 2:
            risk_score += 0.3  # New employees
        if emp.performance < 0.7:
            risk_score += 0.2  # Low performance
        if connections < 3:
            risk_score += 0.2  # Isolated employees
        
        # Key talent indicators
        key_talent = centrality > 0.1 and emp.performance > 0.8
        
        risk_data.append({
            'Employee': emp.name,
            'Department': emp.department,
            'Role': emp.role,
            'Tenure': emp.tenure,
            'Performance': emp.performance,
            'Connections': connections,
            'Centrality': centrality,
            'Risk Score': risk_score,
            'Key Talent': key_talent
        })
    
    df = pd.DataFrame(risk_data)
    
    # Create scatter plot
    fig = px.scatter(df, x='Performance', y='Risk Score', 
                    color='Department', size='Connections',
                    hover_data=['Employee', 'Role', 'Tenure'],
                    title='Talent Risk Analysis')
    
    return fig, df


def create_engagement_dashboard(employees: List[Employee], metrics: Dict) -> Tuple[go.Figure, go.Figure, go.Figure, go.Figure, pd.DataFrame, pd.DataFrame]:
    """Create comprehensive engagement analysis dashboard."""
    # Extract engagement scores
    engagement_scores = [emp.engagement_score for emp in employees]
    departments = [emp.department for emp in employees]
    performance_scores = [emp.performance for emp in employees]
    
    # 1. Engagement distribution
    fig_dist = px.histogram(
        x=engagement_scores,
        nbins=20,
        title='Employee Engagement Distribution',
        labels={'x': 'Engagement Score', 'y': 'Number of Employees'}
    )
    
    # 2. Department engagement box plot
    dept_engagement_data = []
    for emp in employees:
        dept_engagement_data.append({
            'Department': emp.department,
            'Engagement': emp.engagement_score,
            'Employee': emp.name
        })
    
    dept_df = pd.DataFrame(dept_engagement_data)
    fig_dept_box = px.box(
        dept_df,
        x='Department',
        y='Engagement',
        title='Engagement by Department'
    )
    
    # 3. Performance vs Engagement scatter
    fig_perf_eng = px.scatter(
        x=performance_scores,
        y=engagement_scores,
        title='Performance vs Engagement',
        labels={'x': 'Performance Score', 'y': 'Engagement Score'}
    )
    
    # 4. Risk analysis
    risk_data = []
    for emp in employees:
        risk_data.append({
            'Employee': emp.name,
            'Department': emp.department,
            'Burnout Risk': emp.burnout_risk,
            'Retention Risk': 1 - emp.retention_likelihood,
            'Influence Potential': emp.influence_potential
        })
    
    risk_df = pd.DataFrame(risk_data)
    fig_risk = px.scatter(
        risk_df,
        x='Burnout Risk',
        y='Retention Risk',
        color='Department',
        size='Influence Potential',
        hover_data=['Employee'],
        title='Employee Risk Analysis'
    )
    
    return fig_dist, fig_dept_box, fig_perf_eng, fig_risk, dept_df, risk_df


def create_influence_network_analysis(G, employees: List[Employee]) -> Tuple[Optional[go.Figure], Optional[pd.DataFrame]]:
    """Create influence analysis combining network position and engagement."""
    if G.number_of_edges() == 0:
        return None, None
    
    # Calculate influence scores
    influence_data = []
    degrees = dict(G.degree())
    
    for i, emp in enumerate(employees):
        # Normalize degree centrality
        degree_centrality = degrees[i] / max(1, max(degrees.values()))
        
        # Combined influence score
        influence_score = (
            degree_centrality * 0.4 +
            emp.engagement_score * 0.3 +
            emp.influence_potential * 0.3
        )
        
        influence_data.append({
            'Employee': emp.name,
            'Department': emp.department,
            'Role': emp.role,
            'Network Centrality': degree_centrality,
            'Engagement Score': emp.engagement_score,
            'Influence Potential': emp.influence_potential,
            'Combined Influence Score': influence_score,
            'Connections': degrees[i]
        })
    
    df = pd.DataFrame(influence_data)
    
    # Create scatter plot
    fig = px.scatter(
        df,
        x='Network Centrality',
        y='Engagement Score',
        color='Combined Influence Score',
        size='Connections',
        hover_data=['Employee', 'Role', 'Department'],
        title='Network Position vs Engagement (Influence Analysis)',
        labels={
            'Network Centrality': 'Network Centrality (Normalized)',
            'Engagement Score': 'Employee Engagement Score'
        }
    )
    
    return fig, df