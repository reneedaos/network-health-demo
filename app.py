import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import bernoulli
import time
from typing import Dict, List, Tuple
import random
from datetime import datetime, timedelta
import json

st.set_page_config(page_title="Organizational Network Analysis Dashboard", layout="wide")

class Employee:
    def __init__(self, id: int, name: str, department: str, role: str, tenure: int, performance: float, 
                 engagement_score: float = None, communication_frequency: float = None, 
                 collaboration_index: float = None, satisfaction_score: float = None):
        self.id = id
        self.name = name
        self.department = department
        self.role = role
        self.tenure = tenure
        self.performance = performance
        
        # Employee Engagement Metrics
        self.engagement_score = engagement_score or random.uniform(0.3, 1.0)  # Overall engagement (0-1)
        self.communication_frequency = communication_frequency or random.uniform(0.2, 1.0)  # Communication activity (0-1)
        self.collaboration_index = collaboration_index or random.uniform(0.1, 1.0)  # Team collaboration (0-1)
        self.satisfaction_score = satisfaction_score or random.uniform(0.4, 1.0)  # Job satisfaction (0-1)
        
        # Derived engagement metrics
        self.burnout_risk = self._calculate_burnout_risk()
        self.influence_potential = self._calculate_influence_potential()
        self.retention_likelihood = self._calculate_retention_likelihood()
    
    def _calculate_burnout_risk(self) -> float:
        """Calculate burnout risk based on engagement factors"""
        # Higher performance with lower satisfaction/engagement indicates potential burnout
        stress_factor = max(0, self.performance - self.satisfaction_score)
        low_engagement_factor = max(0, 0.7 - self.engagement_score)
        return min(1.0, stress_factor + low_engagement_factor)
    
    def _calculate_influence_potential(self) -> float:
        """Calculate potential for organizational influence"""
        return (self.engagement_score * 0.4 + 
                self.communication_frequency * 0.3 + 
                self.collaboration_index * 0.2 + 
                self.performance * 0.1)
    
    def _calculate_retention_likelihood(self) -> float:
        """Calculate likelihood of employee retention"""
        return (self.satisfaction_score * 0.35 + 
                self.engagement_score * 0.25 + 
                (self.tenure / 10) * 0.15 +  # Tenure factor (normalized)
                self.collaboration_index * 0.15 + 
                self.performance * 0.1)

class ONASimulator:
    def __init__(self, n_employees: int = 50):
        self.n_employees = n_employees
        self.generate_employees()
        self.reset_network()
    
    def generate_employees(self):
        """Generate realistic employee data"""
        departments = ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance', 'Operations']
        roles = ['Manager', 'Senior', 'Mid-level', 'Junior', 'Executive']
        
        self.employees = []
        for i in range(self.n_employees):
            name = f"Employee_{i+1}"
            dept = random.choice(departments)
            role = random.choice(roles)
            tenure = random.randint(1, 10)
            performance = random.uniform(0.6, 1.0)
            
            # Generate engagement metrics with realistic correlations
            engagement_score = self._generate_correlated_engagement(performance, tenure)
            communication_freq = self._generate_communication_frequency(role, dept)
            collaboration_idx = self._generate_collaboration_index(role, performance)
            satisfaction = self._generate_satisfaction(performance, tenure, engagement_score)
            
            self.employees.append(Employee(i, name, dept, role, tenure, performance,
                                         engagement_score, communication_freq, 
                                         collaboration_idx, satisfaction))
    
    def _generate_correlated_engagement(self, performance: float, tenure: int) -> float:
        """Generate engagement score with realistic correlations"""
        # Higher performance generally correlates with higher engagement
        # But very long tenure might reduce engagement (stagnation)
        base_engagement = performance * 0.7 + random.uniform(0.1, 0.3)
        tenure_factor = 1.0 if tenure < 7 else max(0.7, 1.0 - (tenure - 7) * 0.05)
        return min(1.0, base_engagement * tenure_factor)
    
    def _generate_communication_frequency(self, role: str, department: str) -> float:
        """Generate communication frequency based on role and department"""
        base_freq = random.uniform(0.3, 0.8)
        if 'Manager' in role or 'Executive' in role:
            base_freq += 0.2
        if department in ['Sales', 'Marketing', 'HR']:
            base_freq += 0.1
        return min(1.0, base_freq)
    
    def _generate_collaboration_index(self, role: str, performance: float) -> float:
        """Generate collaboration index based on role and performance"""
        base_collab = random.uniform(0.2, 0.7)
        if 'Senior' in role:
            base_collab += 0.15
        # High performers often collaborate more
        if performance > 0.8:
            base_collab += 0.1
        return min(1.0, base_collab)
    
    def _generate_satisfaction(self, performance: float, tenure: int, engagement: float) -> float:
        """Generate satisfaction score with realistic correlations"""
        # Satisfaction correlates with performance and engagement
        # But decreases with very long tenure (career stagnation)
        base_satisfaction = (performance * 0.4 + engagement * 0.4 + random.uniform(0.1, 0.2))
        if tenure > 8:
            base_satisfaction *= 0.9  # Slight decrease for long tenure
        return min(1.0, base_satisfaction)
    
    def reset_network(self):
        self.G = nx.Graph()
        self.G.add_nodes_from(range(self.n_employees))
        
        # Add node attributes
        for i, emp in enumerate(self.employees):
            self.G.nodes[i]['name'] = emp.name
            self.G.nodes[i]['department'] = emp.department
            self.G.nodes[i]['role'] = emp.role
            self.G.nodes[i]['tenure'] = emp.tenure
            self.G.nodes[i]['performance'] = emp.performance
            self.G.nodes[i]['engagement_score'] = emp.engagement_score
            self.G.nodes[i]['communication_frequency'] = emp.communication_frequency
            self.G.nodes[i]['collaboration_index'] = emp.collaboration_index
            self.G.nodes[i]['satisfaction_score'] = emp.satisfaction_score
            self.G.nodes[i]['burnout_risk'] = emp.burnout_risk
            self.G.nodes[i]['influence_potential'] = emp.influence_potential
            self.G.nodes[i]['retention_likelihood'] = emp.retention_likelihood
        
        self.time_steps = []
        self.networks = []
        
    def ergm_probability(self, i: int, j: int, params: Dict) -> float:
        """Calculate ERGM probability for edge (i,j) with organizational context"""
        if i == j:
            return 0.0
            
        logit = params['edges']
        
        # Same department bonus
        if self.employees[i].department == self.employees[j].department:
            logit += params['same_department']
        
        # Hierarchy effect (managers connect more)
        if 'Manager' in self.employees[i].role or 'Manager' in self.employees[j].role:
            logit += params['hierarchy']
        
        # Performance similarity
        perf_diff = abs(self.employees[i].performance - self.employees[j].performance)
        logit += params['performance_similarity'] * (1 - perf_diff)
        
        # Tenure similarity
        tenure_diff = abs(self.employees[i].tenure - self.employees[j].tenure)
        logit += params['tenure_similarity'] * np.exp(-tenure_diff / 3)
        
        # Engagement-based connection factors
        if 'engagement_similarity' in params:
            engagement_diff = abs(self.employees[i].engagement_score - self.employees[j].engagement_score)
            logit += params['engagement_similarity'] * (1 - engagement_diff)
        
        if 'collaboration_boost' in params:
            collab_factor = (self.employees[i].collaboration_index + self.employees[j].collaboration_index) / 2
            logit += params['collaboration_boost'] * collab_factor
        
        if 'communication_affinity' in params:
            comm_factor = (self.employees[i].communication_frequency + self.employees[j].communication_frequency) / 2
            logit += params['communication_affinity'] * comm_factor
        
        # Triangle closure
        common_neighbors = len(list(nx.common_neighbors(self.G, i, j)))
        logit += params['triangles'] * common_neighbors
        
        # Degree effects
        degree_i = self.G.degree(i)
        degree_j = self.G.degree(j)
        logit += params['preferential_attachment'] * np.log(1 + degree_i + degree_j)
        
        return 1 / (1 + np.exp(-logit))
    
    def simulate_step(self, params: Dict):
        """Single simulation step"""
        edges_to_consider = []
        
        for i in range(self.n_employees):
            for j in range(i + 1, self.n_employees):
                if not self.G.has_edge(i, j):
                    prob = self.ergm_probability(i, j, params)
                    if np.random.random() < prob:
                        edges_to_consider.append((i, j))
        
        for edge in edges_to_consider:
            self.G.add_edge(*edge)
        
        # Edge dissolution
        edges_to_remove = []
        for edge in self.G.edges():
            if np.random.random() < params['dissolution']:
                edges_to_remove.append(edge)
        
        for edge in edges_to_remove:
            self.G.remove_edge(*edge)
        
        self.networks.append(self.G.copy())
        self.time_steps.append(len(self.time_steps))

def calculate_engagement_metrics(employees: List[Employee], G: nx.Graph = None) -> Dict:
    """Calculate comprehensive employee engagement metrics"""
    engagement_metrics = {}
    
    # Overall engagement statistics
    engagement_scores = [emp.engagement_score for emp in employees]
    engagement_metrics['overall_engagement'] = {
        'mean': np.mean(engagement_scores),
        'std': np.std(engagement_scores),
        'min': np.min(engagement_scores),
        'max': np.max(engagement_scores),
        'quartiles': np.percentile(engagement_scores, [25, 50, 75])
    }
    
    # Engagement by department
    dept_engagement = {}
    departments = set(emp.department for emp in employees)
    for dept in departments:
        dept_employees = [emp for emp in employees if emp.department == dept]
        dept_scores = [emp.engagement_score for emp in dept_employees]
        dept_engagement[dept] = {
            'mean_engagement': np.mean(dept_scores),
            'employees_count': len(dept_employees),
            'high_engagement_count': len([s for s in dept_scores if s > 0.8]),
            'low_engagement_count': len([s for s in dept_scores if s < 0.5])
        }
    
    engagement_metrics['department_engagement'] = dept_engagement
    
    # Risk analysis
    high_burnout_risk = [emp for emp in employees if emp.burnout_risk > 0.7]
    low_retention_risk = [emp for emp in employees if emp.retention_likelihood < 0.5]
    high_influence_potential = [emp for emp in employees if emp.influence_potential > 0.8]
    
    engagement_metrics['risk_analysis'] = {
        'high_burnout_risk': len(high_burnout_risk),
        'low_retention_risk': len(low_retention_risk),
        'high_influence_potential': len(high_influence_potential),
        'burnout_risk_employees': [(emp.name, emp.department, emp.burnout_risk) for emp in high_burnout_risk],
        'retention_risk_employees': [(emp.name, emp.department, emp.retention_likelihood) for emp in low_retention_risk],
        'high_influence_employees': [(emp.name, emp.department, emp.influence_potential) for emp in high_influence_potential]
    }
    
    # Engagement-Network correlation (if network provided)
    if G is not None and G.number_of_edges() > 0:
        engagement_network_corr = {}
        degrees = dict(G.degree())
        
        # Calculate correlations between engagement metrics and network position
        degree_values = [degrees.get(i, 0) for i in range(len(employees))]
        engagement_values = [emp.engagement_score for emp in employees]
        collaboration_values = [emp.collaboration_index for emp in employees]
        
        if len(set(degree_values)) > 1:  # Avoid correlation with constant values
            engagement_network_corr['engagement_degree_correlation'] = np.corrcoef(engagement_values, degree_values)[0, 1]
            engagement_network_corr['collaboration_degree_correlation'] = np.corrcoef(collaboration_values, degree_values)[0, 1]
        else:
            engagement_network_corr['engagement_degree_correlation'] = 0
            engagement_network_corr['collaboration_degree_correlation'] = 0
        
        engagement_metrics['network_correlation'] = engagement_network_corr
    
    return engagement_metrics

def calculate_ona_metrics(G: nx.Graph, employees: List[Employee]) -> Dict:
    """Calculate key ONA metrics including engagement analysis"""
    metrics = {}
    
    # Centrality measures
    if G.number_of_edges() > 0:
        metrics['degree_centrality'] = nx.degree_centrality(G)
        metrics['betweenness_centrality'] = nx.betweenness_centrality(G)
        metrics['closeness_centrality'] = nx.closeness_centrality(G)
        metrics['eigenvector_centrality'] = nx.eigenvector_centrality(G, max_iter=1000)
    else:
        metrics['degree_centrality'] = {i: 0 for i in G.nodes()}
        metrics['betweenness_centrality'] = {i: 0 for i in G.nodes()}
        metrics['closeness_centrality'] = {i: 0 for i in G.nodes()}
        metrics['eigenvector_centrality'] = {i: 0 for i in G.nodes()}
    
    # Department connectivity
    departments = set(emp.department for emp in employees)
    dept_connections = {}
    for dept in departments:
        dept_nodes = [i for i, emp in enumerate(employees) if emp.department == dept]
        dept_subgraph = G.subgraph(dept_nodes)
        dept_connections[dept] = {
            'internal_edges': dept_subgraph.number_of_edges(),
            'nodes': len(dept_nodes),
            'density': nx.density(dept_subgraph) if len(dept_nodes) > 1 else 0
        }
    
    metrics['department_analysis'] = dept_connections
    
    # Network health indicators
    metrics['network_health'] = {
        'overall_density': nx.density(G),
        'clustering_coefficient': nx.average_clustering(G),
        'connected_components': nx.number_connected_components(G),
        'largest_component_size': len(max(nx.connected_components(G), key=len)) if G.number_of_edges() > 0 else 0
    }
    
    # Add engagement metrics
    metrics['engagement_metrics'] = calculate_engagement_metrics(employees, G)
    
    return metrics

def create_ona_network_plot(G: nx.Graph, employees: List[Employee], color_by: str = 'department', 
                           layout_type: str = 'spring', selected_nodes: List[int] = None,
                           filter_department: str = None, filter_role: str = None, 
                           engagement_filter: str = None):
    """Create interactive ONA network visualization"""
    if G.number_of_edges() == 0:
        # Handle empty network
        fig = go.Figure()
        fig.add_annotation(
            text="No connections in the network yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(
            title="Organizational Network",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
        return fig
    
    # Apply filters
    filtered_nodes = list(G.nodes())
    if filter_department:
        filtered_nodes = [n for n in filtered_nodes if employees[n].department == filter_department]
    if filter_role:
        filtered_nodes = [n for n in filtered_nodes if filter_role.lower() in employees[n].role.lower()]
    if engagement_filter:
        if engagement_filter == 'High Engagement':
            filtered_nodes = [n for n in filtered_nodes if employees[n].engagement_score > 0.8]
        elif engagement_filter == 'Low Engagement':
            filtered_nodes = [n for n in filtered_nodes if employees[n].engagement_score < 0.5]
        elif engagement_filter == 'Burnout Risk':
            filtered_nodes = [n for n in filtered_nodes if employees[n].burnout_risk > 0.7]
        elif engagement_filter == 'High Influence':
            filtered_nodes = [n for n in filtered_nodes if employees[n].influence_potential > 0.8]
    
    # Create subgraph with filtered nodes
    G_filtered = G.subgraph(filtered_nodes)
    
    if G_filtered.number_of_nodes() == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No nodes match the current filters",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(
            title="Organizational Network (Filtered)",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
        return fig
    
    # Dynamic layout selection
    if layout_type == 'spring':
        pos = nx.spring_layout(G_filtered, k=2, iterations=50)
    elif layout_type == 'circular':
        pos = nx.circular_layout(G_filtered)
    elif layout_type == 'random':
        pos = nx.random_layout(G_filtered)
    elif layout_type == 'shell':
        pos = nx.shell_layout(G_filtered)
    elif layout_type == 'hierarchical':
        try:
            pos = nx.nx_agraph.graphviz_layout(G_filtered, prog='dot')
        except:
            pos = nx.spring_layout(G_filtered, k=2, iterations=50)
    else:
        pos = nx.spring_layout(G_filtered, k=2, iterations=50)
    
    # Extract edges with weights
    edge_x = []
    edge_y = []
    edge_weights = []
    edge_info = []
    
    for edge in G_filtered.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        # Calculate edge weight based on employee similarity
        emp1, emp2 = employees[edge[0]], employees[edge[1]]
        weight = 1.0
        if emp1.department == emp2.department:
            weight += 0.5
        if abs(emp1.performance - emp2.performance) < 0.2:
            weight += 0.3
        if abs(emp1.tenure - emp2.tenure) < 2:
            weight += 0.2
        
        edge_weights.append(weight)
        edge_info.append(f"{emp1.name} ↔ {emp2.name}<br>Connection Strength: {weight:.2f}")
    
    # Extract nodes
    node_x = []
    node_y = []
    node_info = []
    node_colors = []
    node_sizes = []
    node_symbols = []
    
    departments = list(set(emp.department for emp in employees))
    dept_color_map = {dept: i for i, dept in enumerate(departments)}
    
    for node in G_filtered.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        emp = employees[node]
        degree = G_filtered.degree(node)
        
        # Enhanced node information with engagement metrics
        node_info.append(
            f"<b>{emp.name}</b><br>"
            f"Department: {emp.department}<br>"
            f"Role: {emp.role}<br>"
            f"Tenure: {emp.tenure} years<br>"
            f"Performance: {emp.performance:.2f}<br>"
            f"Engagement: {emp.engagement_score:.2f}<br>"
            f"Satisfaction: {emp.satisfaction_score:.2f}<br>"
            f"Burnout Risk: {emp.burnout_risk:.2f}<br>"
            f"Retention: {emp.retention_likelihood:.2f}<br>"
            f"Connections: {degree}<br>"
            f"<i>Click to select/deselect</i>"
        )
        
        # Color coding with engagement options
        if color_by == 'department':
            node_colors.append(dept_color_map[emp.department])
        elif color_by == 'performance':
            node_colors.append(emp.performance)
        elif color_by == 'tenure':
            node_colors.append(emp.tenure)
        elif color_by == 'engagement':
            node_colors.append(emp.engagement_score)
        elif color_by == 'satisfaction':
            node_colors.append(emp.satisfaction_score)
        elif color_by == 'burnout_risk':
            node_colors.append(emp.burnout_risk)
        elif color_by == 'influence_potential':
            node_colors.append(emp.influence_potential)
        elif color_by == 'retention_likelihood':
            node_colors.append(emp.retention_likelihood)
        else:
            node_colors.append(degree)
        
        # Size based on degree centrality and engagement
        base_size = max(10, min(25, 8 + degree * 2))
        # Boost size for high engagement employees
        if emp.engagement_score > 0.8:
            base_size += 3
        # Reduce size for burnout risk employees
        if emp.burnout_risk > 0.7:
            base_size = max(8, base_size - 2)
        node_sizes.append(base_size)
        
        # Symbol based on role and engagement status
        if 'Manager' in emp.role or 'Executive' in emp.role:
            node_symbols.append('diamond')
        elif 'Senior' in emp.role:
            node_symbols.append('square')
        elif emp.burnout_risk > 0.7:
            node_symbols.append('triangle-up')  # Warning symbol for burnout risk
        elif emp.influence_potential > 0.8:
            node_symbols.append('star')  # Star for high influence
        else:
            node_symbols.append('circle')
    
    # Create plot
    fig = go.Figure()
    
    # Add edges with varying opacity based on weight
    for i in range(0, len(edge_x), 3):  # Every third point is None (separator)
        if i + 1 < len(edge_x) and edge_x[i] is not None and edge_x[i+1] is not None:
            edge_idx = i // 3
            if edge_idx < len(edge_weights):
                weight = edge_weights[edge_idx]
                opacity = min(0.8, 0.2 + weight * 0.3)
                width = min(3, 0.5 + weight * 0.5)
                
                fig.add_trace(go.Scatter(
                    x=[edge_x[i], edge_x[i+1]],
                    y=[edge_y[i], edge_y[i+1]],
                    mode='lines',
                    line=dict(width=width, color=f'rgba(125,125,125,{opacity})'),
                    hoverinfo='text',
                    hovertext=edge_info[edge_idx] if edge_idx < len(edge_info) else "",
                    showlegend=False
                ))
    
    # Add nodes with enhanced interactivity
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_info,
        marker=dict(
            size=node_sizes,
            color=node_colors,
            colorscale='Viridis' if color_by != 'department' else None,
            colorbar=dict(title=color_by.title()) if color_by != 'department' else None,
            line=dict(width=2, color='white'),
            symbol=node_symbols,
            opacity=0.8
        ),
        showlegend=False,
        customdata=list(G_filtered.nodes()),  # Store node IDs for selection
    ))
    
    # Highlight selected nodes
    if selected_nodes:
        selected_x = [pos[n][0] for n in selected_nodes if n in G_filtered.nodes()]
        selected_y = [pos[n][1] for n in selected_nodes if n in G_filtered.nodes()]
        
        if selected_x and selected_y:
            fig.add_trace(go.Scatter(
                x=selected_x, y=selected_y,
                mode='markers',
                marker=dict(
                    size=25,
                    color='rgba(255,0,0,0)',
                    line=dict(width=4, color='red')
                ),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Enhanced layout with interactive controls
    fig.update_layout(
        title=f"Organizational Network ({layout_type.title()} Layout)",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,
        # Ensure responsive width
        autosize=True,
        # Enable dragging and zooming
        dragmode='pan',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        # Add zoom controls
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        args=[{"dragmode": "zoom"}],
                        label="Zoom",
                        method="relayout"
                    ),
                    dict(
                        args=[{"dragmode": "pan"}],
                        label="Pan",
                        method="relayout"
                    ),
                    dict(
                        args=[{"xaxis.autorange": True, "yaxis.autorange": True}],
                        label="Reset View",
                        method="relayout"
                    )
                ]),
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.01,
                xanchor="left",
                y=1.02,
                yanchor="top"
            ),
        ]
    )
    
    return fig

def create_centrality_analysis(metrics: Dict, employees: List[Employee]):
    """Create centrality analysis visualizations"""
    if not metrics:
        return None, None
    
    # Top influencers
    top_degree = sorted(metrics['degree_centrality'].items(), key=lambda x: x[1], reverse=True)[:10]
    top_betweenness = sorted(metrics['betweenness_centrality'].items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Create dataframes
    degree_df = pd.DataFrame([
        {
            'Employee': employees[emp_id].name,
            'Department': employees[emp_id].department,
            'Role': employees[emp_id].role,
            'Degree Centrality': score
        }
        for emp_id, score in top_degree
    ])
    
    betweenness_df = pd.DataFrame([
        {
            'Employee': employees[emp_id].name,
            'Department': employees[emp_id].department,
            'Role': employees[emp_id].role,
            'Betweenness Centrality': score
        }
        for emp_id, score in top_betweenness
    ])
    
    # Create plots
    fig_degree = px.bar(degree_df, x='Degree Centrality', y='Employee', 
                       color='Department', title='Top 10 Most Connected Employees',
                       orientation='h', height=400)
    
    fig_betweenness = px.bar(betweenness_df, x='Betweenness Centrality', y='Employee',
                            color='Department', title='Top 10 Bridge Builders',
                            orientation='h', height=400)
    
    return fig_degree, fig_betweenness

def create_department_analysis(metrics: Dict):
    """Create department-level analysis"""
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

def create_network_health_dashboard(metrics: Dict):
    """Create network health indicators"""
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

def create_talent_risk_analysis(G: nx.Graph, employees: List[Employee], metrics: Dict):
    """Identify employees at risk of leaving or key talent"""
    if not metrics:
        return None
    
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

def main():
    st.title("🏢 Organizational Network Analysis Dashboard")
    st.markdown("Advanced ONA insights for HR professionals")
    
    # Sidebar navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox("Select View", 
                               ["Executive Summary", "Network Visualization", "Centrality Analysis", 
                                "Department Analysis", "Talent Risk", "Network Health", "Simulation"])
    
    # Initialize simulator
    if 'ona_simulator' not in st.session_state:
        st.session_state.ona_simulator = ONASimulator(50)
    
    simulator = st.session_state.ona_simulator
    
    # Get current network
    if simulator.networks:
        current_network = simulator.networks[-1]
    else:
        current_network = simulator.G
    
    # Calculate metrics
    metrics = calculate_ona_metrics(current_network, simulator.employees)
    
    if page == "Executive Summary":
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
    
    elif page == "Network Visualization":
        st.header("🌐 Interactive Network Visualization")
        
        # Visualization controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            color_by = st.selectbox("Color nodes by:", 
                                  ["department", "performance", "tenure", "connections", 
                                   "engagement", "satisfaction", "burnout_risk", "influence_potential", "retention_likelihood"])
        
        with col2:
            layout_type = st.selectbox("Layout algorithm:", 
                                     ["spring", "circular", "shell", "random", "hierarchical"])
        
        with col3:
            if st.button("Generate Sample Network"):
                # Quick simulation to generate connections
                params = {
                    'edges': -1.5,
                    'same_department': 1.0,
                    'hierarchy': 0.5,
                    'performance_similarity': 0.3,
                    'tenure_similarity': 0.2,
                    'engagement_similarity': 0.3,
                    'collaboration_boost': 0.4,
                    'communication_affinity': 0.2,
                    'triangles': 0.4,
                    'preferential_attachment': 0.3,
                    'dissolution': 0.05
                }
                
                for _ in range(10):
                    simulator.simulate_step(params)
                st.rerun()
        
        # Filtering controls
        st.subheader("🔍 Filters")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            departments = list(set(emp.department for emp in simulator.employees))
            filter_department = st.selectbox("Filter by department:", 
                                           ["All"] + departments, 
                                           index=0)
            if filter_department == "All":
                filter_department = None
        
        with col2:
            roles = list(set(emp.role for emp in simulator.employees))
            filter_role = st.selectbox("Filter by role:", 
                                     ["All"] + roles, 
                                     index=0)
            if filter_role == "All":
                filter_role = None
        
        with col3:
            engagement_filter = st.selectbox("Filter by engagement:",
                                           ["All", "High Engagement", "Low Engagement", 
                                            "Burnout Risk", "High Influence"],
                                           index=0)
            if engagement_filter == "All":
                engagement_filter = None
        
        with col4:
            # Search functionality
            search_term = st.text_input("Search employees:", 
                                      placeholder="Enter name or keyword...")
            if search_term:
                # Find matching employees
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
            else:
                matching_employees = None
        
        # Network statistics for filtered view
        if filter_department or filter_role:
            filtered_nodes = list(current_network.nodes())
            if filter_department:
                filtered_nodes = [n for n in filtered_nodes if simulator.employees[n].department == filter_department]
            if filter_role:
                filtered_nodes = [n for n in filtered_nodes if filter_role.lower() in simulator.employees[n].role.lower()]
            
            filtered_network = current_network.subgraph(filtered_nodes)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Filtered Nodes", filtered_network.number_of_nodes())
            with col2:
                st.metric("Filtered Edges", filtered_network.number_of_edges())
            with col3:
                density = nx.density(filtered_network) if filtered_network.number_of_nodes() > 1 else 0
                st.metric("Filtered Density", f"{density:.3f}")
            with col4:
                components = nx.number_connected_components(filtered_network)
                st.metric("Components", components)
        
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
        
        # Legend and help
        with st.expander("🎯 Visualization Guide"):
            st.markdown("""
            **Interactive Features:**
            - **Zoom**: Use zoom button or mouse wheel
            - **Pan**: Drag to move around the network
            - **Hover**: See detailed employee information
            - **Edge Weights**: Thicker/darker edges indicate stronger connections
            
            **Node Shapes:**
            - 🔷 **Diamond**: Managers and Executives
            - ⬜ **Square**: Senior roles
            - ⭕ **Circle**: Other roles
            
            **Node Sizes**: Larger nodes have more connections
            
            **Connection Strength**: Based on department similarity, performance alignment, and tenure proximity
            """)
        
        # Network insights
        if current_network.number_of_edges() > 0:
            st.subheader("📊 Network Insights")
            
            # Calculate some quick insights
            degree_centrality = nx.degree_centrality(current_network)
            avg_degree = sum(dict(current_network.degree()).values()) / current_network.number_of_nodes()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Average Connections per Employee", f"{avg_degree:.1f}")
                
                # Most connected department
                dept_connections = {}
                for node in current_network.nodes():
                    dept = simulator.employees[node].department
                    if dept not in dept_connections:
                        dept_connections[dept] = 0
                    dept_connections[dept] += current_network.degree(node)
                
                if dept_connections:
                    most_connected_dept = max(dept_connections.items(), key=lambda x: x[1])
                    st.metric("Most Connected Department", most_connected_dept[0])
            
            with col2:
                # Network diameter (if connected)
                if nx.is_connected(current_network):
                    diameter = nx.diameter(current_network)
                    st.metric("Network Diameter", f"{diameter} steps")
                else:
                    largest_cc = max(nx.connected_components(current_network), key=len)
                    cc_subgraph = current_network.subgraph(largest_cc)
                    diameter = nx.diameter(cc_subgraph)
                    st.metric("Largest Component Diameter", f"{diameter} steps")
    
    elif page == "Centrality Analysis":
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
    
    elif page == "Department Analysis":
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
    
    elif page == "Talent Risk":
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
    
    elif page == "Network Health":
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
        
        if density < 0.1:
            st.warning("**Low Network Density:** Consider team-building activities or cross-departmental projects to increase connections.")
        
        if clustering < 0.3:
            st.warning("**Low Clustering:** Encourage team formation and collaborative projects to build stronger working relationships.")
        
        if components > 5:
            st.warning("**Network Fragmentation:** Multiple disconnected groups detected. Consider initiatives to bridge these groups.")
        
        if density > 0.1 and clustering > 0.3 and components < 3:
            st.success("**Healthy Network:** Good balance of connections and collaboration patterns.")
    
    elif page == "Employee Engagement":
        st.header("📊 Employee Engagement Analysis")
        
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
            
            # Risk analysis tables
            st.subheader("⚠️ Risk Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**High Burnout Risk Employees:**")
                burnout_employees = risk_analysis.get('burnout_risk_employees', [])
                if burnout_employees:
                    burnout_df = pd.DataFrame(burnout_employees, columns=['Employee', 'Department', 'Burnout Risk'])
                    st.dataframe(burnout_df)
                else:
                    st.info("No employees at high burnout risk.")
            
            with col2:
                st.write("**Retention Risk Employees:**")
                retention_employees = risk_analysis.get('retention_risk_employees', [])
                if retention_employees:
                    retention_df = pd.DataFrame(retention_employees, columns=['Employee', 'Department', 'Retention Likelihood'])
                    st.dataframe(retention_df)
                else:
                    st.info("No employees at retention risk.")
            
            # Department engagement breakdown
            st.subheader("🏢 Department Engagement Analysis")
            dept_engagement = engagement_metrics.get('department_engagement', {})
            
            if dept_engagement:
                dept_summary = []
                for dept, data in dept_engagement.items():
                    dept_summary.append({
                        'Department': dept,
                        'Mean Engagement': f"{data['mean_engagement']:.2f}",
                        'Total Employees': data['employees_count'],
                        'High Engagement': data['high_engagement_count'],
                        'Low Engagement': data['low_engagement_count']
                    })
                
                dept_summary_df = pd.DataFrame(dept_summary)
                st.dataframe(dept_summary_df, use_container_width=True)
        
        else:
            st.info("Generate a network first to see engagement analysis.")
    
    elif page == "Influence Analysis":
        st.header("⭐ Organizational Influence Analysis")
        
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
                
                # Network correlation insights
                engagement_metrics = metrics.get('engagement_metrics', {})
                network_corr = engagement_metrics.get('network_correlation', {})
                
                if network_corr:
                    st.subheader("🔗 Network-Engagement Correlations")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        eng_corr = network_corr.get('engagement_degree_correlation', 0)
                        st.metric("Engagement-Network Correlation", f"{eng_corr:.3f}")
                        if eng_corr > 0.3:
                            st.success("Strong positive correlation: Engaged employees are well-connected")
                        elif eng_corr < -0.3:
                            st.warning("Negative correlation: Engaged employees may be isolated")
                        else:
                            st.info("Weak correlation between engagement and network position")
                    
                    with col2:
                        collab_corr = network_corr.get('collaboration_degree_correlation', 0)
                        st.metric("Collaboration-Network Correlation", f"{collab_corr:.3f}")
                        if collab_corr > 0.3:
                            st.success("Strong positive correlation: Collaborative employees are well-connected")
                        elif collab_corr < -0.3:
                            st.warning("Negative correlation: Collaborative employees may be isolated")
                        else:
                            st.info("Weak correlation between collaboration and network position")
        
        else:
            st.info("Generate a network first to see influence analysis.")
    
    elif page == "Simulation":
        st.header("🔬 Interactive Network Simulation")
        
        st.sidebar.subheader("ONA Parameters")
        
        # ONA-specific parameters
        edge_param = st.sidebar.slider("Baseline Connection Probability", -3.0, 1.0, -1.5, 0.1)
        dept_param = st.sidebar.slider("Same Department Bonus", 0.0, 2.0, 1.0, 0.1)
        hierarchy_param = st.sidebar.slider("Hierarchy Effect", 0.0, 2.0, 0.5, 0.1)
        performance_param = st.sidebar.slider("Performance Similarity", 0.0, 1.0, 0.3, 0.1)
        tenure_param = st.sidebar.slider("Tenure Similarity", 0.0, 1.0, 0.2, 0.1)
        
        # Engagement-based parameters
        st.sidebar.subheader("Engagement Parameters")
        engagement_param = st.sidebar.slider("Engagement Similarity", 0.0, 1.0, 0.3, 0.1)
        collaboration_param = st.sidebar.slider("Collaboration Boost", 0.0, 1.0, 0.4, 0.1)
        communication_param = st.sidebar.slider("Communication Affinity", 0.0, 1.0, 0.2, 0.1)
        
        # Network structure parameters
        st.sidebar.subheader("Network Structure")
        triangle_param = st.sidebar.slider("Triangle Closure", 0.0, 2.0, 0.4, 0.1)
        preferential_param = st.sidebar.slider("Preferential Attachment", 0.0, 2.0, 0.3, 0.1)
        dissolution_param = st.sidebar.slider("Connection Dissolution", 0.0, 0.2, 0.05, 0.01)
        
        # Simulation controls
        st.sidebar.subheader("Simulation Controls")
        time_steps = st.sidebar.slider("Time Steps", 1, 50, 10)
        animation_speed = st.sidebar.slider("Animation Speed (seconds)", 0.1, 3.0, 1.0, 0.1)
        
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
                if 'current_step' in st.session_state:
                    del st.session_state.current_step
                st.rerun()
        
        with col2:
            if st.button("⏭️ Single Step"):
                simulator.simulate_step(params)
                
                # Update simulation history
                if 'simulation_history' not in st.session_state:
                    st.session_state.simulation_history = []
                
                # Calculate metrics for this step
                step_metrics = {
                    'step': len(simulator.networks),
                    'nodes': current_network.number_of_nodes(),
                    'edges': current_network.number_of_edges(),
                    'density': nx.density(current_network),
                    'clustering': nx.average_clustering(current_network),
                    'components': nx.number_connected_components(current_network)
                }
                st.session_state.simulation_history.append(step_metrics)
                st.rerun()
        
        with col3:
            if st.button("▶️ Run Animated Simulation"):
                simulator.reset_network()
                
                # Initialize simulation tracking
                st.session_state.simulation_history = []
                st.session_state.running_simulation = True
                
                # Show animation section
                st.write("---")
                st.markdown("### 🎬 Running Simulation Animation")
                
                # Create placeholders for real-time updates
                progress_placeholder = st.empty()
                metrics_placeholder = st.empty()
                network_placeholder = st.empty()
                
                for step in range(time_steps):
                    # Run simulation step
                    simulator.simulate_step(params)
                    current_net = simulator.networks[-1]
                    
                    # Calculate metrics
                    step_metrics = {
                        'step': step + 1,
                        'nodes': current_net.number_of_nodes(),
                        'edges': current_net.number_of_edges(),
                        'density': nx.density(current_net),
                        'clustering': nx.average_clustering(current_net),
                        'components': nx.number_connected_components(current_net)
                    }
                    st.session_state.simulation_history.append(step_metrics)
                    
                    # Update progress
                    progress_placeholder.progress((step + 1) / time_steps, 
                                                f"Step {step + 1}/{time_steps}: {current_net.number_of_edges()} connections")
                    
                    # Update real-time metrics
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
                    
                    # Update network visualization - simplified approach
                    with network_placeholder:
                        st.write(f"**Network at Step {step + 1}**")
                        fig = create_ona_network_plot(current_net, simulator.employees, 
                                                    color_by="department", layout_type="spring")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Animation delay
                    time.sleep(animation_speed)
                
                # Clear progress and show completion
                progress_placeholder.empty()
                st.session_state.running_simulation = False
                st.success(f"✅ Simulation completed! Final network has {current_net.number_of_edges()} connections.")
                st.write("---")
        
        with col4:
            if st.button("📈 Quick Sample"):
                # Generate a quick sample network for demonstration
                simulator.reset_network()
                sample_params = {
                    'edges': -1.0,
                    'same_department': 1.2,
                    'hierarchy': 0.8,
                    'performance_similarity': 0.4,
                    'tenure_similarity': 0.3,
                    'engagement_similarity': 0.5,
                    'collaboration_boost': 0.6,
                    'communication_affinity': 0.3,
                    'triangles': 0.6,
                    'preferential_attachment': 0.4,
                    'dissolution': 0.03
                }
                
                for _ in range(15):
                    simulator.simulate_step(sample_params)
                st.rerun()
        
        # Network visualization controls
        st.subheader("🎨 Visualization Controls")
        col1, col2 = st.columns(2)
        with col1:
            sim_layout = st.selectbox("Layout Algorithm:", 
                                    ["spring", "circular", "shell", "random"], 
                                    key="sim_layout")
        with col2:
            sim_color_by = st.selectbox("Color Nodes By:", 
                                      ["department", "performance", "tenure", "connections",
                                       "engagement", "satisfaction", "burnout_risk", "influence_potential"],
                                      key="sim_color")
        
        # Current network state
        if simulator.networks:
            current_step = len(simulator.networks)
            st.subheader(f"🌐 Current Network State (Step {current_step})")
        else:
            st.subheader("🌐 Initial Network State")
        
        # Network visualization
        fig = create_ona_network_plot(current_network, simulator.employees, 
                                    color_by=sim_color_by, layout_type=sim_layout)
        fig.update_layout(height=600, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
        
        # Show simulation history if available
        if 'simulation_history' in st.session_state and st.session_state.simulation_history:
            st.subheader("📊 Simulation Evolution")
            
            # Create time series data
            history_df = pd.DataFrame(st.session_state.simulation_history)
            
            # Time series plots
            col1, col2 = st.columns(2)
            
            with col1:
                fig_edges = px.line(history_df, x='step', y='edges', 
                                  title='Connection Growth Over Time',
                                  labels={'step': 'Simulation Step', 'edges': 'Number of Connections'})
                fig_edges.update_traces(line=dict(color='#1f77b4', width=3))
                st.plotly_chart(fig_edges, use_container_width=True)
                
                fig_clustering = px.line(history_df, x='step', y='clustering',
                                       title='Clustering Coefficient Over Time',
                                       labels={'step': 'Simulation Step', 'clustering': 'Clustering Coefficient'})
                fig_clustering.update_traces(line=dict(color='#ff7f0e', width=3))
                st.plotly_chart(fig_clustering, use_container_width=True)
            
            with col2:
                fig_density = px.line(history_df, x='step', y='density',
                                    title='Network Density Over Time',
                                    labels={'step': 'Simulation Step', 'density': 'Network Density'})
                fig_density.update_traces(line=dict(color='#2ca02c', width=3))
                st.plotly_chart(fig_density, use_container_width=True)
                
                fig_components = px.line(history_df, x='step', y='components',
                                       title='Network Fragmentation Over Time',
                                       labels={'step': 'Simulation Step', 'components': 'Connected Components'})
                fig_components.update_traces(line=dict(color='#d62728', width=3))
                st.plotly_chart(fig_components, use_container_width=True)
            
            # Summary statistics table
            st.subheader("📋 Simulation Summary")
            if len(history_df) > 0:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Steps", len(history_df))
                    st.metric("Final Connections", history_df['edges'].iloc[-1])
                
                with col2:
                    st.metric("Peak Connections", history_df['edges'].max())
                    st.metric("Avg Growth Rate", f"{(history_df['edges'].iloc[-1] / len(history_df)):.1f} conn/step")
                
                with col3:
                    st.metric("Final Density", f"{history_df['density'].iloc[-1]:.3f}")
                    st.metric("Final Clustering", f"{history_df['clustering'].iloc[-1]:.3f}")
                
                # Detailed history table
                with st.expander("📊 Detailed Step History"):
                    st.dataframe(history_df.round(3), use_container_width=True)
        
        # Simulation insights
        if current_network.number_of_edges() > 0:
            st.subheader("🔍 Current Network Insights")
            
            # Calculate department connectivity
            dept_stats = {}
            for node in current_network.nodes():
                dept = simulator.employees[node].department
                if dept not in dept_stats:
                    dept_stats[dept] = {'nodes': 0, 'total_connections': 0}
                dept_stats[dept]['nodes'] += 1
                dept_stats[dept]['total_connections'] += current_network.degree(node)
            
            # Display department insights
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Department Connectivity:**")
                for dept, stats in dept_stats.items():
                    avg_conn = stats['total_connections'] / stats['nodes'] if stats['nodes'] > 0 else 0
                    st.write(f"• {dept}: {avg_conn:.1f} avg connections per employee")
            
            with col2:
                # Most connected employees
                if current_network.number_of_edges() > 0:
                    degrees = dict(current_network.degree())
                    top_connected = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
                    
                    st.write("**Most Connected Employees:**")
                    for emp_id, degree in top_connected:
                        emp = simulator.employees[emp_id]
                        st.write(f"• {emp.name} ({emp.department}): {degree} connections")
        
        # Help section
        with st.expander("ℹ️ Simulation Help"):
            st.markdown("""
            **Simulation Controls:**
            - **🔄 Reset Network**: Clear all connections and start fresh
            - **⏭️ Single Step**: Run one simulation step and see immediate results
            - **▶️ Run Animated Simulation**: Watch network grow step-by-step with real-time visualization
            - **📈 Quick Sample**: Generate a demonstration network with balanced parameters
            
            **Animation Features:**
            - Real-time network visualization updates each step
            - Live metrics tracking (connections, density, clustering)
            - Progress indicator with step-by-step details
            - Time series charts showing network evolution
            
            **Understanding the Results:**
            - **Connection Growth**: How relationships form over time
            - **Density Evolution**: Overall connectivity level changes
            - **Clustering Development**: Team formation patterns
            - **Fragmentation**: Whether the organization stays connected
            """)
    
    # Export functionality
    st.sidebar.subheader("Export Data")
    if st.sidebar.button("Export Network Data"):
        # Create export data
        export_data = {
            'employees': [
                {
                    'id': emp.id,
                    'name': emp.name,
                    'department': emp.department,
                    'role': emp.role,
                    'tenure': emp.tenure,
                    'performance': emp.performance
                }
                for emp in simulator.employees
            ],
            'connections': list(current_network.edges()),
            'metrics': {
                'network_density': nx.density(current_network),
                'clustering_coefficient': nx.average_clustering(current_network),
                'connected_components': nx.number_connected_components(current_network)
            }
        }
        
        st.sidebar.download_button(
            label="Download JSON",
            data=json.dumps(export_data, indent=2),
            file_name=f"ona_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()