"""
Utility functions for calculating ONA and engagement metrics.
"""

import networkx as nx
import numpy as np
from typing import Dict, List
from ..models.employee import Employee


def calculate_engagement_metrics(employees: List[Employee], G: nx.Graph = None) -> Dict:
    """Calculate comprehensive employee engagement metrics."""
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
    """Calculate key ONA metrics including engagement analysis."""
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


def calculate_edge_weight(emp1: Employee, emp2: Employee) -> float:
    """Calculate edge weight based on employee similarity."""
    weight = 1.0
    if emp1.department == emp2.department:
        weight += 0.5
    if abs(emp1.performance - emp2.performance) < 0.2:
        weight += 0.3
    if abs(emp1.tenure - emp2.tenure) < 2:
        weight += 0.2
    return weight


def get_top_employees_by_metric(metrics: Dict, employees: List[Employee], metric: str, top_n: int = 10) -> List[Dict]:
    """Get top N employees by a specific centrality metric."""
    if metric not in metrics:
        return []
    
    top_employees = sorted(metrics[metric].items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    return [
        {
            'Employee': employees[emp_id].name,
            'Department': employees[emp_id].department,
            'Role': employees[emp_id].role,
            'Score': score
        }
        for emp_id, score in top_employees
    ]