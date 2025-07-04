"""
Network visualization components for ONA dashboard.
"""

import networkx as nx
import plotly.graph_objects as go
from typing import List, Dict, Optional
from ..models.employee import Employee
from ..utils.metrics import calculate_edge_weight


def create_ona_network_plot(G: nx.Graph, employees: List[Employee], color_by: str = 'department', 
                           layout_type: str = 'spring', selected_nodes: List[int] = None,
                           filter_department: str = None, filter_role: str = None, 
                           engagement_filter: str = None):
    """Create interactive ONA network visualization."""
    if G.number_of_edges() == 0:
        return _create_empty_network_plot()
    
    # Apply filters
    filtered_nodes = _apply_filters(G, employees, filter_department, filter_role, engagement_filter)
    
    # Create subgraph with filtered nodes
    G_filtered = G.subgraph(filtered_nodes)
    
    if G_filtered.number_of_nodes() == 0:
        return _create_empty_network_plot("No nodes match the current filters")
    
    # Get layout positions
    pos = _get_layout_positions(G_filtered, layout_type)
    
    # Create the plot
    fig = go.Figure()
    
    # Add edges
    _add_edges_to_plot(fig, G_filtered, employees, pos)
    
    # Add nodes
    _add_nodes_to_plot(fig, G_filtered, employees, pos, color_by)
    
    # Highlight selected nodes
    if selected_nodes:
        _highlight_selected_nodes(fig, selected_nodes, G_filtered, pos)
    
    # Update layout
    _update_plot_layout(fig, layout_type)
    
    return fig


def _create_empty_network_plot(message: str = "No connections in the network yet"):
    """Create an empty network plot with a message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
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


def _apply_filters(G: nx.Graph, employees: List[Employee], filter_department: str = None, 
                  filter_role: str = None, engagement_filter: str = None) -> List[int]:
    """Apply filters to get relevant nodes."""
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
    
    return filtered_nodes


def _get_layout_positions(G: nx.Graph, layout_type: str) -> Dict:
    """Get node positions based on layout type."""
    if layout_type == 'spring':
        return nx.spring_layout(G, k=2, iterations=50)
    elif layout_type == 'circular':
        return nx.circular_layout(G)
    elif layout_type == 'random':
        return nx.random_layout(G)
    elif layout_type == 'shell':
        return nx.shell_layout(G)
    elif layout_type == 'hierarchical':
        try:
            return nx.nx_agraph.graphviz_layout(G, prog='dot')
        except:
            return nx.spring_layout(G, k=2, iterations=50)
    else:
        return nx.spring_layout(G, k=2, iterations=50)


def _add_edges_to_plot(fig: go.Figure, G: nx.Graph, employees: List[Employee], pos: Dict):
    """Add edges to the plot with varying weights."""
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        # Calculate edge weight
        emp1, emp2 = employees[edge[0]], employees[edge[1]]
        weight = calculate_edge_weight(emp1, emp2)
        
        opacity = min(0.8, 0.2 + weight * 0.3)
        width = min(3, 0.5 + weight * 0.5)
        
        fig.add_trace(go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode='lines',
            line=dict(width=width, color=f'rgba(125,125,125,{opacity})'),
            hoverinfo='text',
            hovertext=f"{emp1.name} ↔ {emp2.name}<br>Connection Strength: {weight:.2f}",
            showlegend=False
        ))


def _add_nodes_to_plot(fig: go.Figure, G: nx.Graph, employees: List[Employee], pos: Dict, color_by: str):
    """Add nodes to the plot with appropriate styling."""
    node_x = []
    node_y = []
    node_info = []
    node_colors = []
    node_sizes = []
    node_symbols = []
    
    departments = list(set(emp.department for emp in employees))
    dept_color_map = {dept: i for i, dept in enumerate(departments)}
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        emp = employees[node]
        degree = G.degree(node)
        
        # Node information
        node_info.append(_create_node_info(emp, degree))
        
        # Node color
        node_colors.append(_get_node_color(emp, color_by, dept_color_map, degree))
        
        # Node size
        node_sizes.append(_get_node_size(emp, degree))
        
        # Node symbol
        node_symbols.append(_get_node_symbol(emp))
    
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
        customdata=list(G.nodes()),
    ))


def _create_node_info(emp: Employee, degree: int) -> str:
    """Create hover information for a node."""
    return (
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


def _get_node_color(emp: Employee, color_by: str, dept_color_map: Dict, degree: int) -> float:
    """Get node color based on coloring scheme."""
    if color_by == 'department':
        return dept_color_map[emp.department]
    elif color_by == 'performance':
        return emp.performance
    elif color_by == 'tenure':
        return emp.tenure
    elif color_by == 'engagement':
        return emp.engagement_score
    elif color_by == 'satisfaction':
        return emp.satisfaction_score
    elif color_by == 'burnout_risk':
        return emp.burnout_risk
    elif color_by == 'influence_potential':
        return emp.influence_potential
    elif color_by == 'retention_likelihood':
        return emp.retention_likelihood
    else:
        return degree


def _get_node_size(emp: Employee, degree: int) -> int:
    """Get node size based on degree and engagement."""
    base_size = max(10, min(25, 8 + degree * 2))
    
    # Boost size for high engagement employees
    if emp.engagement_score > 0.8:
        base_size += 3
    
    # Reduce size for burnout risk employees
    if emp.burnout_risk > 0.7:
        base_size = max(8, base_size - 2)
    
    return base_size


def _get_node_symbol(emp: Employee) -> str:
    """Get node symbol based on role and engagement status."""
    if 'Manager' in emp.role or 'Executive' in emp.role:
        return 'diamond'
    elif 'Senior' in emp.role:
        return 'square'
    elif emp.burnout_risk > 0.7:
        return 'triangle-up'  # Warning symbol for burnout risk
    elif emp.influence_potential > 0.8:
        return 'star'  # Star for high influence
    else:
        return 'circle'


def _highlight_selected_nodes(fig: go.Figure, selected_nodes: List[int], G: nx.Graph, pos: Dict):
    """Highlight selected nodes on the plot."""
    selected_x = [pos[n][0] for n in selected_nodes if n in G.nodes()]
    selected_y = [pos[n][1] for n in selected_nodes if n in G.nodes()]
    
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


def _update_plot_layout(fig: go.Figure, layout_type: str):
    """Update the plot layout with interactive controls."""
    fig.update_layout(
        title=f"Organizational Network ({layout_type.title()} Layout)",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,
        autosize=True,
        dragmode='pan',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
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