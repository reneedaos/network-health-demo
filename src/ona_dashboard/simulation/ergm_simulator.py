"""
ERGM simulation engine for organizational network analysis.
"""

import networkx as nx
import numpy as np
from typing import Dict, List
from ..models.employee import Employee
from ..utils.generators import generate_employees


class ONASimulator:
    """Organizational Network Analysis ERGM Simulator."""
    
    def __init__(self, n_employees: int = 50):
        self.n_employees = n_employees
        self.employees = generate_employees(n_employees)
        self.reset_network()
    
    def reset_network(self):
        """Reset the network to initial state."""
        self.G = nx.Graph()
        self.G.add_nodes_from(range(self.n_employees))
        
        # Add node attributes
        for i, emp in enumerate(self.employees):
            self.G.nodes[i].update(emp.to_dict())
        
        self.time_steps = []
        self.networks = []
    
    def ergm_probability(self, i: int, j: int, params: Dict) -> float:
        """Calculate ERGM probability for edge (i,j) with organizational context."""
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
        """Single simulation step."""
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
    
    def get_current_network(self) -> nx.Graph:
        """Get the current network state."""
        if self.networks:
            return self.networks[-1]
        return self.G
    
    def get_network_history(self) -> List[nx.Graph]:
        """Get the complete network evolution history."""
        return self.networks
    
    def export_data(self) -> Dict:
        """Export simulation data for analysis."""
        current_network = self.get_current_network()
        
        return {
            'employees': [emp.to_dict() for emp in self.employees],
            'connections': list(current_network.edges()),
            'metrics': {
                'network_density': nx.density(current_network),
                'clustering_coefficient': nx.average_clustering(current_network),
                'connected_components': nx.number_connected_components(current_network),
                'number_of_nodes': current_network.number_of_nodes(),
                'number_of_edges': current_network.number_of_edges()
            }
        }


def get_default_simulation_params() -> Dict:
    """Get default ERGM simulation parameters."""
    return {
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


def get_quick_sample_params() -> Dict:
    """Get parameters for quick sample network generation."""
    return {
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