"""
Configuration settings for ONA Dashboard.
"""

from typing import Dict, List
import streamlit as st


class DashboardConfig:
    """Configuration class for ONA Dashboard settings."""
    
    # Page configuration
    PAGE_TITLE = "Organizational Network Analysis Dashboard"
    PAGE_LAYOUT = "wide"
    
    # Default simulation parameters
    DEFAULT_N_EMPLOYEES = 50
    
    # Available departments and roles
    DEFAULT_DEPARTMENTS = ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance', 'Operations']
    DEFAULT_ROLES = ['Manager', 'Senior', 'Mid-level', 'Junior', 'Executive']
    
    # Visualization settings
    LAYOUT_ALGORITHMS = ["spring", "circular", "shell", "random", "hierarchical"]
    COLOR_SCHEMES = [
        "department", "performance", "tenure", "connections", 
        "engagement", "satisfaction", "burnout_risk", "influence_potential", "retention_likelihood"
    ]
    
    # Engagement filter options
    ENGAGEMENT_FILTERS = [
        "All", "High Engagement", "Low Engagement", 
        "Burnout Risk", "High Influence"
    ]
    
    # Navigation pages
    NAVIGATION_PAGES = [
        "Executive Summary", "Network Visualization", "Centrality Analysis", 
        "Department Analysis", "Talent Risk", "Network Health", 
        "Employee Engagement", "Influence Analysis", "Simulation"
    ]
    
    # Metric thresholds
    HIGH_ENGAGEMENT_THRESHOLD = 0.8
    LOW_ENGAGEMENT_THRESHOLD = 0.5
    HIGH_BURNOUT_THRESHOLD = 0.7
    HIGH_INFLUENCE_THRESHOLD = 0.8
    LOW_RETENTION_THRESHOLD = 0.5
    
    # Network health thresholds
    LOW_DENSITY_THRESHOLD = 0.1
    LOW_CLUSTERING_THRESHOLD = 0.3
    HIGH_FRAGMENTATION_THRESHOLD = 5
    
    # Simulation settings
    MAX_SIMULATION_STEPS = 50
    DEFAULT_SIMULATION_STEPS = 10
    MIN_ANIMATION_SPEED = 0.1
    MAX_ANIMATION_SPEED = 3.0
    DEFAULT_ANIMATION_SPEED = 1.0
    
    # Export settings
    EXPORT_DATETIME_FORMAT = '%Y%m%d_%H%M%S'
    
    @classmethod
    def configure_streamlit_page(cls):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title=cls.PAGE_TITLE, 
            layout=cls.PAGE_LAYOUT
        )
    
    @classmethod
    def get_simulation_param_ranges(cls) -> Dict:
        """Get parameter ranges for simulation sliders."""
        return {
            'edges': {'min_value': -3.0, 'max_value': 1.0, 'value': -1.5, 'step': 0.1},
            'same_department': {'min_value': 0.0, 'max_value': 2.0, 'value': 1.0, 'step': 0.1},
            'hierarchy': {'min_value': 0.0, 'max_value': 2.0, 'value': 0.5, 'step': 0.1},
            'performance_similarity': {'min_value': 0.0, 'max_value': 1.0, 'value': 0.3, 'step': 0.1},
            'tenure_similarity': {'min_value': 0.0, 'max_value': 1.0, 'value': 0.2, 'step': 0.1},
            'engagement_similarity': {'min_value': 0.0, 'max_value': 1.0, 'value': 0.3, 'step': 0.1},
            'collaboration_boost': {'min_value': 0.0, 'max_value': 1.0, 'value': 0.4, 'step': 0.1},
            'communication_affinity': {'min_value': 0.0, 'max_value': 1.0, 'value': 0.2, 'step': 0.1},
            'triangles': {'min_value': 0.0, 'max_value': 2.0, 'value': 0.4, 'step': 0.1},
            'preferential_attachment': {'min_value': 0.0, 'max_value': 2.0, 'value': 0.3, 'step': 0.1},
            'dissolution': {'min_value': 0.0, 'max_value': 0.2, 'value': 0.05, 'step': 0.01}
        }
    
    @classmethod
    def get_visualization_help_text(cls) -> str:
        """Get help text for visualization guide."""
        return """
        **Interactive Features:**
        - **Zoom**: Use zoom button or mouse wheel
        - **Pan**: Drag to move around the network
        - **Hover**: See detailed employee information
        - **Edge Weights**: Thicker/darker edges indicate stronger connections
        
        **Node Shapes:**
        - 🔷 **Diamond**: Managers and Executives
        - ⬜ **Square**: Senior roles
        - ⭕ **Circle**: Other roles
        - ⚠️ **Triangle**: Burnout risk employees
        - ⭐ **Star**: High influence potential
        
        **Node Sizes**: Larger nodes have more connections
        
        **Connection Strength**: Based on department similarity, performance alignment, and tenure proximity
        """
    
    @classmethod
    def get_simulation_help_text(cls) -> str:
        """Get help text for simulation controls."""
        return """
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
        """