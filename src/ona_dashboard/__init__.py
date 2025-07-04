"""
ONA Dashboard - Organizational Network Analysis Dashboard

A modular Streamlit application for organizational network analysis.
"""

__version__ = "1.0.0"
__author__ = "ONA Dashboard Team"

from .models.employee import Employee
from .simulation.ergm_simulator import ONASimulator
from .utils.metrics import calculate_ona_metrics, calculate_engagement_metrics
from .visualizations.network_plots import create_ona_network_plot
from .visualizations.analysis_plots import (
    create_centrality_analysis,
    create_department_analysis,
    create_network_health_dashboard,
    create_talent_risk_analysis,
    create_engagement_dashboard,
    create_influence_network_analysis
)
from .config.settings import DashboardConfig

__all__ = [
    'Employee',
    'ONASimulator',
    'calculate_ona_metrics',
    'calculate_engagement_metrics',
    'create_ona_network_plot',
    'create_centrality_analysis',
    'create_department_analysis',
    'create_network_health_dashboard',
    'create_talent_risk_analysis',
    'create_engagement_dashboard',
    'create_influence_network_analysis',
    'DashboardConfig'
]