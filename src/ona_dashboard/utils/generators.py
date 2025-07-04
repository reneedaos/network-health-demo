"""
Utility functions for generating employee data and network structures.
"""

import random
from typing import List
from ..models.employee import Employee


def generate_employees(n_employees: int = 50, 
                      departments: List[str] = None, 
                      roles: List[str] = None) -> List[Employee]:
    """Generate realistic employee data."""
    if departments is None:
        departments = ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance', 'Operations']
    if roles is None:
        roles = ['Manager', 'Senior', 'Mid-level', 'Junior', 'Executive']
    
    employees = []
    for i in range(n_employees):
        name = f"Employee_{i+1}"
        dept = random.choice(departments)
        role = random.choice(roles)
        tenure = random.randint(1, 10)
        performance = random.uniform(0.6, 1.0)
        
        # Generate engagement metrics with realistic correlations
        engagement_score = _generate_correlated_engagement(performance, tenure)
        communication_freq = _generate_communication_frequency(role, dept)
        collaboration_idx = _generate_collaboration_index(role, performance)
        satisfaction = _generate_satisfaction(performance, tenure, engagement_score)
        
        employees.append(Employee(i, name, dept, role, tenure, performance,
                                engagement_score, communication_freq, 
                                collaboration_idx, satisfaction))
    
    return employees


def _generate_correlated_engagement(performance: float, tenure: int) -> float:
    """Generate engagement score with realistic correlations."""
    # Higher performance generally correlates with higher engagement
    # But very long tenure might reduce engagement (stagnation)
    base_engagement = performance * 0.7 + random.uniform(0.1, 0.3)
    tenure_factor = 1.0 if tenure < 7 else max(0.7, 1.0 - (tenure - 7) * 0.05)
    return min(1.0, base_engagement * tenure_factor)


def _generate_communication_frequency(role: str, department: str) -> float:
    """Generate communication frequency based on role and department."""
    base_freq = random.uniform(0.3, 0.8)
    if 'Manager' in role or 'Executive' in role:
        base_freq += 0.2
    if department in ['Sales', 'Marketing', 'HR']:
        base_freq += 0.1
    return min(1.0, base_freq)


def _generate_collaboration_index(role: str, performance: float) -> float:
    """Generate collaboration index based on role and performance."""
    base_collab = random.uniform(0.2, 0.7)
    if 'Senior' in role:
        base_collab += 0.15
    # High performers often collaborate more
    if performance > 0.8:
        base_collab += 0.1
    return min(1.0, base_collab)


def _generate_satisfaction(performance: float, tenure: int, engagement: float) -> float:
    """Generate satisfaction score with realistic correlations."""
    # Satisfaction correlates with performance and engagement
    # But decreases with very long tenure (career stagnation)
    base_satisfaction = (performance * 0.4 + engagement * 0.4 + random.uniform(0.1, 0.2))
    if tenure > 8:
        base_satisfaction *= 0.9  # Slight decrease for long tenure
    return min(1.0, base_satisfaction)