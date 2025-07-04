"""
Employee model for Organizational Network Analysis.
"""

import random
from typing import Dict, List, Optional


class Employee:
    """Represents an employee in the organization with engagement metrics."""
    
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
        """Calculate burnout risk based on engagement factors."""
        # Higher performance with lower satisfaction/engagement indicates potential burnout
        stress_factor = max(0, self.performance - self.satisfaction_score)
        low_engagement_factor = max(0, 0.7 - self.engagement_score)
        return min(1.0, stress_factor + low_engagement_factor)
    
    def _calculate_influence_potential(self) -> float:
        """Calculate potential for organizational influence."""
        return (self.engagement_score * 0.4 + 
                self.communication_frequency * 0.3 + 
                self.collaboration_index * 0.2 + 
                self.performance * 0.1)
    
    def _calculate_retention_likelihood(self) -> float:
        """Calculate likelihood of employee retention."""
        return (self.satisfaction_score * 0.35 + 
                self.engagement_score * 0.25 + 
                (self.tenure / 10) * 0.15 +  # Tenure factor (normalized)
                self.collaboration_index * 0.15 + 
                self.performance * 0.1)
    
    def to_dict(self) -> Dict:
        """Convert employee to dictionary for serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'department': self.department,
            'role': self.role,
            'tenure': self.tenure,
            'performance': self.performance,
            'engagement_score': self.engagement_score,
            'communication_frequency': self.communication_frequency,
            'collaboration_index': self.collaboration_index,
            'satisfaction_score': self.satisfaction_score,
            'burnout_risk': self.burnout_risk,
            'influence_potential': self.influence_potential,
            'retention_likelihood': self.retention_likelihood
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Employee':
        """Create employee from dictionary."""
        return cls(
            id=data['id'],
            name=data['name'],
            department=data['department'],
            role=data['role'],
            tenure=data['tenure'],
            performance=data['performance'],
            engagement_score=data.get('engagement_score'),
            communication_frequency=data.get('communication_frequency'),
            collaboration_index=data.get('collaboration_index'),
            satisfaction_score=data.get('satisfaction_score')
        )
    
    def __repr__(self) -> str:
        return f"Employee(id={self.id}, name='{self.name}', department='{self.department}', role='{self.role}')"