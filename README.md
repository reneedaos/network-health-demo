# 🏢 Organizational Network Analysis Dashboard

A comprehensive Streamlit application for Organizational Network Analysis (ONA) designed for HR professionals. Built with advanced ERGM modeling and interactive visualizations to understand workplace relationships, identify key influencers, and assess organizational health.

## 🚀 Key Features

### 📊 Executive Dashboard
- **Real-time ONA Metrics**: Employee connections, network density, fragmentation analysis
- **Key Insights**: Automatic identification of most connected and influential employees
- **Department Overview**: Cross-departmental connectivity analysis

### 🌐 Interactive Network Visualization
- **Multiple Layout Algorithms**: Spring, circular, shell, hierarchical layouts
- **Dynamic Filtering**: Filter by department, role, or search employees
- **Smart Visual Encoding**: Node shapes (roles), sizes (connections), colors (attributes)
- **Interactive Controls**: Zoom, pan, hover details, edge weight visualization
- **Connection Intelligence**: Weighted edges based on similarity metrics

### ⭐ Advanced Analytics
- **Centrality Analysis**: Degree, betweenness, closeness, eigenvector centrality
- **Talent Risk Assessment**: Identify flight risks and key talent
- **Department Analysis**: Internal connectivity and collaboration patterns
- **Network Health**: Automated health indicators and recommendations

### 🔬 ERGM Simulation
- **Organizational Context**: Department homophily, hierarchy effects, performance similarity
- **Real-time Parameter Adjustment**: Watch networks evolve with different organizational dynamics
- **Scenario Testing**: Model changes in organizational structure

## Installation

1. Clone the repository:
```bash
git clone https://github.com/reneedaos/network-health-demo.git
cd network-health-demo
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## 🎯 Usage Guide

### Navigation

The dashboard includes seven main views accessible via the sidebar:

1. **📊 Executive Summary**: High-level organizational insights and key metrics
2. **🌐 Network Visualization**: Interactive network exploration with filters
3. **⭐ Centrality Analysis**: Identify influential employees and bridge-builders  
4. **🏢 Department Analysis**: Cross-departmental connectivity patterns
5. **⚠️ Talent Risk**: Assess flight risk and identify key talent
6. **🏥 Network Health**: Organizational health indicators and recommendations
7. **🔬 Simulation**: ERGM modeling and scenario testing

### Interactive Features

#### Network Visualization
- **Layout Selection**: Choose from spring, circular, shell, random, or hierarchical layouts
- **Filtering**: Filter by department, role, or search for specific employees
- **Visual Controls**: Color by department, performance, tenure, or connections
- **Interactive Elements**: Zoom, pan, hover for details, clickable nodes

#### ONA Parameters (Simulation)
- **Same Department Bonus**: Tendency for colleagues to connect (0.0-2.0)
- **Hierarchy Effect**: Manager connectivity preference (0.0-2.0) 
- **Performance Similarity**: Performance-based homophily (0.0-1.0)
- **Tenure Similarity**: Experience-based connections (0.0-1.0)
- **Triangle Closure**: Tendency to form collaborative triangles (0.0-2.0)

### Key Metrics Explained

#### Centrality Measures
- **Degree Centrality**: Number of direct connections (popularity)
- **Betweenness Centrality**: Bridge-building capacity (information flow)
- **Closeness Centrality**: Access to information (efficiency)
- **Eigenvector Centrality**: Influence through high-status connections

#### Network Health Indicators
- **Density**: Overall connectivity level (0-1 scale)
- **Clustering Coefficient**: Team formation tendency (0-1 scale) 
- **Fragmentation**: Number of disconnected groups
- **Component Size**: Largest connected group size

## 🧠 ERGM Model for Organizations

The application implements an organizational ERGM that models workplace relationship formation based on:

### Structural Effects
- **Baseline Connection Probability**: Overall likelihood of any two employees connecting
- **Triangle Closure**: Tendency for mutual colleagues to connect (team formation)
- **Preferential Attachment**: High-connected employees attract more connections

### Organizational Effects  
- **Department Homophily**: Same-department employees more likely to connect
- **Hierarchy Effects**: Managers and executives have higher connectivity
- **Performance Similarity**: Similar performers tend to connect
- **Tenure Similarity**: Employees with similar experience levels connect more

### Edge Formation Probability
```
P(connection) = 1 / (1 + exp(-logit))

logit = baseline + same_dept_bonus + hierarchy_effect + 
        performance_similarity + tenure_similarity + 
        triangle_closure + preferential_attachment
```

## 📊 Visualization Features

### Interactive Network Display
- **Dynamic Layouts**: Multiple algorithms for different perspectives
- **Smart Node Encoding**: 
  - Size: Connection count
  - Shape: Role level (diamond=managers, square=senior, circle=others)
  - Color: Department, performance, tenure, or connections
- **Weighted Edges**: Thickness indicates relationship strength
- **Hover Information**: Detailed employee and connection data

### Advanced Controls
- **Real-time Filtering**: Department and role filters
- **Employee Search**: Find specific individuals
- **Zoom/Pan Controls**: Navigate large networks
- **Layout Switching**: Dynamic algorithm changes

### Analytics Dashboard
- **Centrality Rankings**: Top influencers and bridge-builders  
- **Department Heatmaps**: Cross-functional collaboration
- **Risk Assessments**: Flight risk and key talent identification
- **Health Metrics**: Network density, clustering, fragmentation

## 🛠️ Technical Stack

### Dependencies
```
streamlit>=1.28.0    # Web interface framework
numpy>=1.24.0        # Numerical computations  
pandas>=2.0.0        # Data manipulation
networkx>=3.1        # Graph algorithms
matplotlib>=3.7.0    # Static plotting
plotly>=5.15.0       # Interactive visualizations
scipy>=1.11.0        # Statistical functions
```

### Architecture
- **NetworkX**: Graph operations, centrality calculations, layout algorithms
- **Plotly**: Interactive network visualizations with zoom/pan/hover
- **Streamlit**: Multi-page dashboard interface with real-time updates
- **NumPy/SciPy**: ERGM probability calculations and statistical analysis
- **Pandas**: Employee data management and metrics aggregation

## 🎯 Use Cases for HR Professionals

### Organizational Design
- **Identify Silos**: Find disconnected departments needing bridge-building
- **Optimize Communication**: Locate information bottlenecks and super-connectors
- **Plan Restructuring**: Model impact of organizational changes

### Talent Management  
- **Succession Planning**: Identify key individuals and their network dependencies
- **Retention Strategy**: Spot isolated employees at risk of leaving
- **Leadership Development**: Find emerging influencers and bridge-builders

### Team Effectiveness
- **Cross-functional Collaboration**: Measure and improve department connectivity
- **Knowledge Sharing**: Optimize information flow pathways
- **Team Formation**: Use clustering insights for project team assembly

## 🔬 Research Applications

While designed for practical HR use, this tool also serves educational purposes for:
- Network science and social network analysis
- Organizational behavior research  
- ERGM modeling demonstrations
- Interactive data visualization examples

For advanced research applications, consider specialized packages like `statnet` (R) or `ergm` (Python).

## 📄 License

This project is for educational and demonstration purposes. See individual package licenses for dependencies.