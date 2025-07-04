# 🏢 Organizational Network Analysis Dashboard

A comprehensive, modular Streamlit application for Organizational Network Analysis (ONA) designed for HR professionals. Built with advanced ERGM modeling and interactive visualizations to understand workplace relationships, identify key influencers, and assess organizational health.

> **✨ Modular Architecture!** This application features a clean, portable package structure with separated concerns for better development experience, maintainability, and deployment flexibility.

## 🚀 Key Features

### 📊 Executive Dashboard
- **Real-time ONA Metrics**: Employee connections, network density, fragmentation analysis
- **Key Insights**: Automatic identification of most connected and influential employees
- **Department Overview**: Cross-departmental connectivity analysis
- **Engagement Overview**: Average engagement, burnout risk, and retention metrics

### 🌐 Interactive Network Visualization
- **Multiple Layout Algorithms**: Spring, circular, shell, hierarchical layouts
- **Dynamic Filtering**: Filter by department, role, engagement level, or search employees
- **Smart Visual Encoding**: Node shapes (roles + engagement status), sizes (connections), colors (performance/engagement)
- **Interactive Controls**: Zoom, pan, hover details, edge weight visualization
- **Connection Intelligence**: Weighted edges based on similarity and engagement metrics
- **Engagement Indicators**: Visual symbols for burnout risk and high influence potential

### ⭐ Advanced Analytics
- **Centrality Analysis**: Degree, betweenness, closeness, eigenvector centrality
- **Talent Risk Assessment**: Identify flight risks and key talent based on engagement
- **Department Analysis**: Internal connectivity and collaboration patterns
- **Network Health**: Automated health indicators and recommendations

### 💡 Employee Engagement Analytics
- **Engagement Metrics**: Comprehensive scoring including satisfaction, communication, collaboration
- **Burnout Risk Analysis**: Early identification of employees at risk
- **Retention Prediction**: Likelihood modeling based on engagement factors
- **Influence Mapping**: Identify high-potential organizational influencers
- **Department Engagement**: Comparative analysis across organizational units

### 🎯 Influence Analysis
- **Network-Engagement Correlation**: Understanding how engagement relates to network position
- **Organizational Influencers**: Ranking employees by combined network and engagement metrics
- **Leadership Development**: Identify emerging leaders and influential contributors

### 🔬 Enhanced ERGM Simulation
- **Organizational Context**: Department homophily, hierarchy effects, performance similarity
- **Engagement Factors**: Connection probability based on engagement similarity and collaboration index
- **Real-time Parameter Adjustment**: Watch networks evolve with engagement-driven dynamics
- **Scenario Testing**: Model impact of engagement initiatives on network formation

## 📦 Installation & Setup

### Quick Start (Recommended)

1. **Clone the repository:**
```bash
git clone https://github.com/reneedaos/network-health-demo.git
cd network-health-demo
```

2. **Install as a package (recommended):**
```bash
pip install -e .
```

3. **Run the application:**
```bash
# Option 1: Using the package entry point
ona-dashboard

# Option 2: Direct execution (recommended)
streamlit run app.py

# Option 3: Module execution
python -m src.ona_dashboard
```

### Alternative Installation

For minimal dependencies installation:
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Development Setup

For development with additional tools:
```bash
pip install -e ".[dev]"  # Includes pytest, black, flake8, mypy
```

## 🎯 Usage Guide

### Navigation

The dashboard includes nine main views accessible via the sidebar:

1. **📊 Executive Summary**: High-level organizational insights and key metrics
2. **🌐 Network Visualization**: Interactive network exploration with engagement filters
3. **⭐ Centrality Analysis**: Identify influential employees and bridge-builders  
4. **🏢 Department Analysis**: Cross-departmental connectivity patterns
5. **⚠️ Talent Risk**: Assess flight risk and identify key talent
6. **🏥 Network Health**: Organizational health indicators and recommendations
7. **💡 Employee Engagement**: Comprehensive engagement analysis and risk assessment
8. **🎯 Influence Analysis**: Network-engagement correlation and leadership identification
9. **🔬 Simulation**: Enhanced ERGM modeling with engagement factors

### Interactive Features

#### Network Visualization
- **Layout Selection**: Choose from spring, circular, shell, random, or hierarchical layouts
- **Advanced Filtering**: Filter by department, role, engagement level (high/low), burnout risk, or high influence
- **Enhanced Visual Controls**: Color by department, performance, tenure, connections, engagement, satisfaction, burnout risk, influence potential, or retention likelihood
- **Interactive Elements**: Zoom, pan, hover for detailed engagement metrics, clickable nodes
- **Engagement Symbols**: Special indicators for burnout risk (triangle), high influence (star), and management roles (diamond)

#### Enhanced ONA Parameters (Simulation)
- **Same Department Bonus**: Tendency for colleagues to connect (0.0-2.0)
- **Hierarchy Effect**: Manager connectivity preference (0.0-2.0) 
- **Performance Similarity**: Performance-based homophily (0.0-1.0)
- **Tenure Similarity**: Experience-based connections (0.0-1.0)
- **Engagement Similarity**: Engagement-based connection preference (0.0-1.0)
- **Collaboration Boost**: Collaborative employees connect more (0.0-1.0)
- **Communication Affinity**: High communicators form more connections (0.0-1.0)
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

## 🛠️ Technical Stack & Architecture

### Modular Package Structure

```
src/ona_dashboard/
├── models/                     # Data models
│   └── employee.py            # Employee class with engagement metrics
├── utils/                     # Utility functions
│   ├── generators.py          # Data generation utilities
│   └── metrics.py             # ONA and engagement calculations
├── visualizations/            # Visualization components
│   ├── network_plots.py       # Interactive network visualization
│   └── analysis_plots.py      # Analysis charts and dashboards
├── simulation/                # ERGM simulation engine
│   └── ergm_simulator.py      # Organizational network simulation
└── config/                    # Configuration management
    └── settings.py            # Centralized app settings
```

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

### Architecture Benefits

#### **🔧 Modularity**
- **Separation of Concerns**: Each module has a specific responsibility
- **Reusable Components**: Visualization and analysis functions can be used independently
- **Clean Dependencies**: Import relationships are explicit and manageable

#### **📦 Portability**
- **Package Installation**: Can be installed via pip as `ona-dashboard`
- **Clean Imports**: All imports use relative paths within the package
- **Configuration Management**: Centralized settings in `DashboardConfig`

#### **🧪 Maintainability**
- **Single Responsibility**: Each file has a focused purpose
- **Easier Testing**: Individual components can be tested in isolation
- **Code Organization**: Related functionality is grouped together

#### **⚡ Performance**
- **Lazy Loading**: Components are loaded only when needed
- **Optimized Imports**: Reduced memory footprint and faster startup
- **Modular Caching**: Better Streamlit session state management

### Core Technologies
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

## 🚀 Development & Deployment

### Development Workflow

```bash
# Clone and setup development environment
git clone https://github.com/reneedaos/network-health-demo.git
cd network-health-demo
pip install -e ".[dev]"

# Run tests
pytest

# Code formatting
black src/

# Type checking
mypy src/
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -e .

EXPOSE 8501

CMD ["ona-dashboard"]
```

### Production Deployment

The modular architecture supports various deployment options:

- **Streamlit Cloud**: Deploy directly with `streamlit run app.py`
- **Docker**: Containerize with the provided structure
- **Kubernetes**: Scale with the package-based architecture
- **Heroku/Railway**: Deploy as a Python package

### Extending the Application

The modular structure makes it easy to add new features:

1. **New Visualizations**: Add to `src/ona_dashboard/visualizations/`
2. **Additional Metrics**: Extend `src/ona_dashboard/utils/metrics.py`
3. **Custom Models**: Add to `src/ona_dashboard/models/`
4. **Different Simulations**: Enhance `src/ona_dashboard/simulation/`

## 📊 Architecture Comparison

| Feature | Before Refactoring | After Refactoring (`app.py`) |
|---------|--------------------|-----------------------------|
| **Lines of Code** | 1,600+ (single file) | ~300 per module (8 modules) |
| **Maintainability** | Difficult | Excellent |
| **Testability** | Limited | Full unit testing |
| **Reusability** | None | High |
| **Package Installation** | No | Yes (`pip install -e .`) |
| **Configuration** | Hardcoded | Centralized in `DashboardConfig` |
| **Import Structure** | Monolithic | Clean, modular imports |
| **Development** | Single large file | Focused, maintainable modules |
| **Architecture** | All-in-one file | Modular package structure |

## 📄 License

This project is for educational and demonstration purposes. See individual package licenses for dependencies.

---

### 🔗 Quick Links

- **[Refactoring Guide](REFACTORING_GUIDE.md)**: Detailed explanation of the modular architecture  
- **[Main Application](app.py)**: Modular implementation with clean architecture
- **[Package Source](src/ona_dashboard/)**: Modular package structure
- **[Setup Configuration](setup.py)**: Package installation and development setup