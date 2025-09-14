# Agent-Eval: LLM Agent Creation and Evaluation Framework

A Python framework for creating, running, and evaluating LLM agents with quantifiable metrics and auto-discovery tool system.

## Development Setup

### Prerequisites
- Python 3.8+
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd agent-eval
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install in development mode**
   ```bash
   pip install -e .
   ```

## Running Tests

### All Tests
```bash
source venv/bin/activate
python -m pytest tests/ -v
```

### Specific Test File
```bash
python -m pytest tests/test_tool_discovery.py -v
python -m pytest tests/test_user_feedback_tool.py -v
python -m pytest tests/test_web_search_tool.py -v
```

### Specific Test
```bash
python -m pytest tests/test_tool_discovery.py::TestToolDiscovery::test_discover_single_tool -v
```

### Test Coverage
```bash
python -m pytest tests/ --cov=agent_eval
```