# 🤖 Agent-Eval

## ✨ What is Agent-Eval?

Agent-Eval is a **Python framework for building LLM agents with tools**. Go from empty folder to a working agent with custom tools in minutes.

### 🚀 **What You Get**
- ⚡ **Quick setup** - `pip install` to working agent in minutes
- 🔧 **Simple tools** - Drop a Python class in `/tools`, it's automatically discovered
- 🌍 **Any model** - Gemini, OpenAI, Claude, or 100+ others via LiteLLM
- 📊 **Built-in metrics** - Token usage, timing, and conversation tracking
- 📜 **File-based prompts** - Version control your system prompts
- 🎨 **Beautiful CLI** - Interactive mode with colors and helpful commands

### 🎯 **Perfect For**
- **Rapid prototyping** - Test agent ideas quickly
- **Production development** - Scale from prototype to production
- **Multi-model experiments** - Easy switching between LLM providers
- **Team collaboration** - Git-friendly structure for shared development

---

## ⚡ Quick Start - Be Running in 60 Seconds

```bash
# 1. Install
pip install -e .

# 2. Set your API key (Gemini has a generous free tier, no credit card required!)
export GEMINI_API_KEY="your_key_here"

# 3. Chat with your agent instantly
agent-eval run "What are some great movies that came out this year?"

# 4. Start interactive mode
agent-eval chat
```

**That's it!** 🎉 You now have a fully functional LLM agent with tool support, conversation management, and evaluation metrics.

---

## 🔧 Tool Development - The Magic Happens Here

Creating tools is **ridiculously simple**. Just implement the `Tool` class anywhere in the `/tools` directory and the agent will automatically discover and use it:

```python
from agent_eval.tools import Tool

class WeatherTool(Tool):
    def get_schema(self):
        return {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }

    def execute(self, location: str):
        # Your implementation here
        return f"Weather in {location}: Sunny, 72°F"
```

**That's literally it!** 🤯

- ✅ No registration required
- ✅ Auto-discovery at runtime
- ✅ Automatic validation
- ✅ Built-in error handling
- ✅ Tool approval workflow

---

## 📜 System Prompt Management

Agent-Eval treats prompts as **first-class citizens** with proper version control and organization:

```
your-project/
├── prompts/
│   ├── default_system.txt        # Main system prompt
│   ├── customer_support.txt      # Domain-specific variants
│   ├── data_analyst.txt          # Role-specific prompts
│   └── llm_judge_evaluation.txt  # Custom evaluation criteria
```

**Features:**
- 🎯 **File-based prompts** - Easy to version control and collaborate
- 🔄 **Hot-swapping** - Test different prompts instantly
- 📊 **A/B testing** - Compare prompt performance systematically
- 🎨 **Template system** - Reusable prompt components

```bash
# Test different prompts instantly
agent-eval chat --system-prompt="prompts/customer_support.txt"

# Hot-swap during development
echo "You are a specialized data analyst..." > prompts/custom.txt
agent-eval chat --system-prompt="prompts/custom.txt"
```

---

## 🌍 Any LLM Provider - Your Choice

Agent-Eval supports **100+ models** through LiteLLM integration:

### 🆓 **Start Free with Gemini**
```bash
export GEMINI_API_KEY="your_key"
# Generous free tier: 15 RPM, 1M tokens/day
```

### 🚀 **Scale with Premium Providers**
```bash
# OpenAI
export OPENAI_API_KEY="your_key"
agent-eval chat --model="openai/gpt-4"

# Anthropic Claude
export ANTHROPIC_API_KEY="your_key"
agent-eval chat --model="anthropic/claude-3-sonnet-20240229"

# Or any other provider...
```

**Supported Providers:**
- **Gemini** (Free tier available!)
- **OpenAI** (GPT-4, GPT-3.5)
- **Anthropic** (Claude family)
- **Cohere, Hugging Face, Azure** and 100+ more

---

## 🎨 Beautiful CLI Experience

```bash
# Interactive mode with colors and emojis
$ agent-eval chat

🚀 Initializing agent...

╔══════════════════════════════════════╗
║           🤖 Agent-Eval              ║
║    LLM Agent Evaluation Framework    ║
╚══════════════════════════════════════╝

Model: gemini/gemini-2.5-flash
Tools: ask_user, search_web
Auto-approve: No

💬 Ready! Type '/help' for commands

You: What's the capital of France?
🤖 Agent: The capital of France is Paris.

📊 Tokens: 245 | Tools: 0 | Iterations: 1 | Time: 0.8s
```

### 🛠️ Built-in Commands
- `/help` - Show available commands
- `/tools` - List discovered tools
- `/metrics` - Show performance stats
- `/save` - Save conversation to file
- `/clear` - Reset conversation

---

## 📊 Built-in Metrics & Tracking

Every interaction is automatically tracked and monitored:

### 🔍 **Real-time Metrics**
- Token usage per query
- Tool execution count
- Response time tracking
- Error rate monitoring
- Iteration counting

### 💾 **Conversation Persistence**
```json
// Auto-saved to conversation_history/
{
  "conversation_history": [...],
  "metrics": {
    "total_tokens": 1250,
    "tool_calls": 3,
    "iterations": 2,
    "duration": 2.34
  },
  "model": "gemini/gemini-2.5-flash",
  "tools_available": ["search_web", "ask_user"]
}
```

---

## 🔮 What's Coming Next

### 🔄 The Agent Development Loop *(The Secret to Production-Ready Agents)*

The professional workflow that scales from prototype to bulletproof agents:

```
📝 Create Test Cases → 🚀 Run Evaluation → 📊 Review Results
     ⬆                                           ⬇
🔁 Add More Tests  ← ✅ Passes? → 🎯 Tweak System Prompt
```

**The Flow:**
1. **📝 Write test cases** in `golden_dataset/`
2. **🚀 Run evaluation**: `agent-eval evaluate --dataset golden_dataset/`
3. **📊 Analyze failures** in evaluation reports
4. **🎯 Refine system prompt** in `prompts/default_system.txt`
5. **🔁 Re-evaluate** until tests pass
6. **➕ Add edge cases** you discovered
7. **🔄 Repeat** → Build bulletproof agents

This iterative workflow is what separates toy demos from production-grade agents.

### 📊 Evaluation Pipeline *(Coming Soon)*

The missing piece for production agent evaluation:

```
your-project/
├── golden_dataset/           # Your test cases
│   ├── customer_support.json
│   └── data_analysis.json
├── evaluation_logs/          # Automated runs
│   └── run_2025_01_15_14_30/
│       ├── responses.json
│       ├── metrics.json
│       └── failures.log
└── llm_judge_reports/        # AI evaluation
    ├── accuracy_scores.json
    └── quality_analysis.md
```

**How it will work:**

1. **📝 Define Test Cases** - JSON files with input/expected output pairs
2. **🚀 Run Evaluation** - `agent-eval evaluate --dataset golden_dataset/`
3. **🤖 LLM-as-Judge** - Automatic quality scoring for subjective responses
4. **📈 Detailed Reports** - Pass/fail rates, performance trends, insights
5. **🔄 CI/CD Integration** - Automated testing in your deployment pipeline

**Enhanced Evaluation Features:**

🎯 **Custom LLM-as-Judge Prompts**
```
prompts/
├── llm_judge_evaluation.txt    # Your custom evaluation criteria
├── accuracy_rubric.txt         # Domain-specific scoring
└── safety_evaluation.txt       # Safety/compliance checks
```

📊 **Evaluation Dashboard**
- Pass/fail rates by test case category
- Prompt performance comparisons
- Regression detection across iterations
- Cost analysis per evaluation run

🔄 **CI/CD Integration**
```bash
# In your GitHub Actions
- name: Evaluate Agent
  run: agent-eval evaluate --dataset golden_dataset/ --fail-threshold 0.95
```

This will make Agent-Eval the **complete solution** for agent development and evaluation.

---

## 🚀 Advanced Usage

### 🎛️ **Custom Configuration**
```bash
# Override any setting
agent-eval chat \
  --model="openai/gpt-4" \
  --max-iterations=15 \
  --auto-approve \
  --system-prompt="You are a helpful coding assistant"
```


---

## 🏗️ Installation & Setup

### 📦 **Quick Install**
```bash
pip install -e .
```

### 🐍 **Development Setup**
```bash
# Clone and setup
git clone <repository-url>
cd agent-eval
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .

# Run tests
python test_agent.py  # Comprehensive agent tests
python -m pytest tests/ -v  # Tool-specific tests
```

### 🔑 **API Key Setup**
```bash
# Option 1: Environment variables
export GEMINI_API_KEY="your_key"
export OPENAI_API_KEY="your_key"

# Option 2: .env file
echo "GEMINI_API_KEY=your_key" > .env
```

---

## 🏛️ Architecture

Agent-Eval is built on solid foundations:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Beautiful CLI │────│  Agent Core      │────│  Tool Discovery │
│  • Colored UI   │    │  • Conversation  │    │  • Auto-detect  │
│  • Interactive  │    │  • Metrics       │    │  • Validation   │
│  • Intuitive    │    │  • Prompt Mgmt   │    │  • Execution    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                               │                         │
                    ┌──────────┴──────────┐             │
                    │   Prompt System     │             │
                    │  • File-based       │             │
                    │  • Version Control  │             │
                    │  • Hot-swapping     │             │
                    │  • A/B Testing      │             │
                    └─────────────────────┘             │
                               │                         │
                       ┌───────┴────────┐               │
                       │   LiteLLM      │───────────────┘
                       │  • 100+ Models │
                       │  • Unified API │
                       │  • Reliability │
                       └────────────────┘
```

**Core Components:**
- **🎨 CLI Layer** - Beautiful, intuitive developer experience
- **🧠 Agent Core** - Conversation management, metrics, error handling
- **📜 Prompt System** - File-based prompts with version control and A/B testing
- **🔧 Tool System** - Zero-config auto-discovery and execution
- **🌐 LiteLLM Integration** - Universal model provider support

---

## 🐛 **Found a Bug?**
[Open an issue](https://github.com/your-org/agent-eval/issues) with:
- Clear description
- Steps to reproduce
- Expected vs actual behavior
- Your environment details


---

## 🌟 Star Us!

If Agent-Eval makes your LLM agent development easier, **please star the repo**! ⭐

Every star helps us reach more developers who are struggling with complex agent frameworks.

---

<div align="center">

**Built with ❤️ for the LLM developer community**

</div>