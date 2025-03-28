# Technical Documentation: AI Agent Framework

## Table of Contents

- [Architecture Deep Dive](#architecture-deep-dive)
- [Component Overview](#component-overview)
  - [Application Entry Point](#1-application-entry-point-srcapppy)
  - [Agent Orchestration](#2-agent-orchestration-srcagentscrew_workflowpy)
  - [Agent Definitions](#3-agent-definitions-srcagentscrew_agentspy)
  - [LLM Integration](#4-llm-integration-srcllmcrew_llmpy)
  - [Tool System](#5-tool-system-srctoolscrew_toolspy)
- [Key Workflows](#key-workflows)
  - [User Query Processing Flow](#1-user-query-processing-flow)
  - [LLM Provider Selection](#2-llm-provider-selection)
  - [Configuration Management](#3-configuration-management)
- [Alternative Implementation: LangGraph Workflow](#alternative-implementation-langgraph-workflow)
- [Implementation Details](#implementation-details)
  - [Data Models](#1-data-models-srcagentsmodelspy)
  - [UI Implementation](#2-ui-implementation-srcutilsui_helperpy)
  - [Environment Configuration](#3-environment-configuration-srcutilsenv_configpy)
- [Performance Considerations](#performance-considerations)
- [Security Considerations](#security-considerations)
- [Testing Strategy](#testing-strategy)
- [Deployment Considerations](#deployment-considerations)
- [Maintenance and Monitoring](#maintenance-and-monitoring)
- [Feature Expansion Planning](#feature-expansion-planning)
  - [GitHub Integration Implementation](#github-integration-implementation)
  - [Web & Content Integration Implementation](#web--content-integration-implementation)
  - [Code Execution & Data Exploration Implementation](#code-execution--data-exploration-implementation)
  - [Additional Integrations Implementation](#additional-integrations-implementation)

## Architecture Deep Dive

This document provides in-depth technical details about the AI Agent Framework architecture, code structure, and implementation details. While the initial implementation focuses on a Multi-Agent Search Assistant, the architecture is designed to be extensible for a wide range of capabilities as outlined in the roadmap.

## Component Overview

### 1. Application Entry Point (`src/app.py`)

The main entry point initializes the Streamlit UI, sets up the configuration, and handles the user interaction flow:

```python
# Initialize UI and environment
ui = StreamlitUI()
ui.initialize_session_state()
ui.setup_sidebar()

# Create workflow
workflow = CrewWorkflow()

# Handle user input
if question := st.chat_input("Ask your question"):
    # Add user message to chat
    ui.add_chat_message("user", question)
    
    # Process the query using CrewAI workflow
    result = workflow.process_query(
        query=question,
        chat_history=st.session_state.messages,
        lst_res=st.session_state.get('lst_res', [])
    )
```

### 2. Agent Orchestration (`src/agents/crew_workflow.py`)

The `CrewWorkflow` class orchestrates multiple specialized agents to process user queries:

```python
def process_query(self, query: str, chat_history: List[Dict[str, str]], lst_res: List) -> str:
    # Initialize agents
    researcher = self.agent_factory.create_research_agent()
    file_expert = self.agent_factory.create_file_expert()
    synthesizer = self.agent_factory.create_synthesizer_agent()
    
    # Create tasks with explicit dependencies
    research_task = Task(
        description=f"Research the following query:\n{query}\n\nContext from previous interactions:\n{context_str}",
        agent=researcher,
        expected_output="A detailed analysis of the query with information from online sources"
    )
    
    # Create and run the crew
    crew = Crew(
        agents=[researcher, file_expert, synthesizer],
        tasks=[research_task, file_analysis_task, synthesis_task],
        verbose=True,
        process="sequential"
    )
    
    # Execute tasks and get result
    result = crew.kickoff()
```

### 3. Agent Definitions (`src/agents/crew_agents.py`)

The `CrewAgentFactory` creates specialized agents with different capabilities:

- **Research Agent**: Focuses on finding information from online sources
- **File Expert**: Analyzes local files and documentation
- **Synthesizer Agent**: Combines all information into coherent answers

### 4. LLM Integration (`src/llm/crew_llm.py`)

The application supports multiple LLM providers through a unified interface:

```python
def create_llm():
    config = LLMConfig.get_config()
    provider = config["provider"]
    
    if provider == "ollama":
        return LLM(
            model=f"ollama/{model_name}",
            base_url="http://localhost:11434"
        )
    elif provider == "groq":
        return LLM(
            model=f"groq/{model_name}",
            api_key=config.get("groq_api_key")
        )
    elif provider == "gemini":
        return LLM(
            model=f"gemini/{model_name}",
            temperature=0.7,
            api_key=config.get("gemini_api_key")
        )
```

### 5. Tool System (`src/tools/crew_tools.py`)

Tools provide agents with specific capabilities:

- **SerperDevTool**: Web search using the Serper API
- **WebsiteSearchTool**: Extracting information from specific websites
- **DirectoryReadTool**: Reading local directory contents
- **FileReadTool**: Reading file contents
- **CodeDocsSearchTool**: Searching code documentation

## Key Workflows

### 1. User Query Processing Flow

1. User submits a query via the Streamlit interface
2. Query is passed to the `CrewWorkflow.process_query` method
3. Three tasks are created and assigned to specialized agents:
   - Research task (assigned to researcher agent)
   - File analysis task (assigned to file expert agent)
   - Synthesis task (assigned to synthesizer agent)
4. Tasks are executed sequentially with context sharing
5. Final synthesis result is returned to the UI and displayed to the user

### 2. LLM Provider Selection

1. User selects a provider (Ollama, Groq, or Gemini) via the UI
2. Selection is saved to session state
3. When an agent needs the LLM, it calls `create_llm()` which:
   - Reads configuration from session state and environment
   - Creates appropriate LLM instance based on provider
   - Returns consistent interface regardless of provider

### 3. Configuration Management

Configuration is managed in multiple layers:

1. **Default values** are defined in config classes
2. **.env file** overrides defaults
3. **Session state** (from UI) overrides .env values

The `LLMConfig` class manages this hierarchy:

```python
@classmethod
def get_provider(cls):
    # First check session state, then environment variables, then default
    if 'env_vars' in st.session_state and 'LLM_PROVIDER' in st.session_state.env_vars:
        return st.session_state.env_vars['LLM_PROVIDER']
    return os.getenv("LLM_PROVIDER", "ollama")
```

## Alternative Implementation: LangGraph Workflow

The application includes an alternative workflow implementation using LangGraph (`src/agents/workflow.py`):

```python
def create_graph(self) -> StateGraph:
    workflow = StateGraph(State)
    
    # Agent 1
    workflow.add_node("Agent1", action=self.node_agent)
    workflow.set_entry_point("Agent1")
    workflow.add_node("tool_browser", action=self.node_tool)
    workflow.add_node("final_answer", action=self.node_tool)
    workflow.add_edge(start_key="tool_browser", end_key="Agent1")
    workflow.add_conditional_edges(source="Agent1", path=self.conditional_edges)
    
    # Agent 2
    workflow.add_node("Agent2", action=self.node_agent_2)
    workflow.add_node("tool_wikipedia", action=self.node_tool)
    workflow.add_edge(start_key="tool_wikipedia", end_key="Agent2")
    workflow.add_conditional_edges(source="Agent2", path=self.conditional_edges)
```

This implementation:
- Uses a graph-based approach for more flexible agent interaction
- Supports conditional routing between agents
- Allows for more complex decision logic

## Implementation Details

### 1. Data Models (`src/agents/models.py`)

The application uses Pydantic models for data validation:

```python
class AgentRes(BaseModel):
    tool_name: str
    tool_input: dict
    tool_output: Optional[str] = None
    
    @classmethod
    def from_llm(cls, res: dict):
        # Parse LLM response into structured format
```

### 2. UI Implementation (`src/utils/ui_helper.py`)

The StreamlitUI class manages the interface:

```python
class StreamlitUI:
    @staticmethod
    def show_chat_messages():
        """Display all chat messages from the session state"""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    @staticmethod
    def add_chat_message(role: str, content: str, is_progress: bool = False):
        """Add a message to the chat history and display it"""
```

### 3. Environment Configuration (`src/utils/env_config.py`)

The EnvConfig class manages environment variables and UI configuration:

```python
class EnvConfig:
    @staticmethod
    def setup_env_ui():
        """Create UI elements for environment configuration"""
        with st.sidebar:
            st.markdown("### Environment Configuration")
            
            current_provider = st.session_state.env_vars.get("LLM_PROVIDER", "ollama")
            
            # LLM Provider Selection
            provider = st.selectbox(
                "LLM Provider",
                options=["ollama", "groq", "gemini"],
                index=["ollama", "groq", "gemini"].index(current_provider)
            )
```

## Performance Considerations

1. **Sequential vs. Parallel Execution**
   - Current implementation uses sequential task execution
   - Future improvement: Implement parallel execution for independent tasks

2. **Response Time**
   - Web search operations are typically the slowest component
   - Consider caching frequent searches to improve response time

3. **Memory Usage**
   - Session state can grow large with extensive chat history
   - Consider implementing pagination or history summarization

## Security Considerations

1. **API Key Management**
   - API keys are stored in session state and environment variables
   - Consider implementing secure credential storage for production

2. **User Input Validation**
   - All user input should be validated before processing
   - Implement rate limiting to prevent abuse

3. **External API Security**
   - Web search tools connect to external APIs
   - Implement proper error handling for API failures

## Testing Strategy

1. **Unit Testing**
   - Test individual components in isolation
   - Mock external dependencies

2. **Integration Testing**
   - Test complete query processing workflow
   - Verify correct agent interaction

3. **UI Testing**
   - Test Streamlit interface components
   - Verify correct state management

## Deployment Considerations

1. **Environment Setup**
   - Python 3.9+ required
   - All dependencies listed in requirements.txt

2. **API Key Configuration**
   - Configure .env file with necessary API keys
   - Set up provider-specific configuration

3. **Hosting Options**
   - Streamlit Cloud for simple deployment
   - Docker containers for more complex setups

## Maintenance and Monitoring

1. **Logging**
   - Implement comprehensive logging for debugging
   - Track query performance metrics

2. **Error Handling**
   - Implement graceful failure modes
   - Provide useful error messages to users

3. **Performance Monitoring**
   - Track query response times
   - Monitor external API usage

## Feature Expansion Planning

This section outlines the technical considerations for implementing the roadmap features.

### GitHub Integration Implementation

To implement GitHub integration, we'll need to:

1. **API Integration Layer**:
   - Create a `github_tools.py` module in the tools directory
   - Implement GitHub API client using the PyGithub library
   - Define tool classes for each capability (repo search, issue analysis, etc.)

2. **Authentication Handling**:
   - Add GitHub API key/token storage in configuration
   - Implement OAuth flow for more complex authentication scenarios

3. **Specialized Agents**:
   - Create a GitHub Expert agent in `crew_agents.py`
   - Train agent prompts for GitHub-specific tasks

4. **UI Components**:
   - Add GitHub configuration section to sidebar
   - Create specialized views for repository data

### Web & Content Integration Implementation

For web and content integration:

1. **Enhanced Search Tools**:
   - Expand `crew_tools.py` with specialized web scrapers
   - Implement Medium blog parser with premium content handling
   - Create NewsAPI integration for current events

2. **RAG Memory System**:
   - Implement `dynamic_memory.py` with vector database integration
   - Create context expansion mechanisms based on search results
   - Implement dspy or CoT techniques for search refinement

3. **Document Processing**:
   - Add summarization capabilities to the tools system
   - Implement specialized parsers for different content types

### Code Execution & Data Exploration Implementation

For code execution and data exploration:

1. **Secure Code Execution**:
   - Implement sandboxed execution environment
   - Create code validation and security checks

2. **Database Connectors**:
   - Add SQL database connectors with query generation
   - Implement NoSQL database support

3. **Visualization Tools**:
   - Create plotting and charting capabilities
   - Implement data transformation utilities

### Additional Integrations Implementation

For Spotify and CodeForces integration:

1. **API Clients**:
   - Create `spotify_tools.py` and `codeforces_tools.py` modules
   - Implement authentication flows for each service

2. **Specialized Data Processing**:
   - Add music analysis capabilities for Spotify
   - Implement contest performance analysis for CodeForces

3. **Agent Specialization**:
   - Create domain-specific prompts for these services
   - Implement specialized agents for each domain