# AI Agent Framework

A flexible and modular framework for building AI agents with different LLM providers, memory systems, and tool integrations.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Architecture](#architecture)
- [Extending the Framework](#extending-the-framework)
- [Development Guidelines](#development-guidelines)

## Features

- **Multiple LLM Providers**: Support for Ollama, Groq, and Gemini
- **Model Context Protocol (MCP)**: Integration with MCP server for context management
- **Configurable Memory Systems**: Choose between Chroma, Qdrant, or FAISS vector stores
- **Various Embedding Models**: Options for sentence-transformers, OpenAI, and Hugging Face
- **Tool Integrations**: Built-in search tools with extensible architecture
- **Interactive UI**: Streamlit-based interface for easy interaction and configuration
- **Agent Workflows**: Structured agent workflows with tool execution capabilities

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install core dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install vector store specific dependencies** (choose one based on your needs):
   ```bash
   # For Chroma (default)
   pip install -r requirements-chroma.txt
   
   # For Qdrant
   pip install -r requirements-qdrant.txt
   
   # For FAISS
   pip install -r requirements-faiss.txt
   
   # For MCP integration
   pip install -r requirements-mcp.txt
   ```

5. **Set up environment variables**:
   ```bash
   cp .env.example .env
   ```
   Edit the `.env` file with your API keys and configuration.

## Configuration

### Environment Variables

The framework uses the following environment variables (which can also be configured through the UI):

| Variable | Description | Default |
|----------|-------------|---------|
| LLM_PROVIDER | LLM provider (ollama, groq, gemini) | ollama |
| LLM_MODEL | Model name for the selected provider | gemma3:4b |
| GROQ_API_KEY | API key for Groq | - |
| GEMINI_API_KEY | API key for Google Gemini | - |
| VECTOR_STORE | Vector store type (chroma, qdrant, faiss) | chroma |
| EMBEDDING_MODEL | Embedding model type (sentence-transformers, openai, huggingface) | sentence-transformers |
| EMBEDDING_MODEL_NAME | Name of the embedding model | all-MiniLM-L6-v2 |
| CHROMA_PERSIST_DIR | Directory for Chroma database | ./chroma_db |
| QDRANT_URL | URL for Qdrant server | - |
| QDRANT_API_KEY | API key for Qdrant | - |
| OPENAI_API_KEY | API key for OpenAI embeddings | - |
| HF_API_KEY | API key for HuggingFace embeddings | - |
| MCP_SERVER_URL | URL for the MCP server | http://localhost:8000 |
| MCP_API_KEY | API key for MCP authentication | - |

### UI Configuration

The application provides a user-friendly interface to configure:
- LLM provider and model
- Vector store settings
- Embedding model selection
- API keys and other provider-specific settings

## Usage

1. **Start the Streamlit application**:
   
   There are multiple ways to run the app while maintaining correct imports:

   ```bash
   # Option 1: Using the -m flag (recommended for proper imports)
   streamlit run -m src.app

   # Option 2: From the project root, specify the full path
   streamlit run src/app.py

   # Option 3: Setting PYTHONPATH before running
   # On Windows
   set PYTHONPATH=C:\Users\Devasy\OneDrive\Desktop\AGI
   streamlit run src/app.py
   
   # On Linux/macOS
   export PYTHONPATH=/path/to/AGI
   streamlit run src/app.py
   ```

2. **Configure your environment** using the sidebar controls.

3. **Interact with the agent** by sending messages and questions.

4. **Explore the notebooks** for examples of how to use the framework programmatically.

## Architecture

The framework follows a modular architecture:

### Core Components

- **LLM Providers** ([src/llm/](src/llm/)): Implementations for different LLM services
  - [llm_interface.py](src/llm/llm_interface.py): Base interface
  - [ollama_llm.py](src/llm/ollama_llm.py): Ollama implementation
  - [groq_llm.py](src/llm/groq_llm.py): Groq implementation
  - [gemini_llm.py](src/llm/gemini_llm.py): Google Gemini implementation
  - [llm_factory.py](src/llm/llm_factory.py): Factory to create appropriate LLM instance

- **Memory Systems** ([src/memory/](src/memory/)): Vector store implementations
  - [memory_interface.py](src/memory/memory_interface.py): Base interface
  - [simple_memory.py](src/memory/simple_memory.py): Simple implementation

- **Tools** ([src/tools/](src/tools/)): Tool implementations
  - [base_tool.py](src/tools/base_tool.py): Base tool class
  - [search_tools.py](src/tools/search_tools.py): Search implementations

- **Agents** ([src/agents/](src/agents/)): Agent implementations
  - [models.py](src/agents/models.py): Data models for agents
  - [workflow.py](src/agents/workflow.py): Agent workflow definitions

- **Configuration** ([src/config/](src/config/)): Configuration management
  - [llm_config.py](src/config/llm_config.py): LLM configuration
  - [memory_config.py](src/config/memory_config.py): Memory configuration

- **Utils** ([src/utils/](src/utils/)): Utility functions
  - [ui_helper.py](src/utils/ui_helper.py): UI components and helpers
  - [env_config.py](src/utils/env_config.py): Environment configuration

## Extending the Framework

### Adding a New LLM Provider

1. Create a new file in [src/llm/](src/llm/) (e.g., `new_provider_llm.py`)
2. Implement the LLM interface defined in [llm_interface.py](src/llm/llm_interface.py)
3. Update the [llm_factory.py](src/llm/llm_factory.py) to include your new provider
4. Update the UI options in [env_config.py](src/utils/env_config.py)

### Adding a New Tool

1. Create a new file in [src/tools/](src/tools/) or extend an existing file
2. Inherit from the base tool class in [base_tool.py](src/tools/base_tool.py)
3. Implement the required methods
4. Register the tool in your agent workflow

### Adding a New Vector Store

1. Add required dependencies to a new requirements file (e.g., `requirements-new-store.txt`)
2. Create a new implementation in [src/memory/](src/memory/)
3. Update the [memory_config.py](src/config/memory_config.py) to include your new option
4. Update the UI options in [ui_helper.py](src/utils/ui_helper.py)

## Development Guidelines

1. **Follow the existing architecture**: Maintain separation of concerns between components.

2. **Use interfaces**: Implement the appropriate interfaces when adding new features.

3. **Testing**: Add tests for new functionality.

4. **Environment Variables**: Update `.env.example` when adding new configuration options.

5. **Documentation**: Document new features in code comments and update the README as needed.

6. **Dependencies**: Keep dependencies separated by functionality in the requirements files.

7. **Use Pydantic Models**: Follow the pattern of using Pydantic models for structured data.

8. **Error Handling**: Implement proper error handling and provide informative error messages.

## Troubleshooting

### Import Errors

If you encounter errors like `ImportError: attempted relative import beyond top-level package`, try one of these solutions:

1. **Run the application as a module**:
   ```bash
   python -m src.app
   ```
   
2. **Use absolute imports** in your code instead of relative imports:
   ```python
   # Instead of: from ..llm.llm_factory import LLMFactory
   from src.llm.llm_factory import LLMFactory
   ```

3. **Set the PYTHONPATH** environment variable before running the app:
   ```bash
   # On Windows
   set PYTHONPATH=C:\Users\Devasy\OneDrive\Desktop\AGI
   
   # On Linux/macOS
   export PYTHONPATH=/path/to/AGI
   ```

## License

[Add your license information here]

---

For questions or support, please [open an issue](link-to-issues) or contact [your contact information].
