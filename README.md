# MCP Foundry DevDocs

This repository contains tools for document processing and AI agent integration:

1. **Azure AI Agent MCP Server**: A Model Context Protocol (MCP) server that integrates with Azure AI Agent Service
2. **OpenAI API Specification Processor**: A utility to prepare OpenAI's API documentation for Retrieval Augmented Generation (RAG)

## Azure AI Agent MCP Server

An MCP server that connects to Azure AI Agent Service, allowing VS Code extensions and other MCP clients to interact with Azure AI Agents.

### Features

- Connect to any agent in your Azure AI project
- Query a default agent with a simpler interface
- List all available agents
- Format responses with proper Markdown including citations

### Setup

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables in `.env`:
   ```
   # Required for Azure AI Agent MCP Server
   PROJECT_CONNECTION_STRING=your_project_connection_string
   DEFAULT_AGENT_ID=your_default_agent_id
   ```

4. The connection string format should be:
   ```
   [region].api.azureml.ms;[tenant_id];[project_name];[workspace_name]
   ```

### Running the MCP Server

Start the server:
```bash
python -m azure_agent_mcp_server
```

### Configuring in VS Code

Add the server to your VS Code settings.json:

```json
"mcp": {
  "servers": {
    "azure-agent": {
      "command": "/path/to/your/.venv/bin/python",
      "args": [
        "-m",
        "azure_agent_mcp_server"
      ],
      "cwd": "/path/to/your/src",
      "env": {
        "PYTHONPATH": "/path/to/your/src",
        "PROJECT_CONNECTION_STRING": "your_connection_string",
        "DEFAULT_AGENT_ID": "your_default_agent_id"
      }
    }
  }
}
```

### Available Tools

The MCP server provides these tools:

- `connect_agent`: Connect to a specific Azure AI Agent using its ID
- `query_default_agent`: Query the default agent specified in your configuration
- `list_agents`: List all available agents in your Azure AI project

## OpenAI API Specification Processor

A Python utility to fetch, process, and index the OpenAI API specification for RAG applications.

### Overview

This tool:
- Fetches the OpenAI API specification from GitHub
- Processes it into searchable chunks
- Uploads the chunks to Azure Blob Storage
- Creates and configures an Azure AI Search index

### Setup

Configure environment variables in `.env` (in addition to the Azure AI Agent variables):
```
# Azure Storage Account settings
AZURE_STORAGE_CONNECTION_STRING=your_storage_connection_string
AZURE_STORAGE_ACCOUNT_NAME=your_storage_account_name
AZURE_STORAGE_CONTAINER_NAME=your_container_name

# Azure AI Search settings
AZURE_SEARCH_SERVICE_ENDPOINT=https://your-service-name.search.windows.net
AZURE_SEARCH_API_KEY=your_search_api_key
AZURE_SEARCH_INDEX_NAME=your_index_name

# Optional OpenAI API key for embeddings
OPENAI_API_KEY=your_openai_api_key
```

### Usage

Run the processor:
```bash
python src/openai_spec_processor.py
```

#### Command Line Options

- `--fetch-only`: Only fetch the API spec
- `--process-only`: Only process the API spec (no upload)
- `--upload-only`: Only upload processed documents
- `--index-only`: Only create/update the search index
- `--chunk-size`: Size of content chunks (default: 1000)
- `--chunk-overlap`: Overlap between chunks (default: 200)
- `--skip-embeddings`: Skip generating embeddings
- `--test-query`: Run a test query against the index

Example:
```bash
python src/openai_spec_processor.py --test-query "How do I use the completions API?"
```