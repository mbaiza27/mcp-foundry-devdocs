# MCP Server

*coming soon*

# OpenAI API Specification for RAG

A Python utility to fetch, process, and index the OpenAI API specification for use in Retrieval Augmented Generation (RAG) applications.

## Overview

This tool:
- Fetches the OpenAI API specification from GitHub
- Processes it into searchable chunks
- Uploads the chunks to Azure Blob Storage
- Creates and configures an Azure AI Search index

## Setup

1. Create a virtual environment:
   ```bash
   uv venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```

3. Configure environment variables in `.env`:
   ```
   # Azure Storage Account settings
   AZURE_STORAGE_CONNECTION_STRING=your_storage_connection_string
   AZURE_STORAGE_ACCOUNT_NAME=your_storage_account_name
   AZURE_STORAGE_CONTAINER_NAME=openai-api-spec

   # Azure AI Search settings
   AZURE_SEARCH_SERVICE_ENDPOINT=https://your-service-name.search.windows.net
   AZURE_SEARCH_API_KEY=your_search_api_key
   AZURE_SEARCH_INDEX_NAME=openai-api-spec-index
   ```

## Usage

Run the processor:
```bash
python src/openai_spec_processor.py
```

## Integration with RAG

After processing, you can use the Azure AI Search index as a knowledge source for your RAG applications, providing models with up-to-date context about the OpenAI API.