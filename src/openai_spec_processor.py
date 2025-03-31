#!/usr/bin/env python3
"""
OpenAI API Specification Processor for RAG

This script fetches the OpenAI API specification YAML file, processes it,
uploads it to Azure Blob Storage, and indexes it with Azure AI Search
for use in Retrieval Augmented Generation (RAG) applications.
"""

import os
import json
import yaml
import requests
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from typing import Dict, List, Any, Tuple

# Azure Storage Blob imports
from azure.storage.blob import BlobServiceClient, ContentSettings

# Azure AI Search imports
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    SearchField,
    SemanticConfiguration,
    SemanticField,
    SemanticSettings,
    VectorSearch,
    VectorSearchAlgorithmConfiguration,
)
from azure.core.credentials import AzureKeyCredential

# Load environment variables
load_dotenv()

# Constants
OPENAI_SPEC_URL = "https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml"
LOCAL_SPEC_PATH = Path("data/openai_spec.yaml")
PROCESSED_DOCS_PATH = Path("data/processed_docs")

# Azure Storage settings
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
AZURE_STORAGE_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "openai-api-spec")

# Azure AI Search settings
AZURE_SEARCH_SERVICE_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "openai-api-spec-index")


def fetch_openai_spec() -> Dict[str, Any]:
    """
    Fetch the OpenAI API specification from GitHub and save it locally.
    
    Returns:
        Dict[str, Any]: The parsed YAML content
    """
    # Create the data directory if it doesn't exist
    LOCAL_SPEC_PATH.parent.mkdir(exist_ok=True)
    
    print(f"Fetching OpenAI API specification from {OPENAI_SPEC_URL}")
    response = requests.get(OPENAI_SPEC_URL)
    response.raise_for_status()
    
    # Save the raw YAML content
    with open(LOCAL_SPEC_PATH, "w") as f:
        f.write(response.text)
    
    # Parse and return the YAML content
    return yaml.safe_load(response.text)


def process_openai_spec(spec_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Process the OpenAI API specification into searchable documents.
    
    Args:
        spec_data (Dict[str, Any]): The parsed YAML specification
        
    Returns:
        List[Dict[str, Any]]: A list of documents ready for indexing
    """
    documents = []
    
    # Basic info about the API
    api_info = {
        "id": "api_info",
        "title": spec_data.get("info", {}).get("title", "OpenAI API"),
        "description": spec_data.get("info", {}).get("description", ""),
        "version": spec_data.get("info", {}).get("version", ""),
        "type": "api_info",
        "content": json.dumps(spec_data.get("info", {}))
    }
    documents.append(api_info)
    
    # Process each endpoint
    for path, path_data in spec_data.get("paths", {}).items():
        for method, operation in path_data.items():
            if method in ["get", "post", "put", "delete", "patch"]:
                endpoint_doc = {
                    "id": f"{path.replace('/', '_')}_{method}",
                    "title": operation.get("summary", f"{method.upper()} {path}"),
                    "description": operation.get("description", ""),
                    "path": path,
                    "method": method,
                    "type": "endpoint",
                    "operationId": operation.get("operationId", ""),
                    "content": json.dumps({
                        "path": path,
                        "method": method,
                        "summary": operation.get("summary", ""),
                        "description": operation.get("description", ""),
                        "parameters": operation.get("parameters", []),
                        "requestBody": operation.get("requestBody", {}),
                        "responses": operation.get("responses", {})
                    })
                }
                documents.append(endpoint_doc)
    
    # Process schemas (data models)
    for schema_name, schema in spec_data.get("components", {}).get("schemas", {}).items():
        schema_doc = {
            "id": f"schema_{schema_name}",
            "title": f"Schema: {schema_name}",
            "description": schema.get("description", ""),
            "type": "schema",
            "content": json.dumps({
                "name": schema_name,
                "description": schema.get("description", ""),
                "properties": schema.get("properties", {}),
                "required": schema.get("required", []),
                "type": schema.get("type", "")
            })
        }
        documents.append(schema_doc)
    
    # Create directory for processed documents if needed
    PROCESSED_DOCS_PATH.mkdir(exist_ok=True)
    
    # Save processed documents to disk
    with open(PROCESSED_DOCS_PATH / "processed_documents.json", "w") as f:
        json.dump(documents, f, indent=2)
    
    print(f"Processed {len(documents)} documents from the OpenAI API specification")
    return documents


def upload_to_blob_storage(documents: List[Dict[str, Any]]) -> bool:
    """
    Upload the processed documents to Azure Blob Storage.
    
    Args:
        documents (List[Dict[str, Any]]): The processed documents
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create a blob service client
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        
        # Create the container if it doesn't exist
        try:
            container_client = blob_service_client.get_container_client(AZURE_STORAGE_CONTAINER_NAME)
            container_client.get_container_properties()  # Will raise if container doesn't exist
        except Exception:
            container_client = blob_service_client.create_container(AZURE_STORAGE_CONTAINER_NAME)
            print(f"Created container: {AZURE_STORAGE_CONTAINER_NAME}")
        
        # Upload each document as a separate JSON blob
        for doc in tqdm(documents, desc="Uploading documents to Blob Storage"):
            blob_name = f"{doc['id']}.json"
            blob_client = container_client.get_blob_client(blob_name)
            
            # Convert document to JSON string
            json_content = json.dumps(doc)
            
            # Upload the document with content type set to application/json
            blob_client.upload_blob(
                json_content,
                overwrite=True,
                content_settings=ContentSettings(content_type="application/json")
            )
        
        # Upload the full collection as a single file
        collection_blob_client = container_client.get_blob_client("full_collection.json")
        collection_blob_client.upload_blob(
            json.dumps(documents),
            overwrite=True,
            content_settings=ContentSettings(content_type="application/json")
        )
        
        print(f"Successfully uploaded {len(documents)} documents to Azure Blob Storage")
        return True
    
    except Exception as e:
        print(f"Error uploading to Blob Storage: {e}")
        return False


def create_search_index() -> Tuple[bool, str]:
    """
    Create an Azure AI Search index for the OpenAI API specification.
    
    Returns:
        Tuple[bool, str]: (Success status, Error message if any)
    """
    try:
        # Create a search index client
        credential = AzureKeyCredential(AZURE_SEARCH_API_KEY)
        index_client = SearchIndexClient(
            endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
            credential=credential
        )
        
        # Define the index fields
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="title", type=SearchFieldDataType.String, 
                          searchable=True, retrievable=True),
            SearchableField(name="description", type=SearchFieldDataType.String, 
                          searchable=True, retrievable=True),
            SimpleField(name="type", type=SearchFieldDataType.String, 
                      filterable=True, retrievable=True),
            SimpleField(name="path", type=SearchFieldDataType.String, 
                      filterable=True, retrievable=True),
            SimpleField(name="method", type=SearchFieldDataType.String, 
                      filterable=True, retrievable=True),
            SimpleField(name="operationId", type=SearchFieldDataType.String, 
                      filterable=True, retrievable=True),
            SearchableField(name="content", type=SearchFieldDataType.String, 
                          searchable=True, retrievable=True)
        ]
        
        # Create semantic settings for the index
        semantic_config = SemanticConfiguration(
            name="openai-spec-semantic-config",
            prioritized_fields=SemanticField(
                title_field=SemanticField(field_name="title"),
                content_fields=[SemanticField(field_name="content"), 
                                SemanticField(field_name="description")]
            )
        )
        
        semantic_settings = SemanticSettings(
            configurations=[semantic_config],
            default_configuration="openai-spec-semantic-config"
        )
        
        # Define the search index
        index = SearchIndex(
            name=AZURE_SEARCH_INDEX_NAME,
            fields=fields,
            semantic_settings=semantic_settings
        )
        
        # Create the index
        result = index_client.create_or_update_index(index)
        print(f"Index {result.name} created or updated")
        
        return True, ""
    
    except Exception as e:
        error_message = f"Error creating search index: {str(e)}"
        print(error_message)
        return False, error_message


def run_test_query() -> None:
    """
    Run a test query against the Azure AI Search index.
    """
    try:
        # Create a search client
        credential = AzureKeyCredential(AZURE_SEARCH_API_KEY)
        search_client = SearchClient(
            endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
            index_name=AZURE_SEARCH_INDEX_NAME,
            credential=credential
        )
        
        # Run a test query
        test_query = "chat completions api"
        results = search_client.search(
            search_text=test_query,
            select="id,title,description,type",
            top=5
        )
        
        print(f"\nTest query: '{test_query}'")
        print("Results:")
        
        for result in results:
            print(f"- {result['title']} ({result['type']})")
            print(f"  {result['description'][:100]}...")
            print()
            
    except Exception as e:
        print(f"Error running test query: {e}")


def main():
    """Main execution function."""
    print("Starting OpenAI API Specification Processor")
    
    # Validate environment variables
    missing_vars = []
    if not AZURE_STORAGE_CONNECTION_STRING:
        missing_vars.append("AZURE_STORAGE_CONNECTION_STRING")
    if not AZURE_SEARCH_SERVICE_ENDPOINT:
        missing_vars.append("AZURE_SEARCH_SERVICE_ENDPOINT")
    if not AZURE_SEARCH_API_KEY:
        missing_vars.append("AZURE_SEARCH_API_KEY")
    
    if missing_vars:
        print(f"Error: Missing environment variables: {', '.join(missing_vars)}")
        print("Please update your .env file with the required values")
        return
    
    # Fetch and process the OpenAI API specification
    spec_data = fetch_openai_spec()
    documents = process_openai_spec(spec_data)
    
    # Upload to Azure Blob Storage
    upload_success = upload_to_blob_storage(documents)
    if not upload_success:
        print("Failed to upload documents to Azure Blob Storage")
        return
    
    # Create Azure AI Search index
    index_success, error_message = create_search_index()
    if not index_success:
        print(f"Failed to create Azure AI Search index: {error_message}")
        return
    
    print("\nSetup complete! You can now:")
    print("1. Create an Azure AI Search indexer to process the documents from Blob Storage")
    print("2. Configure your RAG application to use the Azure AI Search index")
    print("3. Run a test query to verify the setup\n")
    
    # Optional: Run a test query
    run_test_query()


if __name__ == "__main__":
    main()