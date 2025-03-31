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
import argparse
import logging
from datetime import datetime

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
    SemanticPrioritizedFields,
    VectorSearch,
    VectorSearchAlgorithmConfiguration,
    SemanticSearch,  # Using SemanticSearch instead of SemanticSettings
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


# Create necessary directories
Path("logs").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/openai_spec_processor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration defaults
DEFAULT_CONFIG = {
    "CONTAINER_NAME": "openai-api-spec",
    "INDEX_NAME": "openai-api-spec-index",
    "CHUNK_SIZE": 1000,
    "CHUNK_OVERLAP": 200,
    "EMBEDDING_MODEL": "text-embedding-3-small",
    "VECTOR_DIMENSIONS": 1536  # Dimensions for text-embedding-3-small
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process OpenAI API specification for RAG")
    parser.add_argument("--fetch-only", action="store_true", help="Only fetch the API spec")
    parser.add_argument("--process-only", action="store_true", help="Only process the API spec (no upload)")
    parser.add_argument("--upload-only", action="store_true", help="Only upload processed documents")
    parser.add_argument("--index-only", action="store_true", help="Only create/update the search index")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CONFIG["CHUNK_SIZE"], 
                        help="Size of content chunks in characters")
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CONFIG["CHUNK_OVERLAP"], 
                        help="Overlap between chunks in characters")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip generating embeddings")
    parser.add_argument("--container", type=str, default=DEFAULT_CONFIG["CONTAINER_NAME"], 
                        help="Azure Storage container name")
    parser.add_argument("--index", type=str, default=DEFAULT_CONFIG["INDEX_NAME"], 
                        help="Azure AI Search index name")
    parser.add_argument("--test-query", type=str, help="Run a test query against the index")
    return parser.parse_args()


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
    
    # Process tags - these categorize endpoints into groups
    for tag in spec_data.get("tags", []):
        tag_doc = {
            "id": f"tag_{tag['name'].lower().replace(' ', '_')}",
            "title": f"Tag: {tag['name']}",
            "description": tag.get("description", ""),
            "type": "tag",
            "content": json.dumps(tag)
        }
        documents.append(tag_doc)
    
    # Process each endpoint
    for path, path_data in spec_data.get("paths", {}).items():
        for method, operation in path_data.items():
            if method in ["get", "post", "put", "delete", "patch"]:
                # Extract the operation metadata (x-oaiMeta)
                operation_meta = operation.get("x-oaiMeta", {})
                
                # Get tags for this endpoint
                tags = operation.get("tags", [])
                
                endpoint_doc = {
                    "id": f"{path.replace('/', '_')}_{method}",
                    "title": operation.get("summary", f"{method.upper()} {path}"),
                    "description": operation.get("description", ""),
                    "path": path,
                    "method": method,
                    "type": "endpoint",
                    "operationId": operation.get("operationId", ""),
                    "tags": tags,  # Add tags
                    "content": json.dumps({
                        "path": path,
                        "method": method,
                        "summary": operation.get("summary", ""),
                        "description": operation.get("description", ""),
                        "parameters": operation.get("parameters", []),
                        "requestBody": operation.get("requestBody", {}),
                        "responses": operation.get("responses", {}),
                        "x-oaiMeta": operation_meta,  # Include extension metadata
                        "tags": tags
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


def create_search_index(index_name: str, include_vector: bool = True) -> Tuple[bool, str]:
    """
    Create an Azure AI Search index for the OpenAI API specification.
    
    Args:
        index_name (str): Name of the index to create/update
        include_vector (bool): Whether to include vector search capabilities
        
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
                          searchable=True, retrievable=True),
            SimpleField(name="parent_id", type=SearchFieldDataType.String,
                      filterable=True, retrievable=True),
            SimpleField(name="chunk_index", type=SearchFieldDataType.Int32,
                      filterable=True, retrievable=True, sortable=True)
        ]
        
        # Add vector field if requested
        if include_vector:
            vector_field = SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                vector_search_dimensions=DEFAULT_CONFIG["VECTOR_DIMENSIONS"],
                vector_search_profile_name="vector-profile"
            )
            fields.append(vector_field)
        
        # Create semantic settings
        semantic_config = SemanticConfiguration(
            name="openai-spec-semantic-config",
            prioritized_fields=SemanticPrioritizedFields(
                title_field=SemanticField(field_name="title"),
                content_fields=[SemanticField(field_name="content"), 
                                SemanticField(field_name="description")]
            )
        )
        
        # Create the vector search configuration if requested
        vector_search = None
        if include_vector:
            # Import the correct classes
            from azure.search.documents.indexes.models import HnswAlgorithmConfiguration
            
            # Create the algorithm configuration using the proper subclass
            vector_algorithm = HnswAlgorithmConfiguration(
                name="vector-config",
                m=4,
                ef_construction=400,
                ef_search=500,
                metric="cosine"
            )
            
            # Create vector search profile
            from azure.search.documents.indexes.models import VectorSearchProfile
            vector_profile = VectorSearchProfile(
                name="vector-profile",
                algorithm_configuration_name="vector-config"
            )
            
            # Create complete vector search config
            vector_search = VectorSearch(
                algorithms=[vector_algorithm],
                profiles=[vector_profile]
            )
        
        # Create the search index
        index = SearchIndex(
            name=index_name,
            fields=fields
        )
        
        # Add semantic settings if available
        if semantic_config:
            # Use SemanticSearch instead of SemanticSettings
            semantic_search = SemanticSearch(
                configurations=[semantic_config]
            )
            index.semantic_search = semantic_search
        
        # Add vector search if available
        if vector_search:
            index.vector_search = vector_search
        
        # Create or update the index
        result = index_client.create_or_update_index(index)
        logger.info(f"Index {result.name} created or updated")
        
        return True, ""
    
    except Exception as e:
        error_message = f"Error creating search index: {str(e)}"
        logger.error(error_message)
        return False, error_message


def run_test_query(query: str, index_name: str, use_vector: bool = True) -> None:
    """
    Run a test query against the Azure AI Search index.
    
    Args:
        query (str): The search query text
        index_name (str): Name of the search index
        use_vector (bool): Whether to use vector search
    """
    try:
        # Create a search client
        credential = AzureKeyCredential(AZURE_SEARCH_API_KEY)
        search_client = SearchClient(
            endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
            index_name=index_name,
            credential=credential
        )
        
        # If we're using vector search and OpenAI package is available, generate a query embedding
        vector = None
        if use_vector:
            try:
                import openai
                from openai import OpenAI
                
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    client = OpenAI(api_key=api_key)
                    embedding_response = client.embeddings.create(
                        input=query,
                        model="text-embedding-3-small"
                    )
                    vector = embedding_response.data[0].embedding
                    logger.info("Generated query vector embedding for search")
                else:
                    logger.warning("OPENAI_API_KEY not found. Using text search only.")
            except ImportError:
                logger.warning("OpenAI package not installed. Using text search only.")
                
        # Run a hybrid search (text + vector if available)
        if vector:
            # Vector search with text as fallback
            logger.info("Running hybrid search (vector + text)")
            results = search_client.search(
                search_text=query,
                select="id,title,description,type,path,method",
                vectors=[{"value": vector, "fields": "embedding", "k": 10}],
                top=10
            )
        else:
            # Standard text search
            logger.info("Running text search")
            results = search_client.search(
                search_text=query,
                select="id,title,description,type,path,method",
                query_type="semantic",
                semantic_configuration_name="openai-spec-semantic-config",
                top=10
            )
        
        # Display results
        logger.info(f"Test query: '{query}'")
        
        result_list = list(results)
        logger.info(f"Found {len(result_list)} results")
        
        print(f"\nSearch results for: '{query}'")
        print("-" * 50)
        
        for result in result_list:
            # Format output depending on the result type
            result_type = result.get('type', 'unknown')
            
            if result_type == 'endpoint':
                print(f"[{result.get('method', '').upper()}] {result.get('path', '')}")
                print(f"Title: {result.get('title', '')}")
            elif result_type == 'tag':
                print(f"Tag: {result.get('title', '')[5:]}")  # Remove "Tag: " prefix
            elif result_type == 'schema':
                print(f"Schema: {result.get('title', '')[8:]}")  # Remove "Schema: " prefix
            else:
                print(f"{result.get('title', '')}")
                
            # Print description (truncated if too long)
            desc = result.get('description', '')
            if len(desc) > 100:
                desc = desc[:97] + "..."
            print(f"Description: {desc}")
            print("-" * 50)
            
    except Exception as e:
        logger.error(f"Error running test query: {e}")
        print(f"Error running test query: {e}")


def generate_embeddings(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate embeddings for documents using OpenAI's text-embedding API.
    
    Args:
        documents (List[Dict[str, Any]]): The processed documents
        
    Returns:
        List[Dict[str, Any]]: Documents with embeddings added
    """
    try:
        import openai
        from openai import OpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not found. Skipping embeddings generation.")
            return documents
            
        client = OpenAI(api_key=api_key)
        
        # Process documents in batches to avoid rate limits
        batch_size = 20
        for i in tqdm(range(0, len(documents), batch_size), desc="Generating embeddings"):
            batch = documents[i:i+batch_size]
            
            # Extract texts to embed
            texts = [f"{doc.get('title', '')} {doc.get('description', '')} {doc.get('content', '')}" 
                    for doc in batch]
            
            # Generate embeddings
            response = client.embeddings.create(
                input=texts,
                model=DEFAULT_CONFIG["EMBEDDING_MODEL"]
            )
            
            # Add embeddings to documents
            for j, doc in enumerate(batch):
                doc["embedding"] = response.data[j].embedding
                
        logger.info(f"Generated embeddings for {len(documents)} documents")
        return documents
        
    except ImportError:
        logger.warning("OpenAI package not installed. Skipping embeddings generation.")
        return documents
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        logger.warning("Continuing without embeddings")
        return documents


def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_args()
    
    logger.info("Starting OpenAI API Specification Processor")
    
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    # Create data directory if it doesn't exist
    Path("data").mkdir(exist_ok=True)
    
    # Validate environment variables if uploading or indexing
    if not (args.fetch_only or args.process_only):
        missing_vars = []
        if not AZURE_STORAGE_CONNECTION_STRING:
            missing_vars.append("AZURE_STORAGE_CONNECTION_STRING")
        if not AZURE_SEARCH_SERVICE_ENDPOINT:
            missing_vars.append("AZURE_SEARCH_SERVICE_ENDPOINT")
        if not AZURE_SEARCH_API_KEY:
            missing_vars.append("AZURE_SEARCH_API_KEY")
        
        if missing_vars:
            logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
            logger.error("Please update your .env file with the required values")
            return
    
    # Process based on arguments
    spec_data = None
    documents = None
    
    # Fetch the OpenAI API specification if needed
    if not args.upload_only and not args.index_only:
        try:
            logger.info("Fetching OpenAI API specification")
            spec_data = fetch_openai_spec()
            logger.info(f"Successfully fetched API specification (version: {spec_data.get('info', {}).get('version', 'unknown')})")
            
            if args.fetch_only:
                logger.info("Fetch-only mode - exiting")
                return
        except Exception as e:
            logger.error(f"Error fetching OpenAI API specification: {e}")
            return
    
    # Process the API specification if needed
    if not args.fetch_only and not args.upload_only and not args.index_only:
        try:
            # If we don't have spec_data yet (e.g., in process-only mode), load it from file
            if spec_data is None:
                if not LOCAL_SPEC_PATH.exists():
                    logger.error(f"Specification file not found at {LOCAL_SPEC_PATH}. Please run with --fetch-only first.")
                    return
                logger.info("Loading OpenAI API specification from local file")
                with open(LOCAL_SPEC_PATH, "r") as f:
                    spec_data = yaml.safe_load(f)
            
            logger.info("Processing OpenAI API specification")
            documents = process_openai_spec(spec_data)
            logger.info(f"Processed {len(documents)} documents")
            
            # Generate embeddings if requested
            if not args.skip_embeddings:
                logger.info("Generating embeddings for documents")
                documents = generate_embeddings(documents)
                logger.info("Embeddings generation complete")
            
            # Save processed documents with custom chunk size
            PROCESSED_DOCS_PATH.mkdir(exist_ok=True)
            with open(PROCESSED_DOCS_PATH / "processed_documents.json", "w") as f:
                json.dump(documents, f, indent=2)
            
            if args.process_only:
                logger.info("Process-only mode - exiting")
                return
        except Exception as e:
            logger.error(f"Error processing OpenAI API specification: {e}")
            return
    
    # Upload documents to Blob Storage if needed
    if not args.fetch_only and not args.process_only and not args.index_only:
        try:
            # If we don't have documents yet (e.g., in upload-only mode), load them from file
            if documents is None:
                processed_file = PROCESSED_DOCS_PATH / "processed_documents.json"
                if not processed_file.exists():
                    logger.error(f"Processed documents not found at {processed_file}. Please run with --process-only first.")
                    return
                logger.info("Loading processed documents from local file")
                with open(processed_file, "r") as f:
                    documents = json.load(f)
            
            logger.info(f"Uploading {len(documents)} documents to Azure Blob Storage")
            upload_success = upload_to_blob_storage(documents)
            if not upload_success:
                logger.error("Failed to upload documents to Azure Blob Storage")
                return
            logger.info("Upload to Azure Blob Storage complete")
        except Exception as e:
            logger.error(f"Error uploading documents: {e}")
            return
    
    # Create Azure AI Search index if needed
    if not args.fetch_only and not args.process_only and not args.upload_only:
        try:
            logger.info(f"Creating/updating Azure AI Search index '{args.index}'")
            index_success, error_message = create_search_index(args.index, 
                                                              include_vector=not args.skip_embeddings)
            if not index_success:
                logger.error(f"Failed to create Azure AI Search index: {error_message}")
                return
            logger.info("Azure AI Search index creation/update complete")
        except Exception as e:
            logger.error(f"Error creating search index: {e}")
            return
    
    # Run a test query if requested
    if args.test_query:
        try:
            logger.info(f"Running test query: '{args.test_query}'")
            run_test_query(args.test_query, args.index)
        except Exception as e:
            logger.error(f"Error running test query: {e}")
    
    logger.info("OpenAI API Specification Processor completed successfully")


if __name__ == "__main__":
    main()