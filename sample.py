# Enhanced Azure GraphRAG Implementation with Blob Storage, AI Search, and Cosmos DB
# This implementation provides a complete document processing pipeline

import os
import asyncio
import yaml
import json
import pandas as pd
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional
import subprocess
import sys
import fitz  # PyMuPDF for PDF processing
import hashlib
from datetime import datetime, timezone
import uuid

# Azure SDK imports
from openai import AzureOpenAI
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.indexes.models import (
    SearchIndex, SearchField, SearchFieldDataType, SimpleField,
    SearchableField, VectorSearch, HnswAlgorithmConfiguration,
    VectorSearchProfile, SemanticConfiguration, SemanticSearch,
    SemanticPrioritizedFields, SemanticField
)
from azure.cosmos import CosmosClient, PartitionKey
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential

# Load environment variables
from dotenv import load_dotenv
load_dotenv()  # This loads the .env file

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AzureServiceManager:
    """Manages connections to Azure services"""
    
    def __init__(self):
        print("🔄 Loading environment variables...")

        self.openai_key = os.getenv("AZURE_OPENAI_KEY")
        self.openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.openai_deployment_chat = os.getenv("AZURE_OPENAI_DEPLOYMENT_CHAT")
        self.openai_deployment_embed = os.getenv("AZURE_OPENAI_DEPLOYMENT_EMBED")
        self.openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        
        # Load configuration from environment variables - matching your .env file
        self.storage_account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
        self.storage_account_key = os.getenv("AZURE_STORAGE_ACCOUNT_KEY") 
        self.storage_connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        self.container_name = os.getenv("AZURE_BLOB_CONTAINER_NAME", "documents")  # Your .env uses AZURE_BLOB_CONTAINER_NAME

        self.search_service_name = os.getenv("AZURE_SEARCH_SERVICE_NAME")
        self.search_api_key = os.getenv("AZURE_SEARCH_API_KEY")  # Your .env uses AZURE_SEARCH_KEY, not AZURE_SEARCH_API_KEY
        self.search_index_name = os.getenv("AZURE_SEARCH_INDEX", "graphrag-documents")  # Your .env uses AZURE_SEARCH_INDEX
        
        self.cosmos_endpoint = os.getenv("AZURE_COSMOS_ENDPOINT")
        self.cosmos_key = os.getenv("AZURE_COSMOS_KEY")
        self.cosmos_database_name = os.getenv("AZURE_COSMOS_DATABASE_NAME", "GraphRAGDB")
        self.cosmos_container_name = os.getenv("AZURE_COSMOS_CONTAINER_NAME", "DocumentMetadata")
        
        # Debug: Print loaded values (without sensitive data)
        print(f"🔍 Debug - OpenAI Key: {self.openai_key}")
        print(f"🔍 Debug - OpenAI Endpoint: {self.openai_endpoint}")
        print(f"🔍 Debug - OpenAI Chat Deployment: {self.openai_deployment_chat}")
        print(f"🔍 Debug - OpenAI Embed Deployment: {self.openai_deployment_embed}")
        print(f"🔍 Debug - Storage Account: {self.storage_account_name}")
        print(f"🔍 Debug - Container Name: {self.container_name}")
        print(f"🔍 Debug - Search Service: {self.search_service_name}")
        print(f"🔍 Debug - Search Index: {self.search_index_name}")
        print(f"🔍 Debug - Cosmos Endpoint: {self.cosmos_endpoint}")
        print(f"🔍 Debug - Cosmos Database: {self.cosmos_database_name}")
        print(f"🔍 Debug - Cosmos Container: {self.cosmos_container_name}")
        
        # Initialize clients
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize Azure service clients"""
        try:
            print("🔄 Initializing Azure service clients...")
            
            # Validate required variables first
            self._validate_environment_variables()

            if self.openai_key and self.openai_endpoint:
                print("✅ OpenAI configuration is valid")
                # Initialize Azure OpenAI client
                self.openai_client = AzureOpenAI(
                    api_key=self.openai_key,
                    api_version=self.openai_api_version,
                    azure_endpoint=self.openai_endpoint
                )
            
            # Blob Storage Client
            if self.storage_connection_string:
                print("📦 Initializing Blob Storage with connection string...")
                self.blob_service_client = BlobServiceClient.from_connection_string(
                    self.storage_connection_string
                )
            else:
                print("📦 Initializing Blob Storage with account key...")
                account_url = f"https://{self.storage_account_name}.blob.core.windows.net"
                self.blob_service_client = BlobServiceClient(
                    account_url=account_url, 
                    credential=self.storage_account_key
                )
            
            # Create container if it doesn't exist - with better error handling
            try:
                container_client = self.blob_service_client.get_container_client(self.container_name)
                if not container_client.exists():
                    self.blob_service_client.create_container(self.container_name)
                    logger.info(f"✅ Created new container: {self.container_name}")
                else:
                    logger.info(f"✅ Using existing container: {self.container_name}")
            except Exception as e:
                logger.warning(f"⚠️ Container operation warning: {e}")
                # Continue execution even if container operation has issues
            
            # Azure Search Client
            print("🔍 Initializing Azure Search...")
            search_endpoint = f"https://{self.search_service_name}.search.windows.net"
            self.search_client = SearchClient(
                endpoint=search_endpoint,
                index_name=self.search_index_name,
                credential=AzureKeyCredential(self.search_api_key)
            )
            self.search_index_client = SearchIndexClient(
                endpoint=search_endpoint,
                credential=AzureKeyCredential(self.search_api_key)
            )
            
            # Cosmos DB Client
            print("🌌 Initializing Cosmos DB...")
            self.cosmos_client = CosmosClient(self.cosmos_endpoint, credential=self.cosmos_key)
            
            # Create database and container if they don't exist
            try:
                self.database = self.cosmos_client.create_database_if_not_exists(
                    id=self.cosmos_database_name
                )
                logger.info(f"✅ Database ready: {self.cosmos_database_name}")
            except Exception as e:
                logger.error(f"❌ Failed to create/access database: {e}")
                raise
                
            try:
                self.cosmos_container = self.database.create_container_if_not_exists(
                    id=self.cosmos_container_name,
                    partition_key=PartitionKey(path="/document_id"),
                    offer_throughput=400
                )
                logger.info(f"✅ Container ready: {self.cosmos_container_name}")
            except Exception as e:
                logger.error(f"❌ Failed to create/access container: {e}")
                raise
            
            logger.info("✅ All Azure services initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Azure services: {e}")
            raise
    
    def _validate_environment_variables(self):
        """Validate that all required environment variables are set"""
        required_vars = {
            "AZURE_OPENAI_KEY": self.openai_key,
            "AZURE_OPENAI_ENDPOINT": self.openai_endpoint,
            "AZURE_OPENAI_DEPLOYMENT_CHAT": self.openai_deployment_chat,
            "AZURE_OPENAI_DEPLOYMENT_EMBED": self.openai_deployment_embed,
            "AZURE_OPENAI_API_VERSION": self.openai_api_version,
            "AZURE_STORAGE_CONNECTION_STRING": self.storage_connection_string,
            "AZURE_SEARCH_SERVICE_NAME": self.search_service_name,
            "AZURE_SEARCH_KEY": self.search_api_key,
            "AZURE_COSMOS_ENDPOINT": self.cosmos_endpoint,
            "AZURE_COSMOS_KEY": self.cosmos_key
        }
        
        missing_vars = [var for var, value in required_vars.items() if not value]
        
        if missing_vars:
            logger.error("❌ Missing required environment variables:")
            for var in missing_vars:
                logger.error(f"   - {var}")
            raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")
        
        # Special validation for Cosmos DB key
        if not isinstance(self.cosmos_key, str) or len(self.cosmos_key.strip()) < 10:
            raise ValueError("❌ AZURE_COSMOS_KEY must be a valid string")
        
        logger.info("✅ All required environment variables are validated")

class PDFProcessor:
    """Handles PDF document processing"""
    
    def __init__(self, azure_manager: AzureServiceManager):
        self.azure_manager = azure_manager
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text and metadata from PDF"""
        try:
            doc = fitz.open(pdf_path)
        
            # Extract text from all pages
            full_text = ""
            pages_text = []
            
            # Get page count before any processing
            page_count = len(doc)
        
            for page_num in range(page_count):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                pages_text.append({
                    "page_number": page_num + 1,
                    "text": page_text
                })
                full_text += f"\n--- Page {page_num + 1} ---\n{page_text}"
        
            # Extract metadata before closing
            metadata = doc.metadata
            
            # Close the document after extracting all needed data
            doc.close()
        
            # Calculate file hash for deduplication
            with open(pdf_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
        
            return {
                "full_text": full_text.strip(),
                "pages": pages_text,
                "metadata": {
                    "title": metadata.get("title", ""),
                    "author": metadata.get("author", ""),
                    "subject": metadata.get("subject", ""),
                    "creator": metadata.get("creator", ""),
                    "producer": metadata.get("producer", ""),
                    "creation_date": metadata.get("creationDate", ""),
                    "modification_date": metadata.get("modDate", ""),
                    "page_count": page_count  # Use the stored page_count
                },
                "file_hash": file_hash,
                "processing_timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        except Exception as e:
            logger.error(f"Failed to extract text from PDF {pdf_path}: {e}")
            raise

    def upload_pdf_to_blob(self, pdf_path: str, blob_name: Optional[str] = None) -> str:
        """Upload PDF to Azure Blob Storage"""
        try:
            if blob_name is None:
                blob_name = f"pdfs/{Path(pdf_path).name}"
        
            blob_client = self.azure_manager.blob_service_client.get_blob_client(
                container=self.azure_manager.container_name,
                blob=blob_name
            )
        
            with open(pdf_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
        
            blob_url = blob_client.url
            logger.info(f"✅ PDF uploaded to blob storage: {blob_url}")
            return blob_url
        
        except Exception as e:
            logger.error(f"Failed to upload PDF to blob storage: {e}")
            raise

class AzureSearchManager:
    """Manages Azure AI Search integration"""
    
    def __init__(self, azure_manager: AzureServiceManager):
        self.azure_manager = azure_manager
        self.index_name = azure_manager.search_index_name
    
    def create_search_index(self):
        """Create or update the search index for documents"""
        try:
            # Define the search index schema
            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SearchableField(name="document_id", type=SearchFieldDataType.String),
                SearchableField(name="title", type=SearchFieldDataType.String),
                SearchableField(name="content", type=SearchFieldDataType.String),
                SearchableField(name="author", type=SearchFieldDataType.String),
                SimpleField(name="page_number", type=SearchFieldDataType.Int32),
                SimpleField(name="file_path", type=SearchFieldDataType.String),
                SimpleField(name="blob_url", type=SearchFieldDataType.String),
                SimpleField(name="upload_timestamp", type=SearchFieldDataType.DateTimeOffset),
                SimpleField(name="file_hash", type=SearchFieldDataType.String),
                # Vector field for semantic search
                SearchField(
                    name="content_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    vector_search_dimensions=1536,  # Adjust based on your embedding model
                    vector_search_profile_name="vector-profile"
                )
            ]
            
            # Configure vector search
            vector_search = VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(name="hnsw-algorithm")
                ],
                profiles=[
                    VectorSearchProfile(
                        name="vector-profile",
                        algorithm_configuration_name="hnsw-algorithm"
                    )
                ]
            )
            
            # Configure semantic search
            semantic_config = SemanticConfiguration(
                name="semantic-config",
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="title"),
                    content_fields=[SemanticField(field_name="content")]
                )
            )
            
            semantic_search = SemanticSearch(configurations=[semantic_config])
            
            # Create the index
            index = SearchIndex(
                name=self.index_name,
                fields=fields,
                vector_search=vector_search,
                semantic_search=semantic_search
            )
            
            result = self.azure_manager.search_index_client.create_or_update_index(index)
            logger.info(f"✅ Search index created/updated: {result.name}")
            
        except Exception as e:
            logger.error(f"Failed to create search index: {e}")
            raise
    
    def index_document(self, document_data: Dict[str, Any], embeddings: List[float] = None):
        """Index a document in Azure AI Search"""
        try:
            search_document = {
                "id": str(uuid.uuid4()),
                "document_id": document_data.get("document_id"),
                "title": document_data.get("title", ""),
                "content": document_data.get("content", ""),
                "author": document_data.get("author", ""),
                "page_number": document_data.get("page_number"),
                "file_path": document_data.get("file_path", ""),
                "blob_url": document_data.get("blob_url", ""),
                "upload_timestamp": document_data.get("upload_timestamp"),
                "file_hash": document_data.get("file_hash", "")
            }
            
            if embeddings:
                search_document["content_vector"] = embeddings
            
            result = self.azure_manager.search_client.upload_documents(documents=[search_document])
            logger.info(f"✅ Document indexed successfully: {result[0].key}")
            return result[0].key
            
        except Exception as e:
            logger.error(f"Failed to index document: {e}")
            raise
    
    def search_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for documents using hybrid search"""
        try:
            results = self.azure_manager.search_client.search(
                search_text=query,
                top=top_k,
                query_type="semantic",
                semantic_configuration_name="semantic-config",
                select=["document_id", "title", "content", "author", "blob_url"]
            )
            
            search_results = []
            for result in results:
                search_results.append({
                    "document_id": result.get("document_id"),
                    "title": result.get("title"),
                    "content": result.get("content"),
                    "author": result.get("author"),
                    "blob_url": result.get("blob_url"),
                    "score": result.get("@search.score")
                })
            
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to search documents: {e}")
            raise

class CosmosDBManager:
    """Manages document metadata in Cosmos DB"""
    
    def __init__(self, azure_manager: AzureServiceManager):
        self.azure_manager = azure_manager
        self.container = azure_manager.cosmos_container
    
    def store_document_metadata(self, document_data: Dict[str, Any]) -> str:
        """Store document metadata in Cosmos DB"""
        try:
            document_id = str(uuid.uuid4())
            
            cosmos_document = {
                "id": document_id,
                "document_id": document_id,
                "title": document_data.get("title", ""),
                "author": document_data.get("author", ""),
                "subject": document_data.get("subject", ""),
                "page_count": document_data.get("page_count", 0),
                "file_hash": document_data.get("file_hash", ""),
                "file_path": document_data.get("file_path", ""),
                "blob_url": document_data.get("blob_url", ""),
                "processing_status": "indexed",
                "upload_timestamp": document_data.get("upload_timestamp"),
                "processing_timestamp": datetime.now(timezone.utc).isoformat(),
                "graphrag_status": "pending",
                "metadata": document_data.get("metadata", {})
            }
            
            result = self.container.create_item(body=cosmos_document)
            logger.info(f"✅ Document metadata stored in Cosmos DB: {document_id}")
            return document_id
            
        except Exception as e:
            logger.error(f"Failed to store document metadata: {e}")
            raise
    
    def get_document_metadata(self, document_id: str) -> Dict:
        """Retrieve document metadata from Cosmos DB"""
        try:
            item = self.container.read_item(item=document_id, partition_key=document_id)
            return item
        except Exception as e:
            logger.error(f"Failed to retrieve document metadata: {e}")
            raise
    
    def update_graphrag_status(self, document_id: str, status: str, details: Dict = None):
        """Update GraphRAG processing status"""
        try:
            item = self.container.read_item(item=document_id, partition_key=document_id)
            item["graphrag_status"] = status
            item["graphrag_timestamp"] = datetime.now(timezone.utc).isoformat()
            
            if details:
                item["graphrag_details"] = details
            
            self.container.replace_item(item=document_id, body=item)
            logger.info(f"✅ Updated GraphRAG status for {document_id}: {status}")
            
        except Exception as e:
            logger.error(f"Failed to update GraphRAG status: {e}")
            raise
    
    def list_processed_documents(self) -> List[Dict]:
        """List all processed documents"""
        try:
            query = "SELECT * FROM c WHERE c.processing_status = 'indexed'"
            items = list(self.container.query_items(
                query=query,
                enable_cross_partition_query=True
            ))
            return items
        except Exception as e:
            logger.error(f"Failed to list processed documents: {e}")
            raise

class EnhancedGraphRAGConfig:
    """Enhanced GraphRAG configuration with Azure services integration"""
    
    def __init__(self):
        self.project_root = Path("./graphrag_project")
        self.input_dir = self.project_root / "input"
        self.output_dir = self.project_root / "output"
        self.settings_file = self.project_root / "settings.yaml"
        
    def create_project_structure(self):
        """Create the necessary directory structure"""
        self.project_root.mkdir(exist_ok=True)
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        print("✓ Project structure created")
        
    def create_settings_yaml(self):
        """Create comprehensive settings.yaml configuration"""
        # Get environment variables
        llm_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_CHAT")
        embedding_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_EMBED")
        azure_endpoint_full = os.getenv("AZURE_OPENAI_ENDPOINT")
        
        # Extract base endpoint from full URL if needed
        # Your endpoint: "https://rohan-mfs2dn0d-northcentralus.cognitiveservices.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2025-01-01-preview"
        # We need: "https://rohan-mfs2dn0d-northcentralus.cognitiveservices.azure.com"
        if azure_endpoint_full and "/openai/" in azure_endpoint_full:
            azure_endpoint = azure_endpoint_full.split("/openai/")[0]
        else:
            azure_endpoint = azure_endpoint_full
        
        print(f"🔍 Debug - Full endpoint: {azure_endpoint_full}")
        print(f"🔍 Debug - Base endpoint: {azure_endpoint}")
        print(f"🔍 Debug - LLM deployment: {llm_deployment_name}")
        print(f"🔍 Debug - Embedding deployment: {embedding_deployment_name}")
        
        settings = {
            "encoding_model": "cl100k_base",
            "skip_workflows": [],
            
            # LLM Configuration for Azure OpenAI
            "llm": {
                "api_key": "${AZURE_OPENAI_KEY}",
                "type": "azure_openai_chat",
                "model": "gpt-4o",
                "model_supports_json": True,
                "api_base": azure_endpoint,
                "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
                "deployment_name": llm_deployment_name,
                "max_tokens": 4000,
                "temperature": 0.1,
                "top_p": 1.0,
                "request_timeout": 180.0,
                "max_retries": 10,
                "max_retry_wait": 10.0,
                "sleep_on_rate_limit_recommendation": True,
                "concurrent_requests": 3
            },
            
            # Embeddings Configuration
            "embeddings": {
                "api_key": "${AZURE_OPENAI_KEY}",
                "type": "azure_openai_embedding",
                "model": "text-embedding-3-large",
                "api_base": azure_endpoint,
                "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
                "deployment_name": embedding_deployment_name,
                "max_tokens": 8192,
                "request_timeout": 60.0,
                "max_retries": 10,
                "max_retry_wait": 10.0,
                "sleep_on_rate_limit_recommendation": True,
                "concurrent_requests": 5
            },
            
            # Input Configuration
            "input": {
                "type": "file",
                "file_type": "text",
                "base_dir": "input",
                "file_encoding": "utf-8",
                "file_pattern": ".*\\.txt$"
            },
            
            # Storage Configuration
            "storage": {
                "type": "file"
            },
            
            # Cache Configuration
            "cache": {
                "type": "file",
                "base_dir": "cache"
            },
            
            # Reporting Configuration
            "reporting": {
                "type": "file",
                "base_dir": "output"
            },
            
            # Enhanced Entity Extraction Configuration
            "entity_extraction": {
                "entity_types": [
                    "PERSON", "ORGANIZATION", "LOCATION", "EVENT", 
                    "CONCEPT", "TECHNOLOGY", "PRODUCT", "DATE"
                ],
                "max_gleanings": 2,
                "strategy": {
                    "type": "graph_intelligence"
                }
            },
            
            # Community Summarization Configuration
            "community_summarization": {
                "max_length": 3000,
                "strategy": {
                    "type": "graph_intelligence"
                }
            },
            
            # Chunk Configuration for better PDF processing
            "chunks": {
                "size": 1500,
                "overlap": 150,
                "group_by_columns": ["source"]
            },
            
            # Snapshot Configuration
            "snapshots": {
                "graphml": True,
                "raw_entities": True,
                "top_level_nodes": True
            }
        }
        
        with open(self.settings_file, 'w') as f:
            yaml.dump(settings, f, default_flow_style=False, indent=2)
        print("✓ Enhanced settings.yaml created with correct Azure endpoints")

class EnhancedDocumentProcessor:
    """Enhanced document processor with PDF and Azure services support"""
    
    def __init__(self, input_dir: Path, azure_manager: AzureServiceManager):
        self.input_dir = input_dir
        self.azure_manager = azure_manager
        self.pdf_processor = PDFProcessor(azure_manager)
        self.search_manager = AzureSearchManager(azure_manager)
        self.cosmos_manager = CosmosDBManager(azure_manager)
    

    def process_pdf_upload(self, pdf_path: str) -> str:
        """Complete PDF processing pipeline"""
        try:
            logger.info(f"🔄 Processing PDF: {pdf_path}")
        
            # Step 1: Extract text and metadata from PDF (using pdf_processor instance)
            pdf_data = self.pdf_processor.extract_text_from_pdf(pdf_path)
        
            # Step 2: Upload PDF to Blob Storage (using pdf_processor instance)
            blob_url = self.pdf_processor.upload_pdf_to_blob(pdf_path)
        
            # Step 3: Store metadata in Cosmos DB
            document_metadata = {
                "title": pdf_data["metadata"].get("title") or Path(pdf_path).stem,
                "author": pdf_data["metadata"].get("author", ""),
                "subject": pdf_data["metadata"].get("subject", ""),
                "page_count": pdf_data["metadata"].get("page_count", 0),
                "file_hash": pdf_data["file_hash"],
                "file_path": pdf_path,
                "blob_url": blob_url,
                "upload_timestamp": pdf_data["processing_timestamp"],
                "metadata": pdf_data["metadata"]
            }
        
            document_id = self.cosmos_manager.store_document_metadata(document_metadata)
        
            # Step 4: Index in Azure AI Search
            search_document = {
                "document_id": document_id,
                "title": document_metadata["title"],
                "content": pdf_data["full_text"][:50000],  # Truncate for search
                "author": document_metadata["author"],
                "file_path": pdf_path,
                "blob_url": blob_url,
                "upload_timestamp": pdf_data["processing_timestamp"],
                "file_hash": pdf_data["file_hash"]
            }
        
            self.search_manager.index_document(search_document)
        
            # Step 5: Create text file for GraphRAG processing
            text_filename = f"{Path(pdf_path).stem}_{document_id[:8]}.txt"
            text_file_path = self.input_dir / text_filename
        
            with open(text_file_path, 'w', encoding='utf-8') as f:
                f.write(f"Title: {document_metadata['title']}\n")
                f.write(f"Author: {document_metadata['author']}\n")
                f.write(f"Source: {pdf_path}\n")
                f.write(f"Document ID: {document_id}\n\n")
                f.write(pdf_data["full_text"])
        
            logger.info(f"✅ PDF processing completed: {document_id}")
            return document_id
        
        except Exception as e:
            logger.error(f"❌ Failed to process PDF {pdf_path}: {e}")
            raise
    
    def list_documents(self):
        """List all processed documents from Cosmos DB"""
        try:
            documents = self.cosmos_manager.list_processed_documents()
            print(f"\n📄 Processed Documents ({len(documents)} total):")
            for doc in documents:
                print(f"  - {doc['title']} (ID: {doc['document_id'][:8]}...)")
                print(f"    Author: {doc.get('author', 'Unknown')}")
                print(f"    Status: {doc.get('graphrag_status', 'Unknown')}")
                print(f"    Pages: {doc.get('page_count', 'N/A')}")
                print()
            return documents
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []

import importlib.util
import shutil
import tempfile
from pathlib import Path
import os
import logging
from typing import Dict, Any, Optional, List
import subprocess
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class BlobStorageManager:
    """Manages PDF document fetching from Azure Blob Storage"""
    
    def __init__(self, azure_manager):
        self.azure_manager = azure_manager
        self.blob_service_client = azure_manager.blob_service_client
        self.container_name = azure_manager.container_name
        self.temp_dir = Path(tempfile.mkdtemp(prefix="graphrag_pdfs_"))
        
    def fetch_pdf_from_blob(self, blob_name: str, local_filename: Optional[str] = None) -> Path:
        """
        Fetch a PDF file from Azure Blob Storage to local storage
        
        Args:
            blob_name: Name of the blob in storage (e.g., "pdfs/document.pdf")
            local_filename: Optional local filename, if not provided uses blob name
            
        Returns:
            Path: Local file path of the downloaded PDF
        """
        try:
            # Get blob client
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            
            # Check if blob exists
            if not blob_client.exists():
                raise FileNotFoundError(f"Blob not found: {blob_name}")
            
            # Determine local filename
            if local_filename is None:
                local_filename = Path(blob_name).name
            
            local_path = self.temp_dir / local_filename
            
            # Download the blob
            with open(local_path, "wb") as download_file:
                download_data = blob_client.download_blob()
                download_file.write(download_data.readall())
            
            logger.info(f"PDF downloaded from blob: {blob_name} -> {local_path}")
            return local_path
            
        except Exception as e:
            logger.error(f"Failed to fetch PDF from blob {blob_name}: {e}")
            raise
    
    def fetch_all_pdfs_from_container(self, blob_prefix: str = "pdfs/") -> List[Path]:
        """
        Fetch all PDF files from a container with given prefix
        
        Args:
            blob_prefix: Prefix to filter blobs (default: "pdfs/")
            
        Returns:
            List[Path]: List of local file paths of downloaded PDFs
        """
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            blob_list = container_client.list_blobs(name_starts_with=blob_prefix)
            
            downloaded_files = []
            
            for blob in blob_list:
                if blob.name.lower().endswith('.pdf'):
                    try:
                        local_path = self.fetch_pdf_from_blob(blob.name)
                        downloaded_files.append(local_path)
                    except Exception as e:
                        logger.warning(f"Failed to download {blob.name}: {e}")
                        continue
            
            logger.info(f"Downloaded {len(downloaded_files)} PDF files from container")
            return downloaded_files
            
        except Exception as e:
            logger.error(f"Failed to fetch PDFs from container: {e}")
            raise
    
    def get_pdf_metadata_from_blob(self, blob_name: str) -> Dict[str, Any]:
        """
        Get metadata about a PDF blob without downloading it
        
        Args:
            blob_name: Name of the blob
            
        Returns:
            Dict: Blob metadata including size, last modified, etc.
        """
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            
            properties = blob_client.get_blob_properties()
            
            return {
                "blob_name": blob_name,
                "size": properties.size,
                "last_modified": properties.last_modified,
                "content_type": properties.content_settings.content_type,
                "etag": properties.etag,
                "url": blob_client.url
            }
            
        except Exception as e:
            logger.error(f"Failed to get blob metadata for {blob_name}: {e}")
            raise
    
    def cleanup_temp_files(self):
        """Clean up temporary downloaded files"""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.info("Temporary files cleaned up")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp files: {e}")

class EnhancedGraphRAGRunner:
    """Enhanced GraphRAG runner with Azure services integration and blob storage PDF fetching"""
    
    def __init__(self, project_root: Path, azure_manager):
        self.project_root = project_root
        self.azure_manager = azure_manager
        self.cosmos_manager = CosmosDBManager(azure_manager)
        self.search_manager = AzureSearchManager(azure_manager)
        self.blob_manager = BlobStorageManager(azure_manager)
        
    def fetch_and_prepare_pdfs_for_processing(self) -> List[Dict[str, Any]]:
        """
        Fetch PDFs from blob storage and prepare them for GraphRAG processing
        
        Returns:
            List[Dict]: List of prepared PDF information
        """
        try:
            print("📥 Fetching PDF documents from Azure Blob Storage...")
            
            # Get list of processed documents from Cosmos DB
            processed_documents = self.cosmos_manager.list_processed_documents()
            
            prepared_pdfs = []
            
            for doc in processed_documents:
                blob_url = doc.get('blob_url')
                if not blob_url:
                    continue
                
                # Extract blob name from URL
                # URL format: https://storageaccount.blob.core.windows.net/container/blob_name
                try:
                    blob_name = blob_url.split('/')[-1]
                    if not blob_name.endswith('.pdf'):
                        # If the blob name doesn't end with .pdf, it might be in a folder
                        url_parts = blob_url.split('/')
                        container_index = url_parts.index(self.azure_manager.container_name)
                        blob_name = '/'.join(url_parts[container_index + 1:])
                    
                    # Download PDF from blob storage
                    local_pdf_path = self.blob_manager.fetch_pdf_from_blob(blob_name)
                    
                    prepared_pdfs.append({
                        'document_id': doc['document_id'],
                        'title': doc['title'],
                        'local_path': local_pdf_path,
                        'blob_name': blob_name,
                        'blob_url': blob_url,
                        'metadata': doc
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to process document {doc['document_id']}: {e}")
                    continue
            
            logger.info(f"Successfully prepared {len(prepared_pdfs)} PDFs for processing")
            return prepared_pdfs
            
        except Exception as e:
            logger.error(f"Failed to fetch and prepare PDFs: {e}")
            raise
    
    def process_blob_pdfs_for_graphrag(self) -> bool:
        """
        Process PDFs from blob storage and create text files for GraphRAG
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print("🔄 Processing PDFs from blob storage for GraphRAG...")
            
            # Fetch PDFs from blob storage
            prepared_pdfs = self.fetch_and_prepare_pdfs_for_processing()
            
            if not prepared_pdfs:
                print("⚠️ No PDFs found in blob storage or Cosmos DB")
                return False
            
            # Ensure input directory exists
            input_dir = self.project_root / "input"
            input_dir.mkdir(exist_ok=True)
            
            # Process each PDF
            for pdf_info in prepared_pdfs:
                try:
                    # Extract text from the downloaded PDF
                    pdf_processor = PDFProcessor(self.azure_manager)
                    pdf_data = pdf_processor.extract_text_from_pdf(str(pdf_info['local_path']))
                    
                    # Create text file for GraphRAG
                    text_filename = f"{pdf_info['title'][:50].replace(' ', '_')}_{pdf_info['document_id'][:8]}.txt"
                    text_file_path = input_dir / text_filename
                    
                    with open(text_file_path, 'w', encoding='utf-8') as f:
                        f.write(f"Title: {pdf_info['title']}\n")
                        f.write(f"Document ID: {pdf_info['document_id']}\n")
                        f.write(f"Source: {pdf_info['blob_url']}\n")
                        f.write(f"Author: {pdf_info['metadata'].get('author', '')}\n")
                        f.write(f"Processing Date: {datetime.now().isoformat()}\n")
                        f.write("\n" + "="*80 + "\n\n")
                        f.write(pdf_data["full_text"])
                    
                    print(f"✅ Processed: {pdf_info['title']}")
                    
                except Exception as e:
                    logger.error(f"Failed to process PDF {pdf_info['title']}: {e}")
                    continue
            
            print(f"🎯 Created {len(prepared_pdfs)} text files for GraphRAG processing")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process blob PDFs for GraphRAG: {e}")
            return False
    
    def list_available_pdfs_in_blob(self) -> List[Dict[str, Any]]:
        """
        List all available PDF files in blob storage
        
        Returns:
            List[Dict]: List of available PDFs with metadata
        """
        try:
            container_client = self.azure_manager.blob_service_client.get_container_client(
                self.azure_manager.container_name
            )
            
            blob_list = container_client.list_blobs(name_starts_with="pdfs/")
            available_pdfs = []
            
            for blob in blob_list:
                if blob.name.lower().endswith('.pdf'):
                    try:
                        metadata = self.blob_manager.get_pdf_metadata_from_blob(blob.name)
                        available_pdfs.append(metadata)
                    except Exception as e:
                        logger.warning(f"Failed to get metadata for {blob.name}: {e}")
                        continue
            
            print(f"📋 Found {len(available_pdfs)} PDF files in blob storage:")
            for pdf in available_pdfs:
                print(f"  - {pdf['blob_name']} ({pdf['size']} bytes)")
            
            return available_pdfs
            
        except Exception as e:
            logger.error(f"Failed to list PDFs in blob storage: {e}")
            return []
    
    def initialize_graphrag(self):
        """Initialize GraphRAG with auto-fallback to manual config"""
        original_dir = os.getcwd()
        os.chdir(self.project_root)
        
        try:
            # Check if already initialized
            if (self.project_root / "settings.yaml").exists():
                print("✓ GraphRAG already initialized")
                return True
            
            # Fallback: Create manual configuration
            print("🔧 Creating manual GraphRAG configuration...")
            return self._create_manual_config()
            
        except Exception as e:
            print(f"❌ Initialization error: {e}")
            return False
        finally:
            os.chdir(original_dir)
    
    def _create_manual_config(self):
        """Create minimal working GraphRAG configuration"""
        try:
            # Ensure the project root directory exists
            self.project_root.mkdir(parents=True, exist_ok=True)
            
            # Extract base endpoint from full URL if needed
            azure_endpoint_full = self.azure_manager.openai_endpoint
            if azure_endpoint_full and "/openai/" in azure_endpoint_full:
                azure_endpoint = azure_endpoint_full.split("/openai/")[0]
            else:
                azure_endpoint = azure_endpoint_full
            
            settings_content = f"""
llm:
  api_key: {self.azure_manager.openai_key}
  type: azure_openai_chat
  model: gpt-4o
  api_base: {azure_endpoint}
  api_version: {self.azure_manager.openai_api_version}
  deployment_name: {self.azure_manager.openai_deployment_chat}
  tokens_per_minute: 150000
  requests_per_minute: 1000

embeddings:
  llm:
    api_key: {self.azure_manager.openai_key}
    type: azure_openai_embedding
    model: text-embedding-3-large
    api_base: {azure_endpoint}
    api_version: {self.azure_manager.openai_api_version}
    deployment_name: {self.azure_manager.openai_deployment_embed}

input:
  type: file
  base_dir: "./input"
  file_pattern: ".*\\.txt$"

storage:
  type: file
  base_dir: "./output"

chunk:
  size: 300
  overlap: 100

entity_extraction:
  entity_types: [organization,person,geo,event]

local_search:
  top_k_mapped_entities: 10
  max_tokens: 16384
"""
            
            # Create settings.yaml
            settings_file = self.project_root / "settings.yaml"
            with open(settings_file, 'w', encoding='utf-8') as f:
                f.write(settings_content)
            
            # Create necessary directories
            for dir_name in ['input', 'output', 'cache']:
                (self.project_root / dir_name).mkdir(parents=True, exist_ok=True)
            
            print("✓ Manual configuration created successfully")
            print(f"✓ Project directory: {self.project_root}")
            print(f"✓ Settings file: {settings_file}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create manual config: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_indexing(self):
        """Run the GraphRAG indexing process with status updates and blob PDF processing"""
        original_dir = os.getcwd()
        os.chdir(self.project_root)
        
        try:
            print("🔄 Starting GraphRAG indexing process...")
            
            # Step 1: Process PDFs from blob storage
            if not self.process_blob_pdfs_for_graphrag():
                print("❌ Failed to process PDFs from blob storage")
                return False
            
            # Step 2: Check for input files
            input_files = list((self.project_root / "input").glob("*.txt"))
            if not input_files:
                print("❌ No .txt files found in ./input directory after processing")
                return False
            
            print(f"📁 Processing {len(input_files)} files")
            
            # Step 3: Update Cosmos DB status
            documents = self.cosmos_manager.list_processed_documents()
            for doc in documents:
                if doc.get("graphrag_status") == "pending":
                    self.cosmos_manager.update_graphrag_status(
                        doc["document_id"], 
                        "processing",
                        {"stage": "indexing_started"}
                    )
            
            # Step 4: Try indexing commands
            commands_to_try = [
                ['python', '-m', 'graphrag.index', '--root', '.'],
                ['graphrag', 'index', '--root', '.'],
                ['python', '-m', 'graphrag', 'index', '--root', '.']
            ]
            
            success = False
            for cmd in commands_to_try:
                try:
                    print(f"🔄 Running: {' '.join(cmd)}")
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    
                    if result.returncode == 0:
                        print("✓ GraphRAG indexing completed successfully")
                        success = True
                        break
                    else:
                        print(f"⚠️ Command failed: {result.stderr}")
                        
                except subprocess.TimeoutExpired:
                    print("⚠️ Indexing timed out (5 minutes)")
                except Exception as e:
                    print(f"⚠️ Error: {e}")
                    continue
            
            # Step 5: Update Cosmos DB status
            status = "completed" if success else "failed"
            stage_info = {
                "stage": f"indexing_{status}",
                "output" if success else "error": "Graph built successfully" if success else "Indexing failed"
            }
            
            for doc in documents:
                if doc.get("graphrag_status") == "processing":
                    self.cosmos_manager.update_graphrag_status(doc["document_id"], status, stage_info)
            
            return success
            
        except Exception as e:
            print(f"❌ Indexing error: {e}")
            return False
        finally:
            os.chdir(original_dir)
            # Clean up temporary files
            self.blob_manager.cleanup_temp_files()
    
    def enhanced_query(self, query: str, method: str = "both") -> Dict[str, Any]:
        """Enhanced query combining Azure Search and GraphRAG"""
        try:
            results = {
                "query": query,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "azure_search_results": [],
                "graphrag_global": None,
                "graphrag_local": None,
                "combined_answer": None
            }
            
            # Step 1: Azure AI Search
            print(f"🔍 Searching Azure AI Search: {query}")
            search_results = self.search_manager.search_documents(query, top_k=3)
            results["azure_search_results"] = search_results
            
            if search_results:
                print(f"📋 Found {len(search_results)} relevant documents")
            
            # Step 2: GraphRAG queries
            original_dir = os.getcwd()
            os.chdir(self.project_root)
            
            try:
                commands_base = [
                    ['python', '-m', 'graphrag.query', '--root', '.', '--method'],
                    ['graphrag', 'query', '--root', '.', '--method']
                ]
                
                # Global query
                if method in ["global", "both"]:
                    for cmd_base in commands_base:
                        try:
                            cmd = cmd_base + ['global', query]
                            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                            if result.returncode == 0:
                                results["graphrag_global"] = result.stdout.strip()
                                print("📊 Global query completed")
                                break
                        except:
                            continue
                
                # Local query
                if method in ["local", "both"]:
                    for cmd_base in commands_base:
                        try:
                            cmd = cmd_base + ['local', query]
                            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                            if result.returncode == 0:
                                results["graphrag_local"] = result.stdout.strip()
                                print("📋 Local query completed")
                                break
                        except:
                            continue
                
            finally:
                os.chdir(original_dir)
            
            # Step 3: Combine results
            results["combined_answer"] = self._create_combined_answer(results)
            return results
            
        except Exception as e:
            print(f"❌ Query error: {e}")
            return {"error": str(e)}
    
    def _create_combined_answer(self, results: Dict) -> str:
        """Create combined answer from all sources"""
        try:
            parts = []
            
            # Azure Search results
            if results["azure_search_results"]:
                parts.append("🔍 **Relevant Documents:**")
                for result in results["azure_search_results"][:2]:
                    parts.append(f"- {result['title']} by {result['author']}")
            
            # GraphRAG responses
            if results["graphrag_global"]:
                parts.append("\n📊 **Global Analysis:**")
                parts.append(results["graphrag_global"])
            
            if results["graphrag_local"]:
                parts.append("\n📋 **Local Details:**")
                parts.append(results["graphrag_local"])
            
            # Fallback to search results
            if not results["graphrag_global"] and not results["graphrag_local"]:
                if results["azure_search_results"]:
                    parts.append("\n📄 **From Search Results:**")
                    for result in results["azure_search_results"][:2]:
                        parts.append(f"**{result['title']}**: {result['content'][:200]}...")
            
            return "\n".join(parts) if parts else "No relevant information found."
            
        except Exception as e:
            return f"Error creating answer: {e}"
    
    def setup_and_run(self, query: str = None):
        """Complete setup and optional query execution with blob storage integration"""
        print("🚀 Setting up Enhanced GraphRAG with Blob Storage Integration...")
        
        # List available PDFs in blob storage
        self.list_available_pdfs_in_blob()
        
        if not self.initialize_graphrag():
            print("❌ Failed to initialize GraphRAG")
            return None
        
        if not self.run_indexing():
            print("❌ Failed to run indexing")
            return None
        
        if query:
            print(f"🔍 Executing query: {query}")
            return self.enhanced_query(query)
        
        print("✓ GraphRAG setup complete and ready for queries")
        return True
    
    def __del__(self):
        """Cleanup temporary files when object is destroyed"""
        if hasattr(self, 'blob_manager'):
            self.blob_manager.cleanup_temp_files()

def main():
    """Enhanced main execution function"""
    
    print("🚀 Enhanced Azure GraphRAG with Full Azure Integration")
    print("=" * 60)
    
    # Step 1: Validate environment variables
    required_env_vars = [
        "AZURE_OPENAI_KEY", "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT_CHAT", "AZURE_OPENAI_DEPLOYMENT_EMBED","AZURE_OPENAI_API_VERSION",
        "AZURE_STORAGE_CONNECTION_STRING", "AZURE_BLOB_CONTAINER_NAME",
        "AZURE_SEARCH_SERVICE_NAME", "AZURE_SEARCH_API_KEY",
        "AZURE_COSMOS_ENDPOINT", "AZURE_COSMOS_KEY"
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these environment variables and try again.")
        return
    
    print("✅ All required environment variables are set")
    
    try:
        # Step 2: Initialize Azure services
        print("\n🔧 Initializing Azure services...")
        azure_manager = AzureServiceManager()
        
        # Step 3: Create search index
        search_manager = AzureSearchManager(azure_manager)
        search_manager.create_search_index()
        
        # Step 4: Initialize enhanced GraphRAG configuration
        print("\n📁 Setting up enhanced GraphRAG configuration...")
        config = EnhancedGraphRAGConfig()
        config.create_project_structure()
        config.create_settings_yaml()
        
        # Step 5: Initialize document processor
        doc_processor = EnhancedDocumentProcessor(config.input_dir, azure_manager)
        
        # Step 6: PDF Processing Menu
        while True:
            print("\n" + "=" * 60)
            print("📄 DOCUMENT PROCESSING MENU")
            print("=" * 60)
            print("1. Upload and process PDF files")
            print("2. List processed documents")
            print("3. Continue to GraphRAG indexing")
            print("4. Skip to querying (if already indexed)")
            print("5. Exit")
            
            choice = input("\nSelect option (1-5): ").strip()
            
            if choice == "1":
                # PDF Upload and Processing
                print("\n📤 PDF Upload and Processing")
                print("-" * 30)
                
                while True:
                    pdf_path = "13-Reasons-Why.pdf"                            # input("Enter PDF file path (or 'back' to return): ").strip()
                    if pdf_path.lower() == 'back':
                        break
                    
                    if not os.path.exists(pdf_path):
                        print("❌ File not found. Please check the path.")
                        continue
                    
                    if not pdf_path.lower().endswith('.pdf'):
                        print("❌ Please provide a PDF file.")
                        continue
                    
                    try:
                        document_id = doc_processor.process_pdf_upload(pdf_path)
                        print(f"✅ PDF processed successfully!")
                        print(f"   Document ID: {document_id}")
                        print(f"   Status: Indexed and ready for GraphRAG")
                        
                        # Ask if user wants to process more files
                        more_files = input("\nProcess another PDF? (y/n): ").strip().lower()
                        if more_files != 'y':
                            break
                            
                    except Exception as e:
                        print(f"❌ Failed to process PDF: {e}")
            
            elif choice == "2":
                # List processed documents
                doc_processor.list_documents()
            
            elif choice == "3":
                # Continue to GraphRAG indexing
                break
            
            elif choice == "4":
                # Skip to querying
                print("⏩ Skipping to querying phase...")
                break
            
            elif choice == "5":
                print("👋 Goodbye!")
                return
            
            else:
                print("❌ Invalid option. Please try again.")
        
        # Step 7: GraphRAG Processing
        if choice == "3":
            print("\n" + "=" * 60)
            print("🤖 GRAPHRAG PROCESSING")
            print("=" * 60)
            
            runner = EnhancedGraphRAGRunner(config.project_root, azure_manager)
            
            # Check if there are documents to process
            documents = doc_processor.list_documents()
            if not documents:
                print("⚠️ No documents found for processing.")
                print("Please upload and process some PDFs first.")
                return
            
            # Initialize GraphRAG
            print("\n🚀 Initializing GraphRAG...")
            if not runner.initialize_graphrag():
                print("❌ Failed to initialize GraphRAG. Please check your configuration.")
                return
            
            # Run indexing
            print("\n🔄 Running enhanced GraphRAG indexing...")
            print("⏳ This may take several minutes depending on document size...")
            if not runner.run_indexing():
                print("❌ Failed to run indexing. Please check your configuration and try again.")
                return
            
            print("✅ GraphRAG indexing completed successfully!")
        
        # Step 8: Enhanced Querying
        print("\n" + "=" * 60)
        print("🎯 ENHANCED QUERYING")
        print("=" * 60)
        
        runner = EnhancedGraphRAGRunner(config.project_root, azure_manager)
        
        # Sample queries based on processed documents
        print("\n🧪 Sample Queries:")
        sample_queries = [
            "What are the main topics discussed in the documents?",
            "Who are the key people mentioned?",
            "What organizations or companies are referenced?",
            "What are the key concepts and their relationships?",
            "Summarize the main findings or conclusions"
        ]
        
        for i, query in enumerate(sample_queries, 1):
            print(f"{i}. {query}")
        
        # Interactive querying
        print("\n🎯 Interactive Query Mode")
        print("Enter your questions (type 'quit' to exit, 'menu' for options)")
        
        while True:
            user_query = input("\n❓ Your question: ").strip()
            if user_query.lower() in ['quit', 'exit', 'q']:
                break
            elif user_query.lower() == 'menu':
                print("\nQuery Options:")
                print("- 'global': High-level analysis across all documents")
                print("- 'local': Specific entity-based search")
                print("- 'both': Combined global and local search (recommended)")
                print("- 'search': Azure AI Search only")
                continue
            elif user_query.lower() == 'samples':
                for i, query in enumerate(sample_queries, 1):
                    print(f"{i}. {query}")
                continue
            
            if not user_query:
                print("Please enter a question.")
                continue
            
            method = input("Choose method (global/local/both/search) [default: both]: ").strip().lower()
            if not method:
                method = "both"
            
            try:
                print("\n🔄 Processing your query...")
                
                if method == "search":
                    # Azure Search only
                    search_results = runner.search_manager.search_documents(user_query, top_k=5)
                    print(f"\n🔍 Azure AI Search Results for: '{user_query}'")
                    print("-" * 50)
                    
                    if search_results:
                        for i, result in enumerate(search_results, 1):
                            print(f"\n{i}. **{result['title']}**")
                            print(f"   Author: {result['author']}")
                            print(f"   Score: {result['score']:.2f}")
                            print(f"   Content Preview: {result['content'][:200]}...")
                    else:
                        print("No relevant documents found.")
                
                else:
                    # Enhanced query with GraphRAG
                    results = runner.enhanced_query(user_query, method)
                    
                    if "error" in results:
                        print(f"❌ Query failed: {results['error']}")
                    else:
                        print(f"\n" + "=" * 50)
                        print(f"🎯 ENHANCED QUERY RESULTS")
                        print(f"Query: {user_query}")
                        print(f"Method: {method}")
                        print(f"Timestamp: {results['timestamp']}")
                        print("=" * 50)
                        
                        if results["combined_answer"]:
                            print(results["combined_answer"])
                        else:
                            print("⚠️ No results generated. Please try a different query.")
                
            except Exception as e:
                print(f"❌ Query processing failed: {e}")
        
        print("\n✅ Enhanced GraphRAG session completed!")
        print(f"📁 Project files saved in: {config.project_root}")
        print("📊 Document metadata stored in Cosmos DB")
        print("🔍 Documents indexed in Azure AI Search")
        print("☁️ PDFs stored in Azure Blob Storage")
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        print(f"❌ Application failed: {e}")

class QuickStartGuide:
    """Quick start guide for setup"""
    
    @staticmethod
    def print_setup_guide():
        print("""
🚀 AZURE GRAPHRAG QUICK START GUIDE
==================================

STEP 1: Install Required Packages
---------------------------------
pip install graphrag PyMuPDF azure-storage-blob azure-search-documents azure-cosmos azure-identity

STEP 2: Set Environment Variables
--------------------------------
# Azure OpenAI
export AZURE_OPENAI_KEY="your-openai-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_DEPLOYMENT_CHAT="gpt-4o"
export AZURE_OPENAI_DEPLOYMENT_EMBED="text-embedding-3-large"

# Azure Storage
export AZURE_STORAGE_CONNECTION_STRING="your-storage-connection-string"
export AZURE_BLOB_CONTAINER_NAME="documents"

# Azure AI Search
export AZURE_SEARCH_SERVICE_NAME="your-search-service"
export AZURE_SEARCH_API_KEY="your-search-api-key"

# Azure Cosmos DB
export AZURE_COSMOS_ENDPOINT="https://your-cosmos.documents.azure.com:443/"
export AZURE_COSMOS_KEY="your-cosmos-key"

STEP 3: Run the Application
--------------------------
python enhanced_azure_graphrag.py

STEP 4: Follow the Interactive Menu
----------------------------------
1. Upload PDF files
2. Process with GraphRAG
3. Query your documents

For detailed setup instructions, visit:
https://docs.microsoft.com/azure/
        """)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        QuickStartGuide.print_setup_guide()
    else:
        main()