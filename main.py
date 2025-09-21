# Azure GraphRAG Implementation - Complete Setup
# This implementation uses Microsoft's GraphRAG with Azure OpenAI services

import os
import asyncio
import yaml
import json
import pandas as pd
from pathlib import Path
import logging
from typing import List, Dict, Any
import subprocess
import sys


# Configuration Class
class AzureGraphRAGConfig:
    def __init__(self):
        self.project_root = Path("./graphrag_project")
        self.input_dir = self.project_root / "input"
        self.output_dir = self.project_root / "output"
        self.settings_file = self.project_root / "settings.yaml"
        # self.env_file = self.project_root / ".env"
        
    def create_project_structure(self):
        """Create the necessary directory structure"""
        self.project_root.mkdir(exist_ok=True)
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        print("✓ Project structure created")
        
        
    def create_settings_yaml(self, 
                           llm_deployment_name: str,
                           embedding_deployment_name: str,
                           azure_endpoint: str,
                           azure_openai_key: str):
        """Create comprehensive settings.yaml configuration"""
        settings = {
            "encoding_model": "cl100k_base",
            "skip_workflows": [],
            "llm": {
                "api_key": azure_openai_key,
                "type": "azure_openai_chat",
                "model": "gpt-4o",
                "model_supports_json": True,
                "api_base": azure_endpoint,
                "api_version": "2024-02-15-preview",
                "deployment_name": llm_deployment_name,
                "max_tokens": 4000,
            },
            # LLM Configuration for Azure OpenAI
            "llm": {
                "api_key": azure_openai_key,
                "type": "azure_openai_chat",
                "model": "gpt-4o",  # or gpt-4, gpt-35-turbo
                "model_supports_json": True,
                "api_base": azure_endpoint,
                "api_version": "2024-02-15-preview",
                "deployment_name": llm_deployment_name,
                "max_tokens": 4000,
                "temperature": 0.1,
                "top_p": 1.0,
                "request_timeout": 180.0,
                "max_retries": 10,
                "max_retry_wait": 10.0,
                "sleep_on_rate_limit_recommendation": True,
                "concurrent_requests": 5
            },
            
            # Embeddings Configuration
            "embeddings": {
                "api_key": azure_openai_key,
                "type": "azure_openai_embedding",
                "model": "text-embedding-3-large",  # or text-embedding-ada-002
                "api_base": azure_endpoint,
                "api_version": "2024-02-15-preview",
                "deployment_name": embedding_deployment_name,
                "max_tokens": 16384,
                "request_timeout": 60.0,
                "max_retries": 10,
                "max_retry_wait": 10.0,
                "sleep_on_rate_limit_recommendation": True,
                "concurrent_requests": 10
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
            
            # Entity Extraction Configuration
            "entity_extraction": {
                "prompt": "prompts/entity_extraction.txt",
                "entity_types": ["PERSON", "ORGANIZATION", "LOCATION", "EVENT", "CONCEPT"],
                "max_gleanings": 1
            },
            
            # Summarize Descriptions Configuration
            "summarize_descriptions": {
                "prompt": "prompts/summarize_descriptions.txt",
                "max_length": 500
            },
            
            # Claim Extraction Configuration
            "claim_extraction": {
                "prompt": "prompts/claim_extraction.txt",
                "description": "Any claims or facts that could be relevant to information discovery.",
                "max_gleanings": 1
            },
            
            # Community Summarization Configuration
            "community_summarization": {
                "prompt": "prompts/community_summarization.txt",
                "max_length": 2000
            },
            
            # Chunk Configuration
            "chunks": {
                "size": 1200,
                "overlap": 100,
                "group_by_columns": ["source"]
            },
            
            # Snapshot Configuration
            "snapshots": {
                "graphml": True,
                "raw_entities": True,
                "top_level_nodes": True
            },
            
            # Umap Configuration
            "umap": {
                "enabled": True
            },
            
            # Workflows
            "workflows": {
                "create_base_text_units": {
                    "chunk": {
                        "size": 1200,
                        "overlap": 100
                    }
                },
                "create_base_extracted_entities": {
                    "entity_extract": {
                        "strategy": {
                            "type": "graph_intelligence",
                            "llm": {
                                "api_key": azure_openai_key,
                                "type": "azure_openai_chat",
                                "model": "gpt-4o",
                                "api_base": azure_endpoint,
                                "api_version": "2024-02-15-preview",
                                "deployment_name": llm_deployment_name
                            }
                        }
                    }
                },
                "create_summarized_entities": {
                    "summarize_descriptions": {
                        "strategy": {
                            "type": "graph_intelligence",
                            "llm": {
                                "api_key": azure_openai_key,
                                "type": "azure_openai_chat",
                                "model": "gpt-4o",
                                "api_base": azure_endpoint,
                                "api_version": "2024-02-15-preview",
                                "deployment_name": llm_deployment_name
                            }
                        }
                    }
                }
            }
        }
        
        with open(self.settings_file, 'w') as f:
            yaml.dump(settings, f, default_flow_style=False, indent=2)
        print("✓ Settings.yaml created with comprehensive configuration")

class DocumentProcessor:
    """Process documents for GraphRAG indexing"""
    
    def __init__(self, input_dir: Path):
        self.input_dir = input_dir
        
    def create_sample_documents(self):
        """Create sample documents for demonstration"""
        documents = {
            "company_overview.txt": """
Microsoft Corporation is a multinational technology company founded by Bill Gates and Paul Allen in 1975. 
The company is headquartered in Redmond, Washington, and is known for developing, licensing, and selling 
computer software, consumer electronics, and personal computers.

Microsoft's flagship products include the Windows operating system, Microsoft Office suite, and Azure cloud services. 
The company has grown through strategic acquisitions, including LinkedIn in 2016 for $26.2 billion and GitHub in 2018 for $7.5 billion.

Key executives include Satya Nadella as CEO since 2014, who has led the company's transformation to cloud computing. 
Other important figures include Brad Smith as President and Amy Hood as CFO.

Microsoft competes with companies like Google, Amazon, and Apple in various technology sectors including cloud computing, 
productivity software, and artificial intelligence.
            """,
            
            "azure_services.txt": """
Microsoft Azure is a cloud computing platform that offers over 200 services across computing, analytics, storage, 
and networking. Azure was launched in 2010 and has become one of the leading cloud platforms globally.

Key Azure services include:
- Azure Virtual Machines for computing
- Azure Storage for data storage
- Azure SQL Database for managed databases
- Azure Active Directory for identity management
- Azure OpenAI Service for AI capabilities
- Azure Cognitive Services for machine learning

Azure operates in over 60 regions worldwide and serves millions of customers. The platform generates significant 
revenue for Microsoft, contributing to the company's growth in recent years.

Azure competes directly with Amazon Web Services (AWS) and Google Cloud Platform (GCP) in the cloud computing market.
            """,
            
            "ai_initiatives.txt": """
Microsoft has made significant investments in artificial intelligence, particularly through its partnership with OpenAI. 
In 2019, Microsoft invested $1 billion in OpenAI, followed by additional investments totaling over $10 billion.

This partnership led to the integration of GPT models into Microsoft products:
- GitHub Copilot for code generation
- Microsoft 365 Copilot for productivity
- Bing Chat for web search
- Azure OpenAI Service for enterprise customers

Microsoft's AI research is led by teams in Redmond and around the world, focusing on areas like natural language processing, 
computer vision, and machine learning. The company has also developed its own AI models and frameworks.

Key AI personalities at Microsoft include Kevin Scott (CTO) and Eric Boyd (Corporate VP of AI Platform). 
The company's AI strategy is closely tied to its cloud computing offerings through Azure.
            """
        }
        
        for filename, content in documents.items():
            file_path = self.input_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content.strip())
        
        print(f"✓ Created {len(documents)} sample documents in {self.input_dir}")
        
    def add_custom_document(self, filename: str, content: str):
        """Add a custom document to the input directory"""
        file_path = self.input_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ Added custom document: {filename}")
        
    def list_documents(self):
        """List all documents in the input directory"""
        documents = list(self.input_dir.glob("*.txt"))
        print(f"Documents in {self.input_dir}:")
        for doc in documents:
            print(f"  - {doc.name}")
        return documents

class GraphRAGRunner:
    """Main class to run GraphRAG operations"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        
    def initialize_graphrag(self):
        """Initialize GraphRAG in the project directory"""
        os.chdir(self.project_root)
        try:
            result = subprocess.run(['python', '-m', 'graphrag.index', '--init', '--root', '.'], 
                                  capture_output=True, text=True, check=True)
            print("✓ GraphRAG initialized successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ GraphRAG initialization failed: {e}")
            print(f"Error output: {e.stderr}")
            return False
        
    def run_indexing(self):
        """Run the GraphRAG indexing process"""
        os.chdir(self.project_root)
        try:
            print("🔄 Starting GraphRAG indexing process (this may take several minutes)...")
            result = subprocess.run(['python', '-m', 'graphrag.index', '--root', '.'], 
                                  capture_output=True, text=True, check=True)
            print("✓ GraphRAG indexing completed successfully")
            print("Index artifacts created in output directory")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ GraphRAG indexing failed: {e}")
            print(f"Error output: {e.stderr}")
            return False
            
    def query_global(self, query: str):
        """Run a global query against the knowledge graph"""
        os.chdir(self.project_root)
        try:
            result = subprocess.run(['python', '-m', 'graphrag.query', 
                                   '--root', '.', '--method', 'global', query], 
                                  capture_output=True, text=True, check=True)
            print(f"🔍 Global Query: {query}")
            print("📊 Response:")
            print(result.stdout)
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"✗ Global query failed: {e}")
            return None
            
    def query_local(self, query: str):
        """Run a local query against the knowledge graph"""
        os.chdir(self.project_root)
        try:
            result = subprocess.run(['python', '-m', 'graphrag.query', 
                                   '--root', '.', '--method', 'local', query], 
                                  capture_output=True, text=True, check=True)
            print(f"🔍 Local Query: {query}")
            print("📋 Response:")
            print(result.stdout)
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"✗ Local query failed: {e}")
            return None

def main():
    """Main execution function"""
    
    # Step 1: Install requirements
    # print("📦 Installing required packages...")
    # install_requirements()  # Uncomment to install packages
    
    # Step 2: Get Azure OpenAI configuration from user
    print("\n🔧 Checking Azure OpenAI Configuration ....")
    # print("Please provide your Azure OpenAI service details:")
    
    azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    llm_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_CHAT")
    embedding_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_EMBED")

    print("\n🔧 Successfully loaded Azure OpenAI Configuration.")

    # Step 3: Initialize configuration
    config = AzureGraphRAGConfig()
    config.create_project_structure()
    # config.create_env_file(azure_openai_key, azure_endpoint)
    config.create_settings_yaml(llm_deployment_name, embedding_deployment_name, azure_endpoint, azure_openai_key)
    
    # Step 4: Process documents
    doc_processor = DocumentProcessor(config.input_dir)
    
    # Create sample documents (you can modify this)
    choice = input("\nCreate sample documents? (y/n): ").strip().lower()
    if choice == 'y':
        doc_processor.create_sample_documents()
    
    # Option to add custom documents
    while True:
        add_custom = input("Add a custom document? (y/n): ").strip().lower()
        if add_custom != 'y':
            break
        filename = input("Enter filename (with .txt extension): ").strip()
        print("Enter document content (press Enter twice to finish):")
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        content = "\n".join(lines)
        doc_processor.add_custom_document(filename, content)
    
    # Step 5: List all documents
    print("\n📄 Document Summary:")
    doc_processor.list_documents()
    
    # Step 6: Run GraphRAG
    runner = GraphRAGRunner(config.project_root)
    
    # Initialize (only needed once)
    print("\n🚀 Initializing GraphRAG...")
    if not runner.initialize_graphrag():
        print("Failed to initialize GraphRAG. Please check your configuration.")
        return
    
    # Run indexing
    print("\n🔄 Running GraphRAG indexing...")
    if not runner.run_indexing():
        print("Failed to run indexing. Please check your configuration and try again.")
        return
    
    # Step 7: Demonstrate querying
    print("\n🎯 GraphRAG is ready! Let's test some queries...")
    
    # Sample queries
    sample_queries = [
        "What is Microsoft's main business?",
        "Who are the key executives at Microsoft?",
        "What are Microsoft's main AI initiatives?",
        "How does Azure compete with other cloud platforms?"
    ]
    
    for query in sample_queries:
        print(f"\n" + "="*50)
        runner.query_global(query)
        print("-"*30)
        runner.query_local(query)
    
    # Interactive querying
    print("\n" + "="*50)
    print("🎯 Interactive Query Mode")
    print("Enter your questions (type 'quit' to exit):")
    
    while True:
        user_query = input("\n❓ Your question: ").strip()
        if user_query.lower() in ['quit', 'exit', 'q']:
            break
            
        method = input("Choose method (global/local/both): ").strip().lower()
        
        if method in ['global', 'both']:
            runner.query_global(user_query)
        if method in ['local', 'both']:
            if method == 'both':
                print("-"*30)
            runner.query_local(user_query)
    
    print("\n✅ GraphRAG session completed!")
    print(f"Project files are saved in: {config.project_root}")

if __name__ == "__main__":
    main()