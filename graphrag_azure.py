# graphrag_azure.py
"""
GraphRAG-style pipeline using Azure services:
 - downloads PDFs from Azure Blob Storage
 - extracts text (PDF -> chunks)
 - generates embeddings with Azure OpenAI
 - indexes chunks into Azure AI Search (vector index)
 - extracts entities/relations via LLM and stores graph in Cosmos DB (Gremlin/graph)
 - query function: vector retrieval + graph neighbor expansion + final LLM answer
"""

import os
import json
import math
import uuid
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from tqdm import tqdm
from gremlin_python.driver import client, serializer
# Azure blob
from azure.storage.blob import BlobServiceClient

# Azure Search
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchableField,
    SimpleField,
    SearchFieldDataType,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
)
from azure.search.documents import SearchClient, IndexDocumentsBatch

# Azure Cosmos (for graph storage, I'll use the SQL API to store nodes/edges as docs for simplicity,
# but the examples show Gremlin usage. This is easier to run for JSON documents and still lets us do graph-style queries.)
from azure.cosmos import CosmosClient, PartitionKey

# Azure OpenAI via openai package but configured for Azure endpoints
from openai import AzureOpenAI

# PDF parsing
import pypdf

# numpy for embedding array conversion
import numpy as np

# -------------------------
# Load config
# -------------------------
load_dotenv()

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
BLOB_CONTAINER = os.getenv("AZURE_BLOB_CONTAINER_NAME")
BLOB_PREFIX = os.getenv("BLOB_PREFIX", "")

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")  # full url
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX", "graphrag-index")

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_EMBED_MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT_EMBED")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_CHAT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

COSMOS_ENDPOINT = os.getenv("AZURE_COSMOS_ENDPOINT")
COSMOS_KEY = os.getenv("AZURE_COSMOS_KEY")
COSMOS_DATABASE = os.getenv("AZURE_COSMOS_DATABASE_NAME")
COSMOS_CONTAINER = os.getenv("AZURE_COSMOS_CONTAINER_NAME")  # using container to store nodes+edges

LOCAL_TMP_DIR = Path("./tmp")
LOCAL_TMP_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------
# Helpers: blob download
# -------------------------
def download_blobs_to_local() -> List[Path]:
    """Download all blobs (matching prefix) from container to local tmp dir.
    Returns list of local file paths."""
    blob_service_client = BlobServiceClient.from_connection_string(
        AZURE_STORAGE_CONNECTION_STRING
    )
    container_client = blob_service_client.get_container_client(BLOB_CONTAINER)
    blobs = container_client.list_blobs(
        name_starts_with=BLOB_PREFIX if BLOB_PREFIX else None
    )

    local_files = []
    for b in blobs:
        if b.name.endswith("/") or b.name.startswith("$"):  # skip folders/system
            continue
        blob_client = container_client.get_blob_client(b.name)
        local_path = LOCAL_TMP_DIR / Path(b.name).name
        with open(local_path, "wb") as f:
            stream = blob_client.download_blob()
            stream.readinto(f)
        print(f"Downloaded {b.name} -> {local_path}")
        local_files.append(local_path)
    return local_files


# -------------------------
# Fixed: extract text from PDF with page-wise extraction
# -------------------------
def extract_text_from_pdf_pagewise(path: Path) -> List[Dict[str, Any]]:
    """Extract text from PDF page by page using pypdf."""
    reader = pypdf.PdfReader(str(path))
    pages = []
    for page_num, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        if txt.strip():  # Only include pages with content
            pages.append({
                "id": f"{path.stem}_page_{page_num + 1}",
                "text": txt,
                "source": str(path.name),
                "page_info": str(page_num + 1)  # Convert to string for search index
            })
    return pages

# -------------------------
# Fixed: page-wise chunking function
# -------------------------
def chunk_documents_pagewise(
    documents: List[Dict[str, Any]],
    chunk_size: int = 800,
    chunk_overlap: int = 100
) -> List[Dict[str, Any]]:
    """
    Split documents into page-wise chunks while preserving metadata.
    Each document is expected to have: {"id": str, "text": str, "source": str, "page_info": str}.
    If a page's text is longer than `chunk_size`, further split into smaller overlapping chunks.
    """
    all_chunks = []

    for doc in documents:
        text = doc["text"]
        page_info = doc.get("page_info", "1")
        source = doc.get("source", "unknown")
        parent_doc_id = doc["id"]

        # If page text is shorter than chunk_size, keep as single chunk
        if len(text) <= chunk_size:
            all_chunks.append({
                "id": str(uuid.uuid4()),
                "text": text.strip(),
                "source": source,
                "page_info": str(page_info),  # Ensure it's string
                "parent_doc_id": parent_doc_id,
            })
        else:
            # Split long pages into smaller chunks
            start = 0
            text_len = len(text)

            while start < text_len:
                end = min(start + chunk_size, text_len)
                chunk_text = text[start:end]

                # Try not to cut mid-sentence (extend to next period if short extension available)
                if end < text_len:
                    nxt_period = text.find(".", end, min(end + 100, text_len))
                    if nxt_period != -1:
                        chunk_text = text[start : nxt_period + 1]
                        end = nxt_period + 1

                all_chunks.append({
                    "id": str(uuid.uuid4()),   # unique chunk id
                    "text": chunk_text.strip(),
                    "source": source,
                    "page_info": str(page_info),  # Ensure it's string
                    "parent_doc_id": parent_doc_id, # reference back to original doc
                })

                # Move to next chunk with overlap
                start = max(end - chunk_overlap, end)

    return all_chunks

# -------------------------
# Azure OpenAI: embeddings & LLM calls
# -------------------------
def init_openai_client() -> AzureOpenAI:
    print("Using Azure OpenAI endpoint:", AZURE_OPENAI_ENDPOINT)
    print("AZURE_OPENAI_API_KEY:", AZURE_OPENAI_API_KEY)
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        base_url=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
    )
    return client


# -------------------------
# Fixed: Azure OpenAI embeddings function
# -------------------------
def create_embeddings(texts: List[str]) -> List[List[float]]:
    """Create embeddings for a list of texts using Azure OpenAI."""
    vectors = []
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version="2024-12-01-preview",
    )
    embedding_model = os.getenv("AZURE_OPENAI_DEPLOYMENT_EMBED")

    # Process texts in batches to avoid rate limits
    batch_size = 10
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        for text in batch_texts:
            print(f"Processing text of length: {len(text)}")
            response = client.embeddings.create(input=[text], model=embedding_model)
            # Fix: append the embedding array, not extend
            vectors.append(response.data[0].embedding)
            
        # Add small delay to avoid rate limiting
        time.sleep(0.1)

    return vectors

def llm_extract_entities_and_relations(
    openai_client: AzureOpenAI, text_chunk: str
) -> Dict[str, Any]:
    """Use the LLM to extract entities and relations from a chunk. Returns dict with 'entities' and 'relationships'."""
    system_prompt = (
        "You are an information extraction assistant. "
        "Given a passage of text, extract entities (PERSON, ORG, LOCATION, DATE, EVENT, PRODUCT, OTHER) "
        "and relationships between entities in the form (subject, predicate, object). "
        "Return JSON with keys: entities (list of {id, name, type}), relationships (list of {source_id, target_id, relation, evidence})."
    )
    endpoint = "https://rohan-mfs2dn0d-northcentralus.cognitiveservices.azure.com/"
    model_name = "gpt-4o"
    deployment = "gpt-4o"

    subscription_key = "F7m9CCaa1mVkSqFFgjltjFJqquZE77GRS60ZrW7ja9kjJsfIwt84JQQJ99BIACHrzpqXJ3w3AAAAACOGCE09"
    api_version = "2024-12-01-preview"

    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=subscription_key,
    )
    user_prompt = f"Text:\n'''{text_chunk}'''\n\nExtract entities and relations and return only JSON."
    resp = client.chat.completions.create(
        # deployment="gpt-4o",  # ✅ deployment name goes here
        model="gpt-4o",  # ✅ deployment name goes here
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=2000,   # 16384 is too high; many deployments won’t allow it
        temperature=0.3,
    )
    text = resp.choices[0].message.content.strip()
    try:
        parsed = json.loads(text)
        return parsed
    except Exception as e:
        # fallback: try to extract JSON snippet
        import re

        m = re.search(r"(\{.*\})", text, re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except:
                pass
        # if still failing, return empty structure
        return {"entities": [], "relationships": []}


# -------------------------
# Azure Search index creation and indexing
# -------------------------
def create_search_index():
    """Create an Azure Cognitive Search index with vector fields (new SDK style)."""
    admin_credential = AzureKeyCredential(AZURE_SEARCH_API_KEY)
    idx_client = SearchIndexClient(
        endpoint=AZURE_SEARCH_ENDPOINT, credential=admin_credential
    )

    embedding_dimensions = 3072  # adjust depending on your embedding model

    fields = [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
            sortable=True,
        ),
        SearchField(
            name="content",
            type=SearchFieldDataType.String,
            searchable=True,
            analyzer_name="en.microsoft",
        ),
        SimpleField(
            name="source",
            type=SearchFieldDataType.String,
            filterable=True,
            sortable=True,
        ),
        SimpleField(
            name="page_info",
            type=SearchFieldDataType.String,
            filterable=False,
            sortable=False,
        ),
        SearchField(
            name="vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,  # ✅ now supported with vector profile
            filterable=False,
            sortable=False,
            facetable=False,
            retrievable=True,
            vector_search_dimensions=embedding_dimensions,
            vector_search_profile_name="default",  # ✅ must match profile below
        ),
    ]

    # Define vector search
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="default")],
        profiles=[
            VectorSearchProfile(name="default", algorithm_configuration_name="default")
        ],
    )

    index = SearchIndex(
        name=AZURE_SEARCH_INDEX_NAME, fields=fields, vector_search=vector_search
    )

    try:
        idx_client.delete_index(AZURE_SEARCH_INDEX_NAME)
        print(f"Deleted existing index: {AZURE_SEARCH_INDEX_NAME}")
    except Exception:
        pass

    idx_client.create_index(index)
    print("✅ Search index created with vector search:", AZURE_SEARCH_INDEX_NAME)


# -------------------------
# Fixed: indexing function with better error handling
# -------------------------
def index_chunks_to_search(chunks: List[Dict[str, Any]], vectors: List[List[float]]):
    """Index chunk docs into Azure AI Search with vector field 'vector'."""
    credential = AzureKeyCredential(AZURE_SEARCH_API_KEY)
    sclient = SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX_NAME,
        credential=credential,
    )

    documents = []
    for chunk, vec in zip(chunks, vectors):
        doc = {
            "id": chunk["id"],
            "content": chunk["text"],
            "source": chunk.get("source"),
            "page_info": str(chunk.get("page_info", "1")),  # Ensure string type
            "vector": vec,
        }
        documents.append(doc)

    # Index in batches to avoid large payload issues
    batch_size = 50
    total_indexed = 0
    
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        try:
            result = sclient.upload_documents(documents=batch_docs)
            total_indexed += len(batch_docs)
            print(f"Indexed batch {i//batch_size + 1}: {len(batch_docs)} documents")
        except Exception as e:
            print(f"Error indexing batch {i//batch_size + 1}: {str(e)}")
            # Try individual documents in case of batch failure
            for doc in batch_docs:
                try:
                    sclient.upload_documents(documents=[doc])
                    total_indexed += 1
                except Exception as doc_error:
                    print(f"Failed to index document {doc['id']}: {str(doc_error)}")

    print(f"Total indexed {total_indexed} chunks to Azure Search index '{AZURE_SEARCH_INDEX_NAME}'")


# -------------------------
# Cosmos DB helpers (graph storage)
# -------------------------
def init_cosmos():
    print("Using Azure Cosmos Gremlin endpoint:", COSMOS_ENDPOINT)

    gremlin_client = client.Client(
        COSMOS_ENDPOINT,   # e.g. wss://shreeram.gremlin.cosmos.azure.com:443/
        'g',
        username=f"/dbs/{COSMOS_DATABASE}/colls/{COSMOS_CONTAINER}",
        password=COSMOS_KEY,
        message_serializer=serializer.GraphSONSerializersV2d0()
    )

    return gremlin_client


def upsert_entity(gremlin_client, entity: Dict[str, Any]):
    entity_id = entity.get("id")
    entity_name = entity.get("name", "Unnamed")
    entity_type = entity.get("type", "Entity")
    partition_key_value = entity_id  

    query = (
        "g.V().has('id', prop_id).has('partition', prop_partition).fold()"
        ".coalesce(unfold(), addV(entity_type)"
        ".property('id', prop_id)"
        ".property('partition', prop_partition)"
        ".property('name', prop_name))"
    )

    try:
        result = gremlin_client.submit(
            query,
            bindings={
                "entity_type": entity_type,
                "prop_id": entity_id,
                "prop_partition": partition_key_value,
                "prop_name": entity_name,
            },
        ).all().result()
        print(f"Upserted entity: {entity_id}")
        return result
    except Exception as e:
        print(f"Error upserting entity {entity_id}: {e}")
        return None


def upsert_relationship(gremlin_client, rel: Dict[str, Any]):
    """Insert relationship as an edge between two entities in Cosmos Gremlin API."""
    
    source_id = rel.get("source_id")
    target_id = rel.get("target_id")
    relation_type = rel.get("relation", "related_to")
    evidence = rel.get("evidence", "")
    
    if not source_id or not target_id:
        print(f"Warning: Relationship missing source or target ID, skipping: {rel}")
        return
    
    # Generate edge ID if not provided
    edge_id = rel.get("id", str(uuid.uuid4()))
    
    # For edges, the partition key should match the source vertex's partition key
    source_partition = source_id  # Assuming same as entity ID
    target_partition = target_id  # Assuming same as entity ID
    
    query = (
        "g.V().has('id', source_id).has('partition', source_partition)"
        ".addE(relation)"
        ".to(g.V().has('id', target_id).has('partition', target_partition))"
        ".property('id', edge_id)"
        ".property('evidence', evidence)"
    )
    
    try:
        result = gremlin_client.submit(
            query,
            bindings={
                "source_id": source_id,
                "target_id": target_id,
                "source_partition": source_partition,
                "target_partition": target_partition,
                "relation": relation_type,
                "edge_id": edge_id,
                "evidence": evidence,
            },
        ).all().result()
        print(f"Successfully added relationship: {source_id} -> {target_id}")
        return result
    except Exception as e:
        print(f"Error adding relationship {source_id} -> {target_id}: {e}")
        # Don't raise here to allow processing to continue with other relationships
        return None


def build_graph_from_chunks(
    openai_client: AzureOpenAI, gremlin_client, chunks: List[Dict[str, Any]]
):
    """For each chunk, extract entities & relations and persist to Cosmos Gremlin."""
    
    for i, chunk in enumerate(tqdm(chunks, desc="Extract entities")):
        try:
            print(f"Processing chunk {i+1}/{len(chunks)}")
            
            parsed = llm_extract_entities_and_relations(openai_client, chunk["text"])
            entities = parsed.get("entities", [])
            relationships = parsed.get("relationships", [])

            print(f"Found {len(entities)} entities and {len(relationships)} relationships")

            # Process entities first
            successful_entities = set()
            for e in entities:
                # Ensure entity has an ID
                if "id" not in e or not e["id"]:
                    e["id"] = str(
                        uuid.uuid5(uuid.NAMESPACE_OID, e.get("name", "") + str(time.time()))
                    )
                
                try:
                    upsert_entity(gremlin_client, e)
                    successful_entities.add(e["id"])
                except Exception as entity_error:
                    print(f"Failed to add entity {e.get('id', 'unknown')}: {entity_error}")
                    continue

            # Process relationships only for successfully added entities
            for r in relationships:
                if "id" not in r or not r["id"]:
                    r["id"] = str(uuid.uuid4())
                
                # Check if both source and target entities were successfully added
                source_id = r.get("source_id")
                target_id = r.get("target_id")
                
                if source_id in successful_entities and target_id in successful_entities:
                    try:
                        upsert_relationship(gremlin_client, r)
                    except Exception as rel_error:
                        print(f"Failed to add relationship {source_id} -> {target_id}: {rel_error}")
                else:
                    print(f"Skipping relationship {source_id} -> {target_id} - entities not found")
                    
        except Exception as chunk_error:
            print(f"Error processing chunk {i+1}: {chunk_error}")
            continue

    print("Graph build completed and saved to Cosmos Gremlin.")


# Additional helper function to verify partition key configuration
def verify_cosmos_config(gremlin_client):
    """Verify the Cosmos DB configuration and partition key setup."""
    try:
        # Try to get container information
        test_query = "g.V().limit(1)"
        result = gremlin_client.submit(test_query).all().result()
        print("Cosmos DB connection successful")
        return True
    except Exception as e:
        print(f"Cosmos DB connection failed: {e}")
        return False


# Enhanced initialization with configuration check
def init_cosmos_with_validation():
    print("Using Azure Cosmos Gremlin endpoint:", COSMOS_ENDPOINT)

    gremlin_client = client.Client(
        COSMOS_ENDPOINT,
        'g',
        username=f"/dbs/{COSMOS_DATABASE}/colls/{COSMOS_CONTAINER}",
        password=COSMOS_KEY,
        message_serializer=serializer.GraphSONSerializersV2d0()
    )
    
    # Verify connection
    if verify_cosmos_config(gremlin_client):
        print("Cosmos DB Gremlin client initialized successfully")
    else:
        raise Exception("Failed to initialize Cosmos DB Gremlin client")
    
    return gremlin_client

# -------------------------
# Retrieval + graph-based expansion
# -------------------------
def vector_search(
    query: str, openai_client: AzureOpenAI, top_k: int = 5
) -> List[Dict[str, Any]]:
    """Embed query -> ask Azure Search for top_k similar chunks"""
    q_emb = create_embeddings([query])[0]
    credential = AzureKeyCredential(AZURE_SEARCH_API_KEY)
    sclient = SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX_NAME,
        credential=credential,
    )
    # The azure search vector search API format:
    results = sclient.search(
        search_text="*", vector={"value": q_emb, "k": top_k, "fields": "vector"}
    )
    top_chunks = []
    for r in results:
        top_chunks.append(
            {
                "id": r["id"],
                "content": r["content"],
                "score": r.get("@search.score", None),
            }
        )
    return top_chunks


def expand_with_graph(
    container, seed_entity_names: List[str], max_hops: int = 1
) -> Dict[str, Any]:
    """Given a list of entity names (or ids), retrieve neighbor entities and relations from Cosmos container.
    For simplicity we do document queries to find matching entities and edges.
    """
    # find entity docs by name:
    query_entities = []
    for name in seed_entity_names:
        q = f"SELECT * FROM c WHERE c.doc_type='entity' AND CONTAINS(c.name, @name)"
        items = list(
            container.query_items(
                query=q,
                parameters=[{"name": "@name", "value": name}],
                enable_cross_partition_query=True,
            )
        )
        query_entities.extend(items)
    # collect neighbors via edges
    all_entities = {e["id"]: e for e in query_entities}
    edges = []
    for ent in query_entities:
        q_edges = "SELECT * FROM c WHERE c.doc_type='edge' AND (c.source_id=@id OR c.target_id=@id)"
        res_edges = list(
            container.query_items(
                query=q_edges,
                parameters=[{"name": "@id", "value": ent["id"]}],
                enable_cross_partition_query=True,
            )
        )
        for e in res_edges:
            edges.append(e)
            # fetch neighbor entity
            neighbor_id = (
                e["target_id"] if e["source_id"] == ent["id"] else e["source_id"]
            )
            q_nei = "SELECT * FROM c WHERE c.doc_type='entity' AND c.id=@nid"
            neis = list(
                container.query_items(
                    query=q_nei,
                    parameters=[{"name": "@nid", "value": neighbor_id}],
                    enable_cross_partition_query=True,
                )
            )
            for n in neis:
                all_entities[n["id"]] = n
    return {"entities": list(all_entities.values()), "edges": edges}


# -------------------------
# Final answer generation combining contexts
# -------------------------
def generate_answer(
    openai_client: AzureOpenAI,
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    graph_context: Dict[str, Any],
) -> str:
    """Assemble context and ask LLM to answer."""
    system = "You are a helpful domain assistant. Use context and graph info to answer the user query precisely, and cite sources where possible."
    # build context text
    ctx_parts = []
    ctx_parts.append("Top retrieved text chunks:\n")
    for c in retrieved_chunks:
        ctx_parts.append(f"- {c['content']}\n")
    if graph_context:
        ctx_parts.append("\nGraph entities:\n")
        for e in graph_context.get("entities", [])[:20]:
            ctx_parts.append(f"- {e.get('name')} ({e.get('type')})\n")
        ctx_parts.append("\nGraph edges (sample):\n")
        for ed in graph_context.get("edges", [])[:20]:
            ctx_parts.append(
                f"- {ed.get('source_id')} -[{ed.get('relation')}]-> {ed.get('target_id')} (evidence: {ed.get('evidence','')})\n"
            )
    final_prompt = (
        "\n".join(ctx_parts)
        + f"\n\nUser question: {query}\nAnswer concisely using the context above. If not enough info, say 'insufficient info in documents'."
    )
    resp = openai_client.chat.completions.create(
        deployment_id=AZURE_OPENAI_CHAT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": final_prompt},
        ],
        max_tokens=600,
        temperature=0.0,
    )
    return resp.choices[0].message["content"].strip()


# -------------------------
# Updated: main indexing pipeline function
# -------------------------
def run_indexing_pipeline():
    # init clients
    openai_client = init_openai_client()
    
    # 1) download PDFs
    files = download_blobs_to_local()
    all_pages = []
    
    # 2) extract text page-wise from each PDF
    for f in files:
        pages = extract_text_from_pdf_pagewise(f)
        all_pages.extend(pages)
    
    # 3) chunk pages (some pages might be too long)
    all_chunks = chunk_documents_pagewise(all_pages, chunk_size=1200, chunk_overlap=200)
    
    print(f"Total pages extracted: {len(all_pages)}")
    print(f"Total chunks created: {len(all_chunks)}")
    
    # 4) create search index (one-time)
    # create_search_index()
    
    # 5) create embeddings
    # texts = [c["text"] for c in all_chunks]
    # print("Creating embeddings for", len(texts), "chunks...")
    # vectors = create_embeddings(texts)
    
    # print(f"Created {len(vectors)} embeddings")
    
    # # 6) index to Azure Search
    # index_chunks_to_search(all_chunks, vectors)
    
    # 7) build graph & save to Cosmos
    gremlin_client = init_cosmos()
    build_graph_from_chunks(openai_client, gremlin_client, all_chunks)
    
    print("Indexing pipeline finished.")

def run_query_example(question: str):
    openai_client = init_openai_client()
    # vector retrieval
    top_chunks = vector_search(question, openai_client, top_k=6)
    # pick seed entity names heuristically by extracting named entities from the question via LLM
    parsed = llm_extract_entities_and_relations(openai_client, question)
    seed_entities = [e.get("name") for e in parsed.get("entities", []) if e.get("name")]
    cosmos_client, cosmos_db, container = init_cosmos()
    graph_ctx = expand_with_graph(container, seed_entities or [question], max_hops=1)
    answer = generate_answer(openai_client, question, top_chunks, graph_ctx)
    return {"answer": answer, "retrieved": top_chunks, "graph_ctx": graph_ctx}


# -------------------------
# If run as script
# -------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["index", "query"], default="index")
    parser.add_argument("--question", type=str, default="")
    args = parser.parse_args()

    if args.mode == "index":
        run_indexing_pipeline()
    else:
        if not args.question:
            print("Pass --question 'your question' in query mode.")
        else:
            res = run_query_example(args.question)
            print("\n=== ANSWER ===\n")
            print(res["answer"])
            print("\n=== RETRIEVED CHUNKS ===\n")
            for r in res["retrieved"]:
                print("-", r["id"], r["content"][:300].replace("\n", " "))
            print("\n=== GRAPH ENTITIES (sample) ===\n")
            for e in res["graph_ctx"]["entities"][:20]:
                print("-", e.get("id"), e.get("name"), e.get("type"))
