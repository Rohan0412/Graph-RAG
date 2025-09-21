"""
graph_rag_azure.py

Requirements:
pip install azure-ai-openai azure-search-documents gremlinpython azure-identity python-dotenv
"""

import os
import math
import uuid
from typing import List, Dict, Any

from dotenv import load_dotenv

# Azure SDKs
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchableField, VectorSearch, VectorSearchAlgorithmConfiguration, SearchField
)
from azure.core.credentials import AzureKeyCredential

# Gremlin client for Cosmos DB (Gremlin API)
from gremlin_python.driver import client as gremlin_client
from gremlin_python.structure.graph import Graph
from gremlin_python.process.graph_traversal import __
from gremlin_python.process.traversal import T

# Load env
load_dotenv()

# ---------- CONFIG ----------
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")  # e.g. https://my-openai.openai.azure.com
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT_EMBED = os.getenv("AZURE_OPENAI_DEPLOYMENT_EMBED")  # embedding deployment name
AZURE_OPENAI_DEPLOYMENT_CHAT = os.getenv("AZURE_OPENAI_DEPLOYMENT_CHAT")  # chat/deploy name for completion
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")  # check your Azure OpenAI API version

SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")  # e.g. https://my-search.search.windows.net
SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX", "rag-index")

COSMOS_GREMLIN_ENDPOINT = os.getenv("COSMOS_GREMLIN_ENDPOINT")  # e.g. wss://<account>.gremlin.cosmos.azure.com:443/
COSMOS_GREMLIN_PRIMARY_KEY = os.getenv("COSMOS_GREMLIN_PRIMARY_KEY")
COSMOS_DATABASE = os.getenv("COSMOS_DB", "graphdb")
COSMOS_GRAPH = os.getenv("COSMOS_GRAPH", "graph")

EMBEDDING_DIM = 1536  # adjust to your Azure OpenAI embedding dim (depends on model)

# ---------- Clients ----------
# ---------- Clients ----------
openai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_OPENAI_API_VERSION,   # check your Azure OpenAI API version
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

search_admin_client = SearchIndexClient(endpoint=SEARCH_ENDPOINT, credential=AzureKeyCredential(SEARCH_KEY))
search_client = SearchClient(endpoint=SEARCH_ENDPOINT, index_name=SEARCH_INDEX_NAME, credential=AzureKeyCredential(SEARCH_KEY))

gremlin = gremlin_client.Client(
    COSMOS_GREMLIN_ENDPOINT,
    'g',
    username="/dbs/{}/colls/{}".format(COSMOS_DATABASE, COSMOS_GRAPH),
    password=COSMOS_GREMLIN_PRIMARY_KEY,
    message_serializer=None
)

# ---------- Helpers ----------
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    N = len(text)
    while start < N:
        end = min(start + chunk_size, N)
        chunks.append(text[start:end])
        start = max(end - overlap, end)
    return chunks

def embed_texts(texts: List[str]) -> List[List[float]]:
    # Ensure all inputs are strings
    clean_texts = [str(t) for t in texts if t and str(t).strip()]

    if not clean_texts:
        return []

    # Call Azure OpenAI embeddings once for all texts
    resp = openai_client.embeddings.create(
        model=AZURE_OPENAI_DEPLOYMENT_EMBED,   # 👈 must be your DEPLOYMENT NAME
        input=clean_texts
    )

    return [d.embedding for d in resp.data]



# ---------- Create Cognitive Search index (vector-enabled) ----------
def create_vector_index():
    """
    Creates a simple index containing:
      - id (key)
      - content (searchable)
      - metadata (fields)
      - embedding vector (vector search)
    """
    try:
        # Vector search algorithm config (cosine)
        vector_search = VectorSearch(
            algorithm_configurations=[
                VectorSearchAlgorithmConfiguration(
                    name="cosinesimil",
                    kind="hnsw"
                )
            ]
        )
        fields = [
            SimpleField(name="id", type="Edm.String", key=True),
            SearchableField(name="content", type="Edm.String", analyzer_name="en.lucene"),
            SearchField(name="title", type="Edm.String"),
            SearchField(name="source", type="Edm.String"),
            SearchField(name="doc_id", type="Edm.String"),
            # Vector field
            SearchField(
                name="embeddingVector",
                type="Collection(Edm.Single)",
                searchable=True,
                filterable=False,
                sortable=False,
                facetable=False,
                vector_search_dimensions=EMBEDDING_DIM
            )
        ]

        index = SearchIndex(name=SEARCH_INDEX_NAME, fields=fields, vector_search=vector_search)
        search_admin_client.create_index(index)
        print("Index created")
    except Exception as e:
        print("Index creation error (it may already exist):", e)

# ---------- Upsert docs to Cognitive Search ----------
def upsert_documents_to_search(docs: List[Dict[str, Any]]):
    """
    docs: list of {id, title, content, source, doc_id, embeddingVector}
    """
    result = search_client.upload_documents(documents=docs)
    print("Upserted to search:", result)

# ---------- Upsert nodes and edges to CosmosDB Gremlin ----------
def upsert_graph_nodes_edges(nodes: List[Dict], edges: List[Dict]):
    """
    nodes: [{id: "node-1", label: "Document", props: {"title": ..., "doc_id": ...}}]
    edges: [{id: "edge-1", outV: "node-1", inV: "node-2", label: "related_to", props: {...}}]
    """
    # Insert nodes
    for n in nodes:
        q = "g.V().has('id','{id}').fold().coalesce(unfold(), addV('{label}').property('id','{id}'))".format(
            id=n['id'], label=n.get('label', 'Node')
        )
        # add properties
        for k, v in n.get("props", {}).items():
            q += ".property('{}','{}')".format(k, str(v).replace("'", "\\'"))
        try:
            gremlin.submit(q).all().result()
        except Exception as e:
            print("Gremlin node insert error:", e)

    # Insert edges
    for e in edges:
        q = ("g.V().has('id','{outV}').as('a').V().has('id','{inV}').as('b')"
             ".coalesce(__.a.outE('{label}').where(inV().has('id','{inV}')), __.addE('{label}').from('a').to('b'))").format(
            outV=e['outV'], inV=e['inV'], label=e.get('label', 'related_to')
        )
        try:
            gremlin.submit(q).all().result()
        except Exception as e:
            print("Gremlin edge insert error:", e)

# ---------- Ingest pipeline ----------
def ingest_documents(documents: List[Dict[str, Any]]):
    """
    documents: [{ 'id': 'doc1', 'title': '...', 'text': 'full document text', 'source': '...'}]
    Steps:
      - chunk each doc
      - embed chunks
      - upsert to Cognitive Search
      - create nodes & edges in Cosmos graph linking doc -> chunk nodes (example)
    """
    docs_for_search = []
    nodes = []
    edges = []

    for doc in documents:
        chunks = chunk_text(doc['text'], chunk_size=800, overlap=200)
        embeddings = embed_texts(chunks)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc['id']}_chunk_{i}"
            docs_for_search.append({
                "id": chunk_id,
                "title": doc.get("title", ""),
                "content": chunk,
                "source": doc.get("source", ""),
                "doc_id": doc['id'],
                "embeddingVector": embeddings[i]
            })
            # graph node for chunk
            nodes.append({"id": chunk_id, "label": "Chunk", "props": {"doc_id": doc['id'], "title": doc.get("title", "")}})
            # edge doc -> chunk
            edges.append({"outV": doc['id'], "inV": chunk_id, "label": "has_chunk"})

        # add doc node
        nodes.append({"id": doc['id'], "label": "Document", "props": {"title": doc.get("title", ""), "source": doc.get("source","")}})

    # create index if not exists (safe to call)
    create_vector_index()

    # upload docs to Cognitive Search
    upsert_documents_to_search(docs_for_search)

    # upsert graph nodes/edges into Gremlin (Cosmos)
    upsert_graph_nodes_edges(nodes, edges)
    print("Ingestion complete.")

# ---------- Query pipeline (Graph RAG) ----------
def vector_search(query: str, top_k: int = 5):
    # create embedding for the query
    embed_resp = openai_client.embeddings.create(model=AZURE_OPENAI_DEPLOYMENT_EMBED, input=query)
    q_emb = embed_resp.data[0].embedding

    # Use Azure Cognitive Search vector search
    vector_query = {
        "vector": {
            "value": q_emb,
            "fields": "embeddingVector",
            "k": top_k,
            "algorithm": "cosinesimil"
        }
    }
    results = search_client.search(
        search_text="*",  # we'll use vector ranking override
        vector=vector_query
    )
    hits = []
    for r in results:
        hits.append({
            "id": r['id'],
            "content": r.get('content'),
            "title": r.get('title'),
            "source": r.get('source'),
            "doc_id": r.get('doc_id')
        })
    return hits

def graph_expand_from_chunks(chunk_ids: List[str], hop: int = 1, max_nodes: int = 20) -> List[Dict]:
    """
    Given chunk node ids, traverse graph to find related nodes (neighbors), using Gremlin.
    Simple example: from chunk -> inV/outV edges -> other chunks/docs.
    """
    neighbors = []
    seen = set()
    # We'll run a traversal for each chunk (could be batched)
    for cid in chunk_ids:
        # Gremlin: g.V().has('id','cid').both( ).limit(max_nodes).valueMap()
        q = f"g.V().has('id','{cid}').both().limit({max_nodes}).valueMap(true)"
        try:
            res = gremlin.submit(q).all().result()
            for item in res:
                # valueMap(true) returns a map with id and label
                if isinstance(item, dict):
                    vid = item.get('id') or item.get('~id') or str(item)
                    if vid in seen: 
                        continue
                    seen.add(vid)
                    # convert map to simpler dict
                    props = {}
                    for k, v in item.items():
                        props[k] = v
                    neighbors.append({"id": vid, "props": props})
        except Exception as e:
            print("Gremlin query error:", e)
    return neighbors

def build_prompt_from_hits_and_graph(query: str, hits: List[Dict], neighbors: List[Dict]) -> str:
    """
    Build a context prompt: include top hits and graph neighbors (deduplicated),
    then ask model to answer the query using that context.
    """
    context_sections = []
    seen_docs = set()
    for h in hits:
        context_sections.append(f"---\nSource: {h.get('source')}\nTitle: {h.get('title')}\nContent:\n{h.get('content')}\n")
        seen_docs.add(h.get('id'))

    for n in neighbors:
        nid = n['id']
        # the properties may have content stored or not; attempt to include text if available
        props = n.get('props', {})
        # if this neighbor is a chunk containing content
        content = None
        if isinstance(props.get('content'), list):
            content = props.get('content')[0]
        else:
            content = props.get('content')
        if content and nid not in seen_docs:
            context_sections.append(f"---\nGraph Node: {nid}\nContent:\n{content}\n")
            seen_docs.add(nid)

    context = "\n\n".join(context_sections)[:15000]  # crop to safe size
    prompt = f"""You are an expert assistant. Use the context below to answer the question. If the answer is not in the context, say you don't know.

CONTEXT:
{context}

QUESTION:
{query}

Answer concisely with references to the sources when possible.
"""
    return prompt

def generate_answer(prompt: str, temperature: float = 0.3, max_tokens: int = 16384):
    response = openai_client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT_CHAT,  # your chat deployment name
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Always include sources."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content

def query_graph_rag(user_query: str, top_k: int = 5):
    hits = vector_search(user_query, top_k=top_k)
    chunk_ids = [h['id'] for h in hits]
    neighbors = graph_expand_from_chunks(chunk_ids, hop=1, max_nodes=20)
    prompt = build_prompt_from_hits_and_graph(user_query, hits, neighbors)
    answer = generate_answer(prompt)
    return {
        "answer": answer,
        "retrieved": hits,
        "neighbors": neighbors
    }

# ---------- Example usage ----------
if __name__ == "__main__":
    # Example ingestion
    docs = [
        {"id": "doc-1", "title": "Azure Cognitive Search Guide", "text": "Azure Cognitive Search is a search-as-a-service ...", "source": "msdocs"},
        {"id": "doc-2", "title": "Cosmos DB Gremlin quickstart", "text": "Cosmos DB Gremlin API supports graph storage ...", "source": "msdocs"},
    ]
    ingest_documents(docs)

    # Query
    q = "How do I set up a vector index in Azure Cognitive Search?"
    res = query_graph_rag(q, top_k=4)
    print("Answer:\n", res['answer'])
    print("Retrieved hits:", [r['id'] for r in res['retrieved']])
    print("Graph neighbors:", [n['id'] for n in res['neighbors']])
