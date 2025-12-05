# **Soft Knowledge Graph (SoftKG) Proof of Concept**

This repository contains a Jupyter Notebook (soft\_kg.ipynb) demonstrating a **Soft Knowledge Graph** architecture. Unlike traditional Knowledge Graphs that rely on rigid Boolean logic and exact string matching, this system utilizes high-dimensional vector embeddings and physics-inspired "tension" to represent and reason about relationships.

## ** Core Concept**

The SoftKG treats knowledge not as a static database of facts, but as a dynamic physical system.

* **Entities** are points in a high-dimensional vector space (initialized via LLM embeddings).  
* **Relations** are transformation functions (permutations \+ projections) that predict where a target entity *should* be.  
* **Truth** is defined by **low tension** (energy). If Head transformed by Relation is close to Tail, the fact is considered "true".

This allows for:

* **Semantic Cold Starts**: New entities have meaningful positions immediately based on their names (e.g., "Vulnerable" is naturally close to "Risk").  
* **Fuzzy Reasoning**: The system can evaluate the likelihood of edges that haven't been explicitly created.  
* **Self-Correction**: Conflicting facts create "tension" in the graph, which is resolved via relaxation algorithms (similar to spring-mass systems).

## **🛠 Tech Stack**

* **DuckDB**: Used as the primary storage engine. It acts as the "Memory" holding the current state of embeddings and relations.  
  * *Extensions*: Uses vss (Vector Similarity Search) for vector operations.  
* **PyTorch**: Used as the "Physics Engine." It calculates tension and performs gradient descent to relax the graph structure.  
* **Sentence Transformers**: Uses BAAI/bge-base-en-v1.5 to initialize entity embeddings semantically.  
* **JupySQL**: Bridges the SQL storage and Python logic.

## ** Architecture Overview**

The notebook implements four distinct layers:

### **1\. Storage Layer (DuckDB)**

Manages three primary tables:

* entities: Stores the current 768-dim vector and version for every node.  
* relations: Stores the **Shuffle Mask** (how the relation permutes dimensions) and **Projection Weights** (feature attention).  
* adjacency: Stores the graph topology (Head \-\> Relation \-\> Tail).

### **2\. Logic Layer (StreamingReshuffleModule)**

A PyTorch nn.Module that defines the physics of the graph. It implements **Automatic Symmetric Tension**:

* **Forward**: Does Head match the shuffled Tail?  
* **Inverse**: Does Tail match the un-shuffled Head?  
* **Projection**: Applies learnable weights to ignore irrelevant dimensions (e.g., "color" doesn't matter for a "is\_employee" relation).

### **3\. Worker Layer (RippleWorker)**

Implements the **Optimistic Ripple Update Algorithm**. When a new fact is added:

1. It fetches the local neighborhood (Head, Tail, and their neighbors).  
2. It runs a localized physics simulation (gradient descent) to minimize tension.  
3. It attempts a CAS (Compare-And-Swap) update to the database.

It also supports **Negative Constraints** (reject\_fact), enabling the system to learn what is *false* by pushing entities apart (Contrastive Loss).

### **4\. Interface Layer (AgentInterface)**

Simulates an API gateway with ingest and query methods, allowing users to feed JSON events into the system.

## ** Getting Started**

### **Prerequisites**

Ensure you have the following Python packages installed:

pip install duckdb torch numpy sentence-transformers jupysql

### **Running the Demo**

1. Open the notebook:  
   jupyter notebook soft\_kg.ipynb

2. Run all cells. The notebook will:  
   * Initialize the in-memory DuckDB instance.  
   * Load the vss extension.  
   * Define the PyTorch models.  
   * Run a simulation where dependencies and risks are ingested.  
   * Query the system to trace the confidence of a specific risk.

## ** Example Output**

The notebook concludes with a demonstration of **Traceability**. When querying a relationship, the system returns:

* **Confidence**: A score derived from the tension (low tension \= high confidence).  
* **Tension**: The raw "energy" error in the relationship.  
* **Sparsity**: How "strict" the rule is (based on projection weights).

{  
    "head": "Service\_A",  
    "relation": "has\_risk",  
    "tail": "High",  
    "result": true,  
    "confidence": 0.98,  
    "tension": 0.04,  
    "sparsity": 0.85  
}

## ** Notes**

This is a research prototype.

* The DuckDB instance is transient (in-memory).  
* Concurrency handling relies on a basic optimistic locking mechanism (version numbers).