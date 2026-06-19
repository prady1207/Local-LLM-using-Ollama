import ollama
import chromadb
from fastapi import FastAPI

from pydantic import BaseModel
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)

# Creating a document submission model using pydantic
class DocumentSubmission(BaseModel):
    user_id: int
    user_name: str
    content: str

# Creating app using FastAPI
app = FastAPI()

# Saving Data to disk so that it survives restart
client = chromadb.PersistentClient(path="./chromadb")

# Connecting to Ollama Embedding Function to convert text to vectors
em = OllamaEmbeddingFunction(
    model_name = "nomic-embed-text",
    url = "http://localhost:11434", # Ollama's default local address
)

# Create a new collection or reuse an existing collection (when not running for the first time)
collection = client.get_or_create_collection(
    name = "personal_profile",
    embedding_function = em, # Tells Chromadb how to convert text to vectors
)



@app.post("/documents") # creating a POST endpoint that accepts data in the request body
def add_document(submission: DocumentSubmission):
    # Dividing the submission into chunks
    chunks = [chunk.strip() for chunk in submission.content.split("\n\n") if chunk.strip()]

    #Storing each chunk in chromadb collection
    collection.add(
        ids = [f"{submission.user_name}--chunk{i}" for i in range(len(chunks))], # Unique ID per chunk
        documents = chunks, # Actual text contexts
        metadatas = [
            {"source":"profile", "user_name": submission.user_name, "chunk_index": i} for i in range(len(chunks)) # To filter later
        ]
    )

    return {
        "message": f"Added {len(chunks)} chunks for user '{submission.user_name}'.",
        "user_name": submission.user_name,
        "chunks_added": len(chunks),
    }

@app.get("/ask") # Creating a GET endpoint at /ask
def ask(question: str, user: str = None): # FastAPI reads "question" from the URL query string
    # Filtering based on user provided
    query_params = {}
    if user:
        query_params["user_name"] = user # Chormadb metadata filter

    # RETRIEVE - finding the 2 closest chunks in my knowledge base
    results = collection.query(query_texts=[question], n_results=2, where= query_params if query_params else None)
    #Combining chunks into a single string
    context = "\n\n".join(results["documents"][0])

    # AUGEMENT - Using the context retrieved in the step above, we create a prompt that includes the context

    augmented_prompt = f"""Use the following context to answer the question.
If the context doesn't contatin the relevant info, say so.

Context: {context}

Question: {question}
"""
    
    # GENERATE - sending the augemented prompt to the local LLM
    response = ollama.chat(
        model = "qwen2.5:0.5b",
        messages = [{"role": "user", "content": augmented_prompt}],
    )

    # Returning the answer along with the context so that we can verify

    return {
        "question": question,
        "answer": response["message"]["content"],
        "context_used": results["documents"][0],
    }