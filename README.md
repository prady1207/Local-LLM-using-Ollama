# Local-LLM-using-Ollama
Practice project to deploy Ollama locally and connect it with APIs that run locally

Step 1: Installed and ran Ollama locally, created a custom model using Modelfile

Step 2: 
1. Used Chromadb to store vector embeddings as context and then queried using ollama and the context.
2. Used FastAPI to creat API endpoints to run queries on Ollama and to add context for multiple users.
