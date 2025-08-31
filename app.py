import os
from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
from google.genai import types
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from fastapi import UploadFile, File

load_dotenv()

# Initialize Google Gemini client
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is required")

# Initialize ChromaDB client (NEW API - persistent)
client = chromadb.PersistentClient(path="./chroma_db")

# Create embedding function
gemini_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    api_key=google_api_key,
    model_name="models/embedding-001"
)

# Create or get the collection
collection_name = "documents"
collection = client.get_or_create_collection(
    name=collection_name,
    embedding_function=gemini_ef
)

# Initialize Gemini model for generation
gemini_client = genai.Client(api_key=google_api_key)

# Initialize FastAPI app
app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask(query: Query):
    try:
        # Retrieve relevant documents using Chroma's built-in query
        results = collection.query(
            query_texts=[query.question],
            n_results=3
        )
        
        retrieved_docs = results["documents"][0] if results["documents"] and results["documents"][0] else []
        
        if not retrieved_docs:
            return {"answer": "No relevant context found.", "context": []}
        
        # Construct the prompt
        context = "\n".join(retrieved_docs)
        prompt = f"""Answer the following question based only on the context below:
        
        Context:
        {context}
        
        Question: {query.question}
        
        If the context doesn't contain relevant information, say "I don't have enough information to answer this question."
        """
        
        # Generate the answer using Gemini LLM
        response = gemini_client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        
        return {
            "answer": response.text,
            "context": retrieved_docs
        }
        
    except Exception as e:
        return {"error": str(e), "answer": "An error occurred while processing your request."}

@app.post("/reset")
async def reset():
    global collection  # make sure to update the global variable

    # Delete old collection if exists
    if collection_name in [c.name for c in client.list_collections()]:
        client.delete_collection(collection_name)

    # Recreate collection
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_functions.GeminiEmbeddingFunction()
    )
    return {"status": "Collection reset, all embeddings cleared."}

# Add a simple health check endpoint
@app.get("/")
async def root():
    return {"message": "RAG API is running"}

# Add an endpoint to check collection status
@app.get("/status")
async def status():
    try:
        collections = client.list_collections()
        return {
            "status": "healthy", 
            "collections": [col.name for col in collections],
            "current_collection": collection_name
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    

@app.post("/upload")
async def upload_doc(file: UploadFile = File(...)):
    try:
        # Read file content as text
        content = await file.read()
        text = content.decode("utf-8")
        
        # Add the document to the collection
        collection.add(
            documents=[text],
            metadatas=[{"filename": file.filename}],
            ids=[file.filename]
        )
        return {"status": "success", "filename": file.filename}
    except Exception as e:
        return {"status": "error", "message": str(e)}