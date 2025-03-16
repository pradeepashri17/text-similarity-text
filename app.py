from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import torch

# Initialize FastAPI app
app = FastAPI()

# Load the Sentence Transformer model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Define the request data model
class TextPair(BaseModel):
    text1: str
    text2: str

# Function to compute similarity
def compute_similarity(text1: str, text2: str) -> float:
    """Compute cosine similarity between two texts."""
    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)

    # Normalize embeddings to ensure scores are between 0-1
    embedding1 = torch.nn.functional.normalize(embedding1, p=2, dim=0)
    embedding2 = torch.nn.functional.normalize(embedding2, p=2, dim=0)

    similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
    return round(max(0, similarity), 4)  # Ensure score is between 0-1

# API Endpoint
@app.post("/similarity")
def get_similarity(data: TextPair):
    score = compute_similarity(data.text1, data.text2)
    return {"similarity score": score}

# Run the FastAPI app when executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
