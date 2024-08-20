import openai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from pinecone import Pinecone



# Load environment variables from .env.local
load_dotenv('.env.local')

# Retrieve API keys from environment
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

app = FastAPI()

# Model for incoming request
class QueryRequest(BaseModel):
    query: list  # A list of messages (past and present conversation)

# Initialize Pinecone
pinecone_index = Pinecone(api_key=PINECONE_API_KEY).Index("chatbot")

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

@app.get("/")
async def read_root():
    return {"message": "Welcome to the RAG API. Use the /rag/ endpoint with a POST request."}

@app.post("/rag/")
async def perform_rag(request: QueryRequest):
    # Ensure we have messages to work with
    if not request.query or not isinstance(request.query, list):
        raise HTTPException(status_code=400, detail="Invalid input data. 'query' must be a list of messages.")

    # Extract the last user message for embedding
    user_message = request.query[-1]
    if user_message['role'] != 'user':
        raise HTTPException(status_code=400, detail="Last message must be from the user.")

    # Generate embeddings for the latest user query
    raw_query_embedding = openai.embeddings.create(
        input=[user_message['content']],
        model="text-embedding-ada-002"
    )

    query_embedding = raw_query_embedding.data[0].embedding

    # Query Pinecone for relevant documents
    top_matches = pinecone_index.query(
        vector=query_embedding, top_k=10, include_metadata=True, namespace="csmajor"
    )
    #print(top_matches if not None else "nothing here")
    # Extract relevant context from Pinecone results
    contexts = [item['metadata']['text'] for item in top_matches['matches']]
    # Build the augmented query with context
    augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[:10]) + "\n-------\n</CONTEXT>\n\nMY QUESTION:\n" + user_message['content']

    # Define the system's role (AI's persona)
    primer = f"""
    You are an AI computer science advisor. Your role is to recommend classes to students completing the computer science major. Your main task is to:

    
     Provide information on classes, and the computer science major
    
    Answer any questions about majoring or minoring in computer science. Ask if they want general information about the major first before asking if they want specifics. I have given you context for the 
    education requirements and also the period all the needed classes are held(either spring, fall or both) if they need specific information. Give specific classes that can be taken including ones related to the CS major and 
    the other requirements. Ask for their school year and semester if they have not provided it to help you with the search. Ask for the number of credits they want to take for that semester. Remeber for Augustana College you can take 12(3 classes) for 16 credits(4 classes or 5 if two classes are 2 credits each) 
    If they are above freshman year, use the context provided to estimate the classes they have
    already taken given their school year so you will not give recommendations for classes they have already taken. Remember freshmen don't know about the requirements like the number of credits to take. So for freshmen, give them more information. Ask clarifying questions.


    
    Key points to remember:

    Be friendly, patient, and understanding of student concerns

    When interacting with users:

    Greet them politely and ask how you can help
    Clarify their question or issue if needed
    Provide clear, concise answers
    """

    # Call OpenAI API with the full conversation history plus the augmented query
    res = openai.chat.completions.create(
        model="gpt-4o-mini",
        
        messages=[
            {"role": "system", "content": primer},
            *request.query,  # Include the full conversation history
            {"role": "user", "content": augmented_query}  # Include the augmented query as the latest user message
        ]
    )

    # Extract the assistant's response
    openai_answer = res.choices[0].message.content

    return {"response": openai_answer}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
