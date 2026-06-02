from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model=os.getenv("GROQ_MODEL"),
    temperature=float(os.getenv("LLM_TEMPERATURE", 0.1))
)

response = llm.invoke("Say hello in one sentence.")
print("Groq connected successfully.")
print(f"Response: {response.content}")
