import os
from huggingface_hub import InferenceClient
from api.config import settings

client = InferenceClient(
            model=settings.LLM_MODEL_ID,
            token=settings.HF_TOKEN,
            timeout=settings.LLM_TIMEOUT_SECONDS,
        )

def generate_simple_title(query: str, max_length: int = 50) -> str:
    """
    Generate a simple title from the first user query.
    Takes the first sentence or first N characters.
    """
    # Remove extra whitespace
    query = " ".join(query.split())
    
    # Try to get first sentence
    first_sentence = query.split('.')[0].split('?')[0].split('!')[0]
    
    # Truncate if too long
    if len(first_sentence) > max_length:
        title = first_sentence[:max_length].rsplit(' ', 1)[0] + "..."
    else:
        title = first_sentence
    
    return title if title else "New Chat"


def generate_llm_title(query: str, answer: str = None) -> str:
    """
    Use Claude to generate a concise, descriptive title.
    """
    if answer:
        context = f"User query: {query}\n\nAssistant response: {answer[:500]}"
    else:
        context = f"User query: {query}"
    
    prompt = f"""Based on this conversation, generate a short, descriptive title (maximum 6 words).
The title should capture the main topic or question.

{context}

Generate only the title, nothing else. No quotes, no punctuation at the end."""

    try:
        response = client.chat_completion(
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=20,
            temperature=0.2,
        )
        
        title = response.choices[0].message.content.strip().strip('"').strip("'")
        
        # Fallback if title is too long
        if len(title) > 30:
            title = title[:27] + "..."
            
        return title
    
    except Exception as e:
        print(f"Error generating title: {e}")
        return generate_simple_title(query)  # Fallback to simple

