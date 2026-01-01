from dotenv import load_dotenv
import os
from openai import OpenAI
import json

load_dotenv()
client = OpenAI(
      api_key=os.environ.get("UNC")
)

# Step 1: Load the prompt
with open("coding_prompt_2.txt", "r", encoding="utf-8") as f:
    prompt_text = f.read().strip() 

# Step 2: Load the papers
with open("papers_original.txt", "r", encoding="utf-8") as f:
    papers_text = f.read().strip()

def build_prompt_with_papers(prompt, papers_text, start_idx=0, end_idx=5):
    """
    Combine prompt with a flexible range of papers.
    
    start_idx: index of first paper (inclusive, 0-based)
    end_idx: index of last paper (exclusive)
    """
    paper_blocks = papers_text.strip().split("\n\n")
    selected_papers = paper_blocks[start_idx:end_idx]
    combined_text = prompt + "\n\n" + "\n\n".join(selected_papers)
    return combined_text

start = 100
end = 1000
batch_size = 5

for i in range(start, end, batch_size):
    batch_start = i
    batch_end = min(i + batch_size, end)
    
    test_input = build_prompt_with_papers(prompt_text, papers_text, start_idx=batch_start, end_idx=batch_end)

    # Send to the model
    response = client.responses.create(
        model="gpt-4.1-mini",  # small GPT-5 variant
        input=test_input,
        temperature=0
    )

    print(f"=== LLM Output for papers {batch_start+1} to {batch_end} ===")
    print(response.output_text)
    print(response)

    with open(f".\\outputs_2\\{batch_start+1}_{batch_end}_coding_output_2.txt", "w", encoding="utf-8") as f:
        f.write(response.output_text)

    with open(f".\\responses_2\\{batch_start+1}_{batch_end}_coding_response_2.txt", "w", encoding="utf-8") as f:
        f.write(str(response))
