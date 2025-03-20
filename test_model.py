import ollama
import pandas as pd
import json
import re

MODEL_NAME = "deepseek-r1:32b"

def analyze_text(tweet):
    prompt = """please analyse the text provided for offensiveness and hatespeech and return a JSON with the following structure: 
                structure = {
                    "Hate-speech": "boolean",
                    "Offensive": "integer (1-10)",
                    "Targets": ["string"]
                }
                only return the JSON, without any other text or content.
                 
                """ + f"""Provided text: ```{tweet}```"""
    
    structure = {
        "Hate-speech": "boolean",
        "Offensive": "integer (1-10)",
        "Targets": ["string"]
    }

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={"format": "json", "structure": structure}
        )

        raw_response = response.get("message", {}).get("content", "")

        json_match = re.search(r'(\{.*\})', raw_response, re.DOTALL)  

        if json_match:
            return json.loads(json_match.group(0))  # Return the JSON part
        else:
            print()
            return {"error": "No JSON found in response"}
    
    except Exception as e:
        return {"error": f"An error occurred: {e}"}

def process_csv(input_csv, output_json):
    df = pd.read_csv(input_csv)
    count = 0
    analysis_results = []

    with open(output_json, 'w') as f:
        for idx, row in df.iterrows():
            tweet_text = row['tweet']
            result = analyze_text(tweet_text)
            
            analysis_results.append({
                "tweet": tweet_text,
                "analysis": result
            })
            count += 1
            print(f"Processing Tweet: {count}")
        
        json.dump(analysis_results, f, indent=4)

    # Print the total number of processed tweets
    print(f"Processed {count} tweets")

if __name__ == "__main__":
    process_csv('short_labeled_data.csv', 'deepseek_r1_32b_analysis.json')
