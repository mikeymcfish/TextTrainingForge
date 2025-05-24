#!/usr/bin/env python3
import json
import requests
import os
from typing import List, Dict, Tuple

def parse_snippets(text: str, split_token: str) -> List[str]:
    """Parse story text into individual snippets"""
    snippets = [s.strip() for s in text.split(split_token) if s.strip()]
    return snippets

def build_instruction_prompt(snippet: str, template: str) -> str:
    """Build instruction prompt from template and snippet"""
    return template.format(snippet=snippet)

def build_input_prompt(snippet: str, template: str) -> str:
    """Build input prompt from template and snippet"""
    return template.format(snippet=snippet)

def stream_llm_response(prompt: str, api_url: str, model: str, temperature: float, max_tokens: int) -> str:
    """Send request to LLM API and get response"""
    url = f"{api_url}/chat/completions"
    headers = {"Content-Type": "application/json"}
    
    # Check for API key in environment
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, stream=True, timeout=30)
        resp.raise_for_status()

        text = ""
        for line in resp.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            chunk = line[6:].decode("utf-8").strip()
            if chunk == "[DONE]":
                break
            try:
                data = json.loads(chunk)
                delta = data["choices"][0]["delta"].get("content", "")
                if delta:
                    text += delta
            except json.JSONDecodeError:
                continue

        return text.strip()
    
    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed: {str(e)}")
    except Exception as e:
        raise Exception(f"Error processing response: {str(e)}")

def process_snippet(snippet: str, instruction_template: str, input_template: str, 
                   api_url: str, model: str, temperature: float, max_tokens: int) -> Dict[str, str]:
    """Process a single snippet to generate training data"""
    try:
        # Generate instruction
        instruction_prompt = build_instruction_prompt(snippet, instruction_template)
        instruction = stream_llm_response(instruction_prompt, api_url, model, temperature, max_tokens)
        
        # Generate input
        input_prompt = build_input_prompt(snippet, input_template)
        input_text = stream_llm_response(input_prompt, api_url, model, temperature, max_tokens)
        
        return {
            "instruction": instruction,
            "input": input_text,
            "output": snippet
        }
    
    except Exception as e:
        raise Exception(f"Failed to process snippet: {str(e)}")

def test_api_connection(api_url: str, model: str) -> Tuple[bool, str]:
    """Test connection to the LLM API"""
    try:
        url = f"{api_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        
        # Check for API key in environment
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.8,
            "max_tokens": 5,
            "stream": False
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        resp.raise_for_status()
        
        data = resp.json()
        if "choices" in data and len(data["choices"]) > 0:
            return True, "✅ API connection successful!"
        else:
            return False, "❌ API responded but format is unexpected"
            
    except requests.exceptions.ConnectionError:
        return False, "❌ Cannot connect to API. Check if the server is running."
    except requests.exceptions.Timeout:
        return False, "❌ API request timed out"
    except requests.exceptions.HTTPError as e:
        return False, f"❌ HTTP error: {e.response.status_code}"
    except Exception as e:
        return False, f"❌ Connection test failed: {str(e)}"

def validate_prompts(instruction_template: str, input_template: str) -> Tuple[bool, str]:
    """Validate that prompt templates contain the required placeholder"""
    if "{snippet}" not in instruction_template:
        return False, "Instruction template must contain {snippet} placeholder"
    if "{snippet}" not in input_template:
        return False, "Input template must contain {snippet} placeholder"
    return True, "Prompts are valid"

def estimate_processing_time(num_snippets: int, max_workers: int) -> str:
    """Estimate processing time based on number of snippets and workers"""
    # Rough estimate: 2-5 seconds per snippet depending on API response time
    avg_time_per_snippet = 3.5
    total_time = (num_snippets * avg_time_per_snippet) / max_workers
    
    if total_time < 60:
        return f"~{int(total_time)} seconds"
    elif total_time < 3600:
        return f"~{int(total_time/60)} minutes"
    else:
        return f"~{int(total_time/3600)} hours {int((total_time%3600)/60)} minutes"
