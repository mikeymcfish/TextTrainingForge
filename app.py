#!/usr/bin/env python3
import streamlit as st
import json
import os
import io
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading
from utils import process_snippet, test_api_connection, parse_snippets

# Page configuration
st.set_page_config(
    page_title="Story Snippet to Training Data Converter",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'training_data' not in st.session_state:
    st.session_state.training_data = []
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'progress' not in st.session_state:
    st.session_state.progress = 0
if 'total_snippets' not in st.session_state:
    st.session_state.total_snippets = 0
if 'current_snippet' not in st.session_state:
    st.session_state.current_snippet = 0

# Default prompts
DEFAULT_INSTRUCTION_PROMPT = """Below is a short story snippet. Analyze its tone, conflict, characters, and style, then craft a clear one sentence instruction to generate a new scene in the same spirit. Keep it short and to the point.

Examples:
1) "Write a tense exchange between two rivals in a dimly lit alley, focusing on clipped dialogue and shifting power dynamics."
2) "Create a reflective morning scene where a character wrestles with a difficult decision, using vivid sensory details and internal monologue."
3) "Generate an urgent chase through a crowded marketplace, emphasizing fast pacing, narrow escapes, and atmospheric sound cues."

Now, here is your snippet:
---
{snippet}
---

Write one paragraph that captures the mood, conflict, and stylistic direction for the AI. Keep it focused on creativity and inspiration."""

DEFAULT_INPUT_PROMPT = """Below is a short story snippet. Extract only the essential setup that a writer would need to begin crafting a similar scene: approximate ages and genders of the main characters, the setting (time of day, location, era), and the central situation or conflict. Be as brief as possible.

Here is the snippet:
---
{snippet}
---

Provide that context in a single paragraph with no additional commentary."""

def main():
    st.title("📚 Story Snippet to Training Data Converter")
    st.markdown("Convert your story snippets into structured training data for LLM LoRA fine-tuning")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Settings
        st.subheader("API Settings")
        api_url = st.text_input("API URL", value="http://127.0.0.1:5000/v1", help="URL of your local LLM API endpoint")
        model_name = st.text_input("Model Name", value="local-model", help="Name of the model to use")
        temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.8, step=0.1)
        max_tokens = st.number_input("Max Tokens", min_value=1, max_value=1024, value=64)
        
        # Processing Settings
        st.subheader("Processing Settings")
        max_workers = st.slider("Parallel Workers", min_value=1, max_value=10, value=4)
        split_token = st.text_input("Split Token", value="<SPLIT>", help="Token used to split story snippets")
        
        # Test API Connection
        if st.button("🔗 Test API Connection"):
            with st.spinner("Testing connection..."):
                success, message = test_api_connection(api_url, model_name)
                if success:
                    st.success(message)
                else:
                    st.error(message)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input Story Snippets")
        
        # Input method selection
        input_method = st.radio("Choose input method:", ["Upload File", "Paste Text"])
        
        story_text = ""
        if input_method == "Upload File":
            uploaded_file = st.file_uploader("Choose a text file", type=['txt'])
            if uploaded_file is not None:
                story_text = uploaded_file.read().decode('utf-8')
                st.text_area("File Content Preview", story_text[:500] + "..." if len(story_text) > 500 else story_text, height=200, disabled=True)
        else:
            story_text = st.text_area("Paste your story snippets here", height=200, placeholder="Paste your story snippets separated by " + split_token)
        
        # Parse and display snippets
        if story_text:
            snippets = parse_snippets(story_text, split_token)
            st.info(f"Found {len(snippets)} story snippets")
            
            if snippets:
                st.subheader("📋 Snippet Preview")
                for i, snippet in enumerate(snippets[:3]):  # Show first 3 snippets
                    with st.expander(f"Snippet {i+1} ({len(snippet)} characters)"):
                        st.text(snippet[:300] + "..." if len(snippet) > 300 else snippet)
                if len(snippets) > 3:
                    st.info(f"... and {len(snippets) - 3} more snippets")
    
    with col2:
        st.header("🎯 Prompt Templates")
        
        # Editable prompt templates
        st.subheader("Instruction Prompt Template")
        instruction_prompt = st.text_area(
            "Edit the instruction prompt template:",
            value=DEFAULT_INSTRUCTION_PROMPT,
            height=200,
            help="Use {snippet} as placeholder for the story snippet"
        )
        
        st.subheader("Input Prompt Template")
        input_prompt = st.text_area(
            "Edit the input prompt template:",
            value=DEFAULT_INPUT_PROMPT,
            height=150,
            help="Use {snippet} as placeholder for the story snippet"
        )
        
        # Reset prompts button
        if st.button("🔄 Reset to Default Prompts"):
            st.rerun()
    
    # Processing section
    st.header("🚀 Processing")
    
    if story_text and not st.session_state.processing:
        snippets = parse_snippets(story_text, split_token)
        if snippets:
            if st.button("▶️ Start Processing", type="primary"):
                st.session_state.processing = True
                st.session_state.training_data = []
                st.session_state.total_snippets = len(snippets)
                st.session_state.current_snippet = 0
                
                # Start processing in a separate thread
                threading.Thread(
                    target=process_all_snippets,
                    args=(snippets, instruction_prompt, input_prompt, api_url, model_name, temperature, max_tokens, max_workers),
                    daemon=True
                ).start()
                st.rerun()
    
    # Progress display
    if st.session_state.processing:
        st.subheader("⏳ Processing Progress")
        progress_bar = st.progress(st.session_state.progress)
        status_text = st.empty()
        status_text.text(f"Processing snippet {st.session_state.current_snippet}/{st.session_state.total_snippets}")
        
        # Auto-refresh every 2 seconds while processing
        if st.session_state.processing:
            time.sleep(2)
            st.rerun()
    
    # Results section
    if st.session_state.training_data:
        st.header("📊 Generated Training Data")
        
        # Display results summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(st.session_state.training_data))
        with col2:
            avg_instruction_length = sum(len(item['instruction']) for item in st.session_state.training_data) / len(st.session_state.training_data)
            st.metric("Avg Instruction Length", f"{avg_instruction_length:.0f} chars")
        with col3:
            avg_input_length = sum(len(item['input']) for item in st.session_state.training_data) / len(st.session_state.training_data)
            st.metric("Avg Input Length", f"{avg_input_length:.0f} chars")
        
        # Display sample data
        st.subheader("🔍 Sample Training Records")
        for i, record in enumerate(st.session_state.training_data[:3]):
            with st.expander(f"Record {i+1}"):
                st.write("**Instruction:**")
                st.text(record['instruction'])
                st.write("**Input:**")
                st.text(record['input'])
                st.write("**Output:**")
                st.text(record['output'][:200] + "..." if len(record['output']) > 200 else record['output'])
        
        # Download button
        if st.session_state.training_data:
            jsonl_data = "\n".join(json.dumps(record, ensure_ascii=False) for record in st.session_state.training_data)
            st.download_button(
                label="💾 Download Training Data (JSONL)",
                data=jsonl_data,
                file_name="training_data.jsonl",
                mime="application/jsonl"
            )
        
        # Clear results button
        if st.button("🗑️ Clear Results"):
            st.session_state.training_data = []
            st.session_state.processing = False
            st.session_state.progress = 0
            st.session_state.current_snippet = 0
            st.session_state.total_snippets = 0
            st.rerun()

def process_all_snippets(snippets, instruction_prompt, input_prompt, api_url, model_name, temperature, max_tokens, max_workers):
    """Process all snippets in parallel and update session state"""
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(
                    process_snippet,
                    snippet,
                    instruction_prompt,
                    input_prompt,
                    api_url,
                    model_name,
                    temperature,
                    max_tokens
                ): i for i, snippet in enumerate(snippets)
            }
            
            # Process completed tasks
            for future in as_completed(future_to_index):
                try:
                    result = future.result()
                    st.session_state.training_data.append(result)
                    st.session_state.current_snippet += 1
                    st.session_state.progress = st.session_state.current_snippet / st.session_state.total_snippets
                except Exception as e:
                    st.error(f"Error processing snippet: {str(e)}")
        
        st.session_state.processing = False
        
    except Exception as e:
        st.session_state.processing = False
        st.error(f"Processing failed: {str(e)}")

if __name__ == "__main__":
    main()
