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
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'tokens_generated' not in st.session_state:
    st.session_state.tokens_generated = 0
if 'current_step' not in st.session_state:
    st.session_state.current_step = ""

# Default prompts
DEFAULT_INSTRUCTION_PROMPT = """You are a creative assistant. When given a story snippet, you will analyze its tone, conflict, characters, and style, and then reply with exactly one sentence: a clear instruction for generating a new scene in the same spirit. Do not include any additional text. Begin your instruction with a verb like "write", "create", or "generate". Avoid using specific character names and keep the instruction general in terms of age, gender, and physical descriptions.

Examples:
1) "Write a tense exchange between two rivals in a dimly lit alley, focusing on clipped dialogue and shifting power dynamics."
2) "Create a reflective morning scene where a character wrestles with a difficult decision, using vivid sensory details and internal monologue."
3) "Generate an urgent chase through a crowded marketplace, emphasizing fast pacing, narrow escapes, and atmospheric sound cues."

Now, here is your snippet:
---
{snippet}
---

###

Here is your one sentence instruction:
Write

"""

DEFAULT_INPUT_PROMPT = """Below is a short story snippet. Extract only the essential setup that a writer would need to begin crafting a similar scene: approximate ages and genders of the main characters, the setting (time of day, location, era), and the central situation or conflict. Be as brief as possible.

Here is the snippet:
---
{snippet}
---

Provide that context in a single paragraph with no additional commentary. 

Here is your one paragraph context:"""

def display_enhanced_progress():
    """Display enhanced progress information"""
    st.subheader("⏳ Processing Progress")
    
    # Main progress bar
    progress_value = st.session_state.progress
    st.progress(progress_value)
    
    # Progress metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Current Snippet", 
            f"{st.session_state.current_snippet}/{st.session_state.total_snippets}",
            delta=None
        )
    
    with col2:
        st.metric(
            "Progress", 
            f"{int(progress_value * 100)}%",
            delta=None
        )
    
    with col3:
        st.metric(
            "Tokens Generated", 
            f"{st.session_state.tokens_generated:,}",
            delta=None
        )
    
    with col4:
        # Calculate time remaining
        if st.session_state.start_time and st.session_state.current_snippet > 0:
            elapsed_time = time.time() - st.session_state.start_time
            avg_time_per_snippet = elapsed_time / st.session_state.current_snippet
            remaining_snippets = st.session_state.total_snippets - st.session_state.current_snippet
            est_remaining_time = avg_time_per_snippet * remaining_snippets
            
            if est_remaining_time < 60:
                time_display = f"{int(est_remaining_time)}s"
            else:
                time_display = f"{int(est_remaining_time/60)}m {int(est_remaining_time%60)}s"
            
            st.metric(
                "Est. Remaining", 
                time_display,
                delta=None
            )
        else:
            st.metric("Est. Remaining", "Calculating...", delta=None)
    
    # Current step indicator
    st.info(f"🔄 {st.session_state.current_step}")
    
    # Detailed timing information
    if st.session_state.start_time:
        elapsed = time.time() - st.session_state.start_time
        elapsed_display = f"{int(elapsed//60)}m {int(elapsed%60)}s" if elapsed >= 60 else f"{int(elapsed)}s"
        st.caption(f"⏱️ Elapsed time: {elapsed_display}")

def main():
    st.title("📚 Story Snippet to Training Data Converter")
    st.markdown("Convert your story snippets into structured training data for LLM LoRA fine-tuning")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Settings
        st.subheader("API Settings")
        api_url = st.text_input("API URL", value="http://0.0.0.0:5000/v1", help="URL of your local LLM API endpoint")
        model_name = st.text_input("Model Name", value="local-model", help="Name of the model to use")
        temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.55, step=0.01)
        max_tokens = st.number_input("Max Tokens", min_value=1, max_value=1024, value=256)
        
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
                import time
                st.session_state.processing = True
                st.session_state.training_data = []
                st.session_state.total_snippets = len(snippets)
                st.session_state.current_snippet = 0
                st.session_state.start_time = time.time()
                st.session_state.tokens_generated = 0
                st.session_state.current_step = "Starting..."
                
                # Process directly in main thread to avoid session state issues
                progress_placeholder = st.empty()
                
                try:
                    # Process snippets one by one
                    for i, snippet in enumerate(snippets):
                        try:
                            st.session_state.current_step = f"Processing snippet {i+1}/{len(snippets)}"
                            st.session_state.current_snippet = i
                            
                            # Update progress display in the same placeholder
                            with progress_placeholder.container():
                                display_enhanced_progress()
                            
                            result = process_snippet(
                                snippet,
                                instruction_prompt,
                                input_prompt,
                                api_url,
                                model_name,
                                temperature,
                                max_tokens
                            )
                            
                            # Count tokens generated
                            tokens_in_result = len(result['instruction'].split()) + len(result['input'].split())
                            st.session_state.tokens_generated += tokens_in_result
                            
                            st.session_state.training_data.append(result)
                            st.session_state.current_snippet = i + 1
                            st.session_state.progress = (i + 1) / len(snippets)
                            
                        except Exception as e:
                            st.error(f"Error processing snippet {i+1}: {str(e)}")
                            st.session_state.current_snippet = i + 1
                            st.session_state.progress = (i + 1) / len(snippets)
                    
                    st.session_state.current_step = "Completed!"
                    # Final progress update
                    with progress_placeholder.container():
                        display_enhanced_progress()
                    
                except Exception as e:
                    st.error(f"Processing failed: {str(e)}")
                
                st.session_state.processing = False
                st.rerun()
    
    # Progress display - this will be shown if processing but not updated by display_enhanced_progress
    if st.session_state.processing and 'start_time' not in st.session_state:
        st.subheader("⏳ Processing Progress")
        st.progress(st.session_state.progress)
        st.text(f"Processing snippet {st.session_state.current_snippet}/{st.session_state.total_snippets}")
    
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

def process_all_snippets_sync(snippets, instruction_prompt, input_prompt, api_url, model_name, temperature, max_tokens):
    """Process all snippets synchronously in main thread"""
    try:
        # Process snippets one by one
        for i, snippet in enumerate(snippets):
            try:
                result = process_snippet(
                    snippet,
                    instruction_prompt,
                    input_prompt,
                    api_url,
                    model_name,
                    temperature,
                    max_tokens
                )
                
                # Update session state
                if 'training_data' not in st.session_state:
                    st.session_state.training_data = []
                
                st.session_state.training_data.append(result)
                st.session_state.current_snippet = i + 1
                st.session_state.progress = (i + 1) / len(snippets)
                
            except Exception as e:
                st.error(f"Error processing snippet {i+1}: {str(e)}")
                # Still increment progress even if failed
                st.session_state.current_snippet = i + 1
                st.session_state.progress = (i + 1) / len(snippets)
        
    except Exception as e:
        st.error(f"Processing failed: {str(e)}")

def process_all_snippets(snippets, instruction_prompt, input_prompt, api_url, model_name, temperature, max_tokens, max_workers):
    """Legacy function - kept for compatibility"""
    process_all_snippets_sync(snippets, instruction_prompt, input_prompt, api_url, model_name, temperature, max_tokens)

if __name__ == "__main__":
    main()
