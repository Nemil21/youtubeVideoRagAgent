import os
import re
import time
import streamlit as st
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from crewai import LLM
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

#solving sqlite error in streamlit
__import__('pysqlite3') 
import sys 
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# --- Page Configuration ---
st.set_page_config(
    page_title="YouTube Video RAG",
    page_icon="🎥",
    layout="wide"
)

# --- Sidebar for API Key ---
with st.sidebar:
    st.header("Configuration")
    
    # Initialize session state for API key if it doesn't exist
    if 'mistral_api_key' not in st.session_state:
        st.session_state.mistral_api_key = ""
    
    # API Key input
    api_key_input = st.text_input(
        "Mistral API Key", 
        value=st.session_state.mistral_api_key,
        type="password",
        help="Enter your Mistral API Key. This will be used instead of any key in your .env file.",
        placeholder="Enter your Mistral API key here"
    )
    
    # Update session state if value changed
    if api_key_input != st.session_state.mistral_api_key:
        st.session_state.mistral_api_key = api_key_input

# --- App Title and Description ---
st.title("YouTube Video RAG")
st.markdown("""
This app analyzes YouTube videos to extract (retrieve) the knowledge provided in the video and compare it with the general knowledge availabe to the LLM (Mistral-large).
Simply paste a YouTube video URL below and click 'Analyze' to get started.
""")

# --- Environment Setup ---
@st.cache_resource
def initialize_environment():
    load_dotenv()
    
    # First check if API key is in session state (from sidebar)
    if st.session_state.mistral_api_key:
        os.environ['MISTRAL_API_KEY'] = st.session_state.mistral_api_key
        return True
    
    # Fall back to checking environment variables
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    
    if mistral_api_key:
        os.environ['MISTRAL_API_KEY'] = mistral_api_key
        return True
    else:
        return False

# --- Define Tools ---
@tool("YouTube Transcript Fetcher Tool")
def youtube_transcript_tool(youtube_url: str) -> str:
    """
    Fetches the ENGLISH transcript for a given YouTube video URL.
    Returns the transcript text clearly marked or a detailed error message.
    Input MUST be a valid YouTube video URL (e.g., https://www.youtube.com/watch?v=...).
    """
    try:
        # Improved regex
        video_id_match = re.search(r"(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([^\"&?\/\s]{11})", youtube_url)
        if not video_id_match:
            return f"Error: Invalid YouTube URL format. Could not extract video ID from '{youtube_url}'."

        video_id = video_id_match.group(1)

        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        try:
            transcript = transcript_list.find_manually_created_transcript(['en'])
        except NoTranscriptFound:
            try:
                transcript = transcript_list.find_generated_transcript(['en'])
            except NoTranscriptFound:
                 return f"Error: No English transcript found for video ID: {video_id}. Transcripts might be disabled or unavailable in English."

        full_transcript = " ".join([item.text for item in transcript.fetch()])
        full_transcript = re.sub(r'\s+', ' ', full_transcript).strip()

        max_length = 20000 # Characters
        if len(full_transcript) > max_length:
           full_transcript = full_transcript[:max_length] + " ... [Transcript Truncated]"

        # Add identifier tags
        return f"TRANSCRIPT_START\n{full_transcript}\nTRANSCRIPT_END"

    except TranscriptsDisabled:
        return f"Error: Transcripts are disabled for video: {youtube_url}"
    except Exception as e:
        import traceback
        return f"Error: Could not process YouTube URL '{youtube_url}'. An unexpected error occurred: {str(e)}"

# --- Initialize LLM ---
@st.cache_resource(show_spinner=False)
def get_llm():
    # Use the most current API key (either from sidebar or environment)
    initialize_environment()
    
    return LLM(
        model="mistral/mistral-large-latest",
        temperature=0.1
    )

# Check for API key and show warning if not provided
if not initialize_environment():
    st.warning("⚠️ No Mistral API key found. Please enter your API key in the sidebar to use this app.")
    if not st.session_state.mistral_api_key:
        st.stop()

# Get LLM instance
llm = get_llm()

# --- Define Agents ---
@st.cache_resource
def create_agents_and_crew():
    transcript_fetcher_agent = Agent(
        role='YouTube Transcript Retrieval Specialist',
        goal='Accurately fetch the complete English transcript of a given YouTube video URL using the provided tool. Report any errors encountered during fetching.',
        backstory=(
            "You are an automated agent focused solely on retrieving data. "
            "Your task is to invoke the 'YouTube Transcript Fetcher Tool' with the provided URL. "
            "You meticulously pass the URL to the tool and return the exact transcript text (between TRANSCRIPT_START and TRANSCRIPT_END) or error message received from the tool. "
            "You do not interpret the transcript content."
        ),
        tools=[youtube_transcript_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    methodology_extractor_agent = Agent(
        role='Technical Content Analyst',
        goal=(
            'Analyze the provided YouTube video transcript text (between TRANSCRIPT_START and TRANSCRIPT_END) to identify the main subject/domain from the transcript'
            'and list the key methodologies, specific techniques, or step-by-step approaches discussed. '
            'Focus *strictly* on information present *within the provided transcript text*. Do not add external knowledge.'
        ),
        backstory=(
            "You are an expert analyst skilled at dissecting technical and instructional text. "
            "You carefully read the provided transcript between 'TRANSCRIPT_START' and 'TRANSCRIPT_END'. "
            "Your primary objective is accuracy and adherence to the source text. "
            "You identify the most specific domain possible based *only* on the transcript content provided to you. "
            "You then list the distinct methods or procedures described *only in the text*, avoiding assumptions or inferring information not explicitly stated. "
            "If the transcript is unclear or lacks specific methodologies, state that clearly in the output."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    methodology_comparer_agent = Agent(
        role='Methodology Comparison Expert',
        goal=(
            'Compare the specific methodologies extracted *from the video transcript* (provided in context) against your general knowledge '
            'of standard or established methodologies in the identified domain. Identify and clearly report differences, '
            'novel approaches, or variations mentioned *explicitly in the extracted methodologies list*. Ground your comparison *strictly* in the provided analysis.'
        ),
        backstory=(
            "You are a knowledgeable expert with a broad understanding of practices across various domains. "
            "You receive an analysis containing a domain and a list of methodologies *extracted directly from a video transcript*. "
            "Your task is to compare *these specific extracted methodologies* to common practices within that domain based on your internal knowledge base. "
            "Focus exclusively on highlighting aspects of the *transcript's methodologies* that differ from, update, or seem novel compared to standard approaches. "
            "Reference the video's methodology (as provided in the input) when explaining a difference. "
            "If no significant differences are found based *only* on the extracted list, state so clearly. "
            "Do *not* introduce methodologies not mentioned in the provided context. Avoid hallucinating differences."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False
    )
    
    # --- Define Tasks ---
    fetch_transcript_task = Task(
        description=(
            'Fetch the English transcript for the YouTube video URL: {youtube_url}. '
            'Use the "YouTube Transcript Fetcher Tool". The output must be the exact transcript text enclosed in TRANSCRIPT_START/END tags or an error message string.'
        ),
        expected_output=(
            'The full English text transcript of the video as a single string, enclosed within "TRANSCRIPT_START" and "TRANSCRIPT_END" tags, '
            'OR a single string starting with "Error:" detailing why the transcript could not be fetched.'
        ),
        agent=transcript_fetcher_agent,
    )

    extract_methodologies_task = Task(
        description=(
            'Carefully analyze the video transcript provided in the context (the text between TRANSCRIPT_START and TRANSCRIPT_END tags). '
            '1. Identify the specific Subject/Domain discussed *only within the transcript text* (be as specific as the text allows). '
            '2. List the key methodologies, techniques, or step-by-step approaches explicitly mentioned or described *only in the transcript text*. '
            'Base your analysis *strictly* on the provided transcript content. Do not invent or infer information.'
        ),
        expected_output=(
            'A structured analysis containing:\n'
            'Domain: [The specific domain identified strictly from the transcript text, or "Not specified" if unclear]\n\n'
            'Methodologies Extracted from Transcript:\n'
            '- [Methodology 1 explicitly mentioned/described in text]\n'
            '- [Methodology 2 explicitly mentioned/described in text]\n'
            '- ... (Include all distinct methods found)\n\n'
            'Note: If no specific domain or methodologies are clearly identifiable *in the text*, state that clearly (e.g., "Methodologies: None clearly specified in transcript").'
        ),
        agent=methodology_extractor_agent,
        context=[fetch_transcript_task]
    )

    compare_methodologies_task = Task(
        description=(
            'You are given the analysis of a video transcript in the context, containing the "Domain" and "Methodologies Extracted from Transcript".\n'
            'Compare the listed "Methodologies Extracted from Transcript" against your general knowledge of standard practices within the identified "Domain".\n'
            'Identify and explain significant differences, novelties, or variations found *specifically in the extracted methodologies list*. '
            'Focus only on comparing what was extracted from the video. Do not add external information or methods not present in the input list.'
        ),
        expected_output=(
            'A comparison report strictly based on the input:\n'
            'Domain: [Domain from context]\n\n'
            'Comparison Summary:\n'
            '[A brief summary comparing ONLY the provided "Methodologies Extracted from Transcript" to standard practices based on your knowledge. State if they align, differ, or present novel aspects.]\n\n'
            'Detailed Differences/Observations (based *only* on comparing the extracted methodologies list to general knowledge):\n'
            '- [Observation about Methodology 1 from the *input list*, e.g., "The transcript\'s extracted mention of \'X technique\' appears novel compared to the standard \'Y method\' because..." OR "The extracted step-by-step process for Z aligns with standard practice."]\n'
            '- [Observation about Methodology 2 from the *input list*...]\n'
            '- ... (Address key extracted methodologies from the input list)\n\n'
            'Note: If the input list contained no clear methodologies, state "No methodologies were provided for comparison". If all methodologies in the input list match standard known practices, state that.'
        ),
        agent=methodology_comparer_agent,
        context=[extract_methodologies_task]
    )

    # --- Define the Crew ---
    youtube_analysis_crew = Crew(
        agents=[transcript_fetcher_agent, methodology_extractor_agent, methodology_comparer_agent],
        tasks=[fetch_transcript_task, extract_methodologies_task, compare_methodologies_task],
        process=Process.sequential,
        verbose=True
    )
    
    return youtube_analysis_crew

# Create the crew
youtube_analysis_crew = create_agents_and_crew()

# --- Extract video ID and get thumbnail ---
def get_video_id(url):
    try:
        video_id_match = re.search(r"(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([^\"&?\/\s]{11})", url)
        if video_id_match:
            return video_id_match.group(1)
    except:
        pass
    return None

def get_thumbnail_url(video_id):
    return f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg"

# Function to parse results into separate components
def parse_crew_result(result):
    try:
        # Convert result to string if it isn't already
        result_str = str(result)
        
        # Attempt to extract transcript
        transcript_text = ""
        if "TRANSCRIPT_START" in result_str and "TRANSCRIPT_END" in result_str:
            transcript_parts = result_str.split("TRANSCRIPT_START")
            if len(transcript_parts) > 1:
                transcript_text = transcript_parts[1].split("TRANSCRIPT_END")[0].strip()
        
        # Look for the Domain section
        domain_match = re.search(r"Domain:(.+?)(?=\n\n|\Z)", result_str, re.DOTALL)
        domain = domain_match.group(1).strip() if domain_match else "Not found in result"
        
        # Extract methodologies analysis
        methodology_analysis = ""
        if "Methodologies Extracted from Transcript:" in result_str:
            methodology_parts = result_str.split("Methodologies Extracted from Transcript:")
            if len(methodology_parts) > 1:
                analysis_part = methodology_parts[1].split("Comparison Summary:")[0]
                methodology_analysis = f"Domain: {domain}\n\nMethodologies Extracted from Transcript:{analysis_part}"
        
        # Extract comparison report
        comparison_report = ""
        if "Comparison Summary:" in result_str:
            comparison_parts = result_str.split("Comparison Summary:")
            if len(comparison_parts) > 1:
                comparison_report = f"Domain: {domain}\n\nComparison Summary:{comparison_parts[1]}"
        
        return {
            "transcript": transcript_text,
            "methodology_analysis": methodology_analysis,
            "comparison_report": comparison_report
        }
    except Exception as e:
        st.error(f"Error parsing results: {str(e)}")
        # For debugging, add the type and raw result
        st.error(f"Result type: {type(result)}")
        return {
            "transcript": "Error extracting transcript",
            "methodology_analysis": "Error extracting methodology analysis",
            "comparison_report": str(result)  # Convert to string for fallback
        }
# --- User Input Section ---
with st.container():
    st.subheader("Enter YouTube Video")
    video_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    
    # Show video thumbnail if URL is valid
    if video_url:
        video_id = get_video_id(video_url)
        if video_id:
            thumbnail_url = get_thumbnail_url(video_id)
            st.image(thumbnail_url, use_container_width=True)
        else:
            st.warning("Please enter a valid YouTube URL")
    
    analyze_button = st.button("Analyze Video", type="primary", disabled=not video_url)

# --- Analysis Process Section ---
if analyze_button:
    # Initialize session state for results if needed
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    # Progress display
    progress_placeholder = st.empty()
    overall_progress = st.progress(0)
    with progress_placeholder.container():
        st.write("Starting analysis process...")
    
    try:
        # Create inputs dictionary
        inputs = {'youtube_url': video_url.strip()}
        
        # The main analysis with progress updates
        phases = [
            "Fetching transcript...",
            "Analyzing methodologies...",
            "Comparing to established practices..."
        ]
        
        # Use a placeholder for status updates
        status_placeholder = st.empty()
        
        # Start the analysis - in real execution this will go through all three phases
        for i, phase in enumerate(phases):
            # Update progress
            progress_value = (i) * 0.33  # 0, 0.33, 0.66
            overall_progress.progress(progress_value)
            
            # Update status message
            with status_placeholder.container():
                st.write(f"Step {i+1}/3: {phase}")
            
            time.sleep(0.5)
            
            # For the first phase (transcript fetch), we'll process with CrewAI
            if i == 0:
                # Use separate spinner for long-running operations
                with st.spinner(phase):
                    # Perform the full analysis with all three agents
                    result = youtube_analysis_crew.kickoff(inputs=inputs)
                    
                    # Store result in session state
                    st.session_state.analysis_results = parse_crew_result(result)
                
                # After first phase, we already have all results (CrewAI runs sequentially)
                # We're just showing progress visually for better UX
        
        # Complete the progress bar
        overall_progress.progress(1.0)
        time.sleep(0.5)
        
        # Clear progress displays
        progress_placeholder.empty()
        overall_progress.empty()
        status_placeholder.empty()
        
        # Display success message
        st.success("Analysis complete!")
        
        # Get parsed results
        results = st.session_state.analysis_results
        
        # Create tabs for displaying results
        tab1, tab2 = st.tabs(["Comparison Report", "Transcript"])
        
        with tab1:
            st.subheader("Comparison Report")
            st.markdown(results["comparison_report"])
            
        with tab2:
            st.subheader("Video Transcript")
            transcript_text = results["transcript"]
            if transcript_text and len(transcript_text) > 0:
                st.text_area("Full Transcript", transcript_text, height=300)
            else:
                # If transcript wasn't properly extracted, show the raw result
                st.text_area("Raw Result", str(results), height=300)
        
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
        import traceback
        st.exception(traceback.format_exc())