import streamlit as st
import os
import time
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama # Ensure this is imported

# --- Load API Keys Securely ---
load_dotenv() 

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Set environment variables for CrewAI (and other libraries if needed)
# For OpenAI (if used)
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# For Serper (always needed for search tool)
if SERPER_API_KEY:
    os.environ["SERPER_API_KEY"] = SERPER_API_KEY
else:
    # If Serper API key is critical and not found, you might want to stop or warn.
    # For now, we'll let CrewAI handle it if the tool is used without a key.
    st.warning("SERPER_API_KEY not found in .env file. Search functionality might be limited or fail.")


search_tool = SerperDevTool()

# --- LLM Creation Function ---
def create_llm(use_gpt=True):
    """
    Creates and returns the appropriate LLM instance based on user selection.
    Args:
        use_gpt (bool): If True, initializes OpenAI GPT model. Otherwise, initializes Ollama.
    Returns:
        An LLM instance (ChatOpenAI or Ollama) or None if initialization fails.
    """
    if use_gpt:
        if not OPENAI_API_KEY:
            st.error("OpenAI API Key not found. Please set it in your .env file or Streamlit secrets if you wish to use GPT.")
            return None
        try:
            # Ensure OPENAI_API_KEY is set in environment if ChatOpenAI relies on it implicitly
            # os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY 
            return ChatOpenAI(model="gpt-4o-mini") # API key is often picked up from env
        except Exception as e:
            st.error(f"Failed to initialize OpenAI GPT: {e}")
            return None
    else:
        # --- Ollama Initialization ---
        st.info("Attempting to connect to Ollama with model 'llama3.1'...")
        try:
            # By default, Ollama client connects to http://localhost:11434
            # Ensure your Ollama server is running and the 'llama3.1' model is available.
            # You can pull the model using: `ollama pull llama3.1` in your terminal.
            # If 'llama3.1' is not available, you can try 'llama3' or another model you have.
            llm_ollama = Ollama(model="llama3.1")
            
            # Optional: Add a simple check to see if Ollama is responsive with the model
            # This might be slow, so use with caution or a timeout.
            # For now, we assume successful initialization if no exception is raised.
            # Example check (can be slow, consider removing for faster startup):
            # try:
            #     llm_ollama.invoke("Hi") 
            # except Exception as ollama_test_e:
            #     st.error(f"Ollama initialized but failed to respond with model 'llama3.1'. Is the model pulled and Ollama server healthy? Error: {ollama_test_e}")
            #     return None

            st.success("Successfully connected to Ollama with model 'llama3.1'.")
            return llm_ollama
        except Exception as e:
            st.error(f"Failed to initialize or connect to Ollama: {e}\n"
                       "Please ensure:\n"
                       "1. Ollama is installed and running on your local machine.\n"
                       "2. The Ollama server is accessible (usually at http://localhost:11434).\n"
                       "3. You have pulled the 'llama3.1' model (e.g., run 'ollama pull llama3.1' in your terminal).\n"
                       "4. If 'llama3.1' is not the correct model name, replace it with one you have (e.g., 'llama3').")
            return None

# --- Agent Creation Function ---
def create_agents(brand_name, llm):
    """
    Creates and returns a list of CrewAI agents.
    Args:
        brand_name (str): The name of the brand/leader to research.
        llm: The initialized LLM instance.
    Returns:
        A list of Agent objects or None if LLM is not provided.
    """
    if not llm:
        st.error("LLM not initialized. Cannot create agents.")
        return None
        
    researcher = Agent(
        role="Political Journey Researcher",
        goal=f"Research and gather comprehensive information about the political journey of {brand_name}, including key milestones, roles, and public statements.",
        backstory="You are an expert political analyst and researcher with a keen eye for detail and unbiased reporting. You excel at finding verified information from diverse sources.",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=llm,
        max_iter=15
    )

    sentiment_analyzer = Agent(
        role="Public Sentiment Analyzer",
        goal=f"Analyze the public sentiment surrounding {brand_name} based on recent news, social media, and public discourse. Identify key positive, negative, and neutral themes.",
        backstory="You are an expert in natural language processing and sentiment analysis, specializing in political contexts. You can discern nuanced opinions and identify emerging trends.",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool], # Search tool can be used to find relevant articles/posts for sentiment
        llm=llm,
        max_iter=15
    )

    report_generator = Agent(
        role="Political Report Synthesizer",
        goal=f"Generate a comprehensive and balanced report on {brand_name}'s political journey and the public sentiment surrounding them, based on the provided research and analysis.",
        backstory="You are a skilled writer and analyst, adept at synthesizing complex information into clear, concise, and insightful reports for political strategists and public understanding.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        max_iter=15
    )
    return [researcher, sentiment_analyzer, report_generator]

# --- Task Creation Function ---
def create_tasks(brand_name, agents):
    """
    Creates and returns a list of CrewAI tasks.
    Args:
        brand_name (str): The name of the brand/leader.
        agents (list): A list of initialized Agent objects.
    Returns:
        A list of Task objects or None if agents are not provided.
    """
    if not agents or len(agents) < 3: # Ensure all required agents are present
        st.error("Agents not properly initialized. Cannot create tasks.")
        return None

    research_task = Task(
        description=f"Conduct in-depth research on the political career of {brand_name}. Focus on their rise, key positions held, significant policy stances, major achievements, and any notable controversies. Compile a factual overview.",
        agent=agents[0], # Researcher Agent
        expected_output=f"A structured factual summary of {brand_name}'s political journey, including: \n1. Early political activities and entry into politics.\n2. Chronological list of significant roles and offices held.\n3. Key policy initiatives or legislative work associated with them.\n4. Major achievements and recognitions.\n5. Significant controversies or criticisms faced."
    )

    sentiment_analysis_task = Task(
        description=f"Analyze current public sentiment towards {brand_name}. Gather information from recent (last 3-6 months) news articles, social media discussions (if accessible via search), and opinion pieces. Categorize sentiment as predominantly positive, negative, or mixed/neutral, and identify the main drivers for these sentiments.",
        agent=agents[1], # Sentiment Analyzer Agent
        expected_output=f"A sentiment analysis report for {brand_name} covering: \n1. Overall public sentiment (e.g., Positive, Negative, Neutral, Mixed).\n2. Key themes and topics driving positive sentiment, with examples.\n3. Key themes and topics driving negative sentiment, with examples.\n4. Predominant sentiment on major platforms or in key demographics, if discernible from search results.\n5. Any recent shifts in sentiment and potential causes."
    )

    report_generation_task = Task(
        description=f"Compile all gathered information from the research on {brand_name}'s political journey and the sentiment analysis into a single, comprehensive report. The report should be objective, well-structured, and provide actionable insights if possible.",
        agent=agents[2], # Report Generator Agent
        expected_output=f"A comprehensive report on {brand_name}, structured as follows: \n1. Executive Summary: Brief overview of career and current sentiment.\n2. Political Journey: Detailed factual account based on research.\n3. Public Sentiment Analysis: Summary of findings, key themes, and supporting examples.\n4. Integrated Analysis: Connecting aspects of the political journey with public sentiment (e.g., how past actions influence current views).\n5. Conclusion: Overall summary and potential outlook."
    )
    return [research_task, sentiment_analysis_task, report_generation_task]

# --- Main Crew Execution Function ---
def run_crew_analysis(leader_name, use_gpt_model=True, max_retries=3):
    """
    Initializes and runs the CrewAI analysis.
    Args:
        leader_name (str): The name of the political leader.
        use_gpt_model (bool): Flag to use GPT or Ollama.
        max_retries (int): Number of retries for the crew kickoff.
    Returns:
        The result from crew.kickoff() or None if it fails.
    """
    llm = create_llm(use_gpt=use_gpt_model)
    if not llm:
        # Error already shown by create_llm
        return None
        
    agents = create_agents(leader_name, llm)
    if not agents:
        # Error already shown by create_agents
        return None

    tasks = create_tasks(leader_name, agents)
    if not tasks:
        # Error already shown by create_tasks
        return None

    crew = Crew(
        agents=agents,
        tasks=tasks,
        verbose=1 # 0 for silent, 1 for basic, 2 for detailed crew logs
    )

    for attempt in range(max_retries):
        try:
            with st.spinner(f"CrewAI is analyzing '{leader_name}'... Attempt {attempt + 1}/{max_retries}"):
                result = crew.kickoff()
            return result
        except Exception as e:
            st.error(f"CrewAI analysis attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                st.warning("Retrying...")
                time.sleep(5)
            else:
                st.error("Max retries reached for CrewAI analysis. Unable to complete the task.")
                return None
    return None # Should be unreachable if loop completes

# --- Streamlit User Interface ---
st.set_page_config(page_title="Political Leader Sentiment Analysis", layout="wide")

st.title("️Political Leader Journey & Sentiment Analyzer 🕵️‍♂️📊")
st.markdown("""
Research a political leader and analyze public sentiment. Enter their name & LLM to start.
""") # Updated description

# Sidebar for inputs
with st.sidebar:
    st.header("⚙️ Configuration")
    leader_name_input = st.text_input("Enter Political Leader's Name:", placeholder="e.g., Jacinda Ardern")
    
    llm_option = st.radio(
        "Choose LLM:", 
        ("GPT-4o-mini (OpenAI - Requires API Key)", "Ollama (Local - Llama3.1 - Advanced)")
    )
    
    use_gpt_selection = True # Default to GPT
    if llm_option == "Ollama (Local - Llama3.1 - Advanced)":
        use_gpt_selection = False
        st.info("Ensure your Ollama server is running locally and 'llama3.1' model is pulled.")

    submit_button = st.button("🚀 Analyze Leader")

# Main area for results
if submit_button and leader_name_input:
    # API Key Checks
    if use_gpt_selection and not OPENAI_API_KEY:
        st.error("OpenAI API Key is missing. Please add it to your .env file or as a Streamlit secret if deployed.")
    elif not SERPER_API_KEY: # Serper key is always needed for search
        st.error("Serper API Key (SERPER_API_KEY) is missing. Please add it to your .env file. This key is required for web searches by the agents.")
    else:
        st.info(f"Starting analysis for: **{leader_name_input}** using **{'GPT (OpenAI)' if use_gpt_selection else 'Ollama (Llama3.1)'}**...")
        
        report_placeholder = st.empty()
        report_placeholder.markdown("### 📝 Generating Report...")

        final_report = run_crew_analysis(leader_name_input, use_gpt_model=use_gpt_selection)

        report_placeholder.empty() 

        if final_report:
            st.subheader("📈 Final Report:")
            st.markdown(final_report)
        else:
            st.error("Failed to generate the report. Please check the logs or error messages above and ensure your selected LLM is configured correctly.")

elif submit_button and not leader_name_input:
    st.warning("Please enter a political leader's name.")

st.markdown("---")
st.markdown("Powered by CrewAI & Streamlit")
