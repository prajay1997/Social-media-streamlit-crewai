# --- SQLite3 Hotfix for Streamlit Sharing ---
# Attempt to override system sqlite3 with pysqlite3-binary
# This must be at the VERY TOP of the file, before any other imports that might use sqlite3
try:
    print("Attempting to apply pysqlite3 hotfix...")
    import pysqlite3
    import sys
    sys.modules["sqlite3"] = pysqlite3
    print("pysqlite3 hotfix applied successfully.")
except ImportError:
    print("pysqlite3 not found, hotfix not applied. Standard sqlite3 will be used.")
except Exception as e:
    print(f"Error applying pysqlite3 hotfix: {e}")
# --- End SQLite3 Hotfix ---

import streamlit as st
import os
import time
from dotenv import load_dotenv
from crewai import Agent, Task, Crew 
from crewai_tools import SerperDevTool 
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
# Moved st.set_page_config to the top level of the script.
st.set_page_config(page_title="Political Leader Sentiment Analysis", layout="wide")

# --- Password Protection ---
def check_password():
    """Returns True if the password is correct, False otherwise."""
    # IMPORTANT: For better security, store your password as a Streamlit Secret
    # Go to your app's settings on share.streamlit.io, then "Secrets"
    # Add a secret like: APP_PASSWORD = "your_chosen_password"
    
    # Retrieve password from Streamlit secrets if available
    try:
        correct_password = st.secrets.get("APP_PASSWORD")
        if not correct_password: # If APP_PASSWORD is not set in secrets
            st.error("App password not configured in Streamlit secrets. Please contact the administrator.")
            return False 
    except Exception as e: # Handles cases where st.secrets might not be available or other errors
        st.error(f"Could not retrieve app password. Error: {e}. Please contact the administrator.")
        # Fallback for local testing if secrets aren't set up (OPTIONAL - REMOVE FOR DEPLOYMENT):
        # st.warning("LOCAL TEST MODE: APP_PASSWORD secret not found. Using fallback 'testpassword'.")
        # correct_password = "testpassword" 
        return False


    # Get password input from the user
    # Using a unique key for the password input field
    password_input = st.text_input("Enter Password to Access:", type="password", key="app_password_input_field")

    if not password_input: # Don't proceed if password input is empty
        # The text_input field itself serves as a prompt.
        # Returning False will prevent the main app from loading.
        return False

    if password_input == correct_password:
        return True
    elif password_input: # If a password was entered but it's incorrect
        st.error("Password incorrect. Please try again.")
        return False
    return False # Should ideally not be reached if logic above is complete

# --- Main App Logic ---
def run_main_app():
    # --- Load API Keys Securely ---
    load_dotenv() 

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    SERPER_API_KEY = os.getenv("SERPER_API_KEY")

    # Set environment variables for CrewAI (and other libraries if needed)
    if OPENAI_API_KEY:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    if SERPER_API_KEY:
        os.environ["SERPER_API_KEY"] = SERPER_API_KEY
    else:
        # This warning will appear in the main app area if the password is correct
        st.warning("SERPER_API_KEY not found in .env file or Streamlit secrets. Search functionality might be limited or fail.")

    search_tool = SerperDevTool()

    # --- LLM Creation Function ---
    def create_llm(use_gpt=True):
        if use_gpt:
            if not OPENAI_API_KEY:
                st.error("OpenAI API Key not found. Please set it in your .env file or Streamlit secrets if you wish to use GPT.")
                return None
            try:
                return ChatOpenAI(model="gpt-4o-mini") 
            except Exception as e:
                st.error(f"Failed to initialize OpenAI GPT: {e}")
                return None
        else:
            st.info("Attempting to connect to Ollama with model 'llama3.1'...")
            try:
                llm_ollama = Ollama(model="llama3.1")
                st.success("Successfully connected to Ollama with model 'llama3.1'.")
                return llm_ollama
            except Exception as e:
                st.error(f"Failed to initialize or connect to Ollama: {e}\n"
                           "Please ensure Ollama server is running and 'llama3.1' model is pulled.")
                return None

    # --- Agent Creation Function ---
    def create_agents(brand_name, llm):
        if not llm:
            st.error("LLM not initialized. Cannot create agents.")
            return None
        
        researcher = Agent(
            role="Political Journey Researcher",
            goal=f"Research and gather comprehensive information about the political journey of {brand_name}, including key milestones, roles, and public statements.",
            backstory="You are an expert political analyst and researcher with a keen eye for detail and unbiased reporting. You excel at finding verified information from diverse sources.",
            verbose=True, allow_delegation=False, tools=[search_tool], llm=llm, max_iter=15
        )
        sentiment_analyzer = Agent(
            role="Public Sentiment Analyzer",
            goal=f"Analyze the public sentiment surrounding {brand_name} based on recent news, social media, and public discourse. Identify key positive, negative, and neutral themes.",
            backstory="You are an expert in natural language processing and sentiment analysis, specializing in political contexts. You can discern nuanced opinions and identify emerging trends.",
            verbose=True, allow_delegation=False, tools=[search_tool], llm=llm, max_iter=15
        )
        report_generator = Agent(
            role="Political Report Synthesizer",
            goal=f"Generate a comprehensive and balanced report on {brand_name}'s political journey and the public sentiment surrounding them, based on the provided research and analysis.",
            backstory="You are a skilled writer and analyst, adept at synthesizing complex information into clear, concise, and insightful reports for political strategists and public understanding.",
            verbose=True, allow_delegation=False, llm=llm, max_iter=15
        )
        return [researcher, sentiment_analyzer, report_generator]

    # --- Task Creation Function ---
    def create_tasks(brand_name, agents):
        if not agents or len(agents) < 3: 
            st.error("Agents not properly initialized. Cannot create tasks.")
            return None
        # Truncated expected_output for brevity in this example
        research_task = Task(
            description=f"Conduct in-depth research on the political career of {brand_name}. Focus on their rise, key positions held, significant policy stances, major achievements, and any notable controversies. Compile a factual overview.",
            agent=agents[0], 
            expected_output=f"A structured factual summary of {brand_name}'s political journey..." 
        )
        sentiment_analysis_task = Task(
            description=f"Analyze current public sentiment towards {brand_name}. Gather information from recent (last 3-6 months) news articles, social media discussions (if accessible via search), and opinion pieces. Categorize sentiment as predominantly positive, negative, or mixed/neutral, and identify the main drivers for these sentiments.",
            agent=agents[1], 
            expected_output=f"A sentiment analysis report for {brand_name} covering..."
        )
        report_generation_task = Task(
            description=f"Compile all gathered information from the research on {brand_name}'s political journey and the sentiment analysis into a single, comprehensive report. The report should be objective, well-structured, and provide actionable insights if possible.",
            agent=agents[2], 
            expected_output=f"A comprehensive report on {brand_name}, structured as follows..."
        )
        return [research_task, sentiment_analysis_task, report_generation_task]

    # --- Main Crew Execution Function ---
    def run_crew_analysis(leader_name, use_gpt_model=True, max_retries=3):
        llm = create_llm(use_gpt=use_gpt_model)
        if not llm: return None
        agents = create_agents(leader_name, llm)
        if not agents: return None
        tasks = create_tasks(leader_name, agents)
        if not tasks: return None
        crew = Crew(agents=agents, tasks=tasks, verbose=1)
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
        return None 

    # --- Streamlit User Interface (inside run_main_app) ---
    # st.set_page_config is now at the top of the script.
    st.title("️Political Leader Journey & Sentiment Analyzer 🕵️‍♂️📊")
    st.markdown("Research a political leader and analyze public sentiment. Enter their name & LLM to start.")

    with st.sidebar:
        st.header("⚙️ Configuration")
        leader_name_input = st.text_input("Enter Political Leader's Name:", placeholder="e.g., Jacinda Ardern")
        llm_option = st.radio(
            "Choose LLM:", 
            ("GPT-4o-mini (OpenAI - Requires API Key)", "Ollama (Local - Llama3.1 - Advanced)")
        )
        use_gpt_selection = True 
        if llm_option == "Ollama (Local - Llama3.1 - Advanced)":
            use_gpt_selection = False
            st.info("Ensure your Ollama server is running locally and 'llama3.1' model is pulled.")
        submit_button = st.button("🚀 Analyze Leader")

    if submit_button and leader_name_input:
        if use_gpt_selection and not OPENAI_API_KEY:
            st.error("OpenAI API Key is missing. Please add it to your .env file or as a Streamlit secret if deployed.")
        elif not SERPER_API_KEY: 
            st.error("Serper API Key (SERPER_API_KEY) is missing. Please add it to your .env file or as a Streamlit secret. This key is required for web searches by the agents.")
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

# --- App Entry Point with Password Check ---
# Initialize session state for password check if not already done
if "password_correct" not in st.session_state:
    st.session_state.password_correct = False

# Check password and run app if correct
# The password check will now happen after st.set_page_config
if not st.session_state.password_correct:
    if check_password(): # This will also handle input and error messages
        st.session_state.password_correct = True
        st.rerun() # Rerun to hide password input and show main app
    else:
        # check_password() returned False.
        # It would have already displayed an error for an incorrect password,
        # or it returned False because the input was empty.
        # The text_input widget (from within check_password) is still visible, prompting the user.
        st.stop() # Halt further execution until password is correct or input is provided.

# Only run the main app if the password has been successfully verified in this session
if st.session_state.password_correct:
    run_main_app()
