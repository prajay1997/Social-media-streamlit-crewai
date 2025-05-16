import streamlit as st
import os
import time
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama


# --- IMPORTANT: Load API Keys Securely ---
# For local development, load from .env file
load_dotenv() 

# For Streamlit Sharing deployment, you'll set these in Secrets
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Set environment variables for CrewAI (and other libraries if needed)
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY if OPENAI_API_KEY else "YOUR_FALLBACK_OPENAI_KEY_OR_ERROR" # Or handle if None
os.environ["SERPER_API_KEY"] = SERPER_API_KEY if SERPER_API_KEY else "YOUR_FALLBACK_SERPER_KEY_OR_ERROR" # Or handle if None

# --- Your CrewAI Code (adapted from social_media_agent.py) ---
# Note: Removed Colab-specific parts like `from google.colab import userdata`

search_tool = SerperDevTool()


def create_llm(use_gpt=True):
    if use_gpt:
        # The OPENAI_API_KEY is now set as an environment variable globally
        if not OPENAI_API_KEY:
            st.error("OpenAI API Key not found. Please set it in your .env file or Streamlit secrets.")
            return None
        return ChatOpenAI(model="gpt-4o-mini") # API key is picked up from env
    else:
        # Ensure Ollama is configured and running if you use this option
        try:
            return Ollama(model="llama3.1")
        except Exception as e:
            st.error(f"Failed to initialize Ollama: {e}. Make sure Ollama is running and accessible.")
            return None

def create_agents(brand_name, llm):
    if not llm:
        return None # Propagate error if LLM couldn't be created
    researcher = Agent(
        role="Social Media Researcher",
        goal=f"Research and gather information about {brand_name} from various sources",
        backstory="You are an expert researcher with a knack for finding relevant information quickly.",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=llm,
        max_iter=15
    )
    social_media_monitor = Agent(
        role="Social Media Monitor",
        goal=f"Monitor social media platforms for mentions of {brand_name}",
        backstory="You are an experienced social media analyst with keen eyes for trends and mentions.",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=llm,
        max_iter=15
    )
    sentiment_analyzer = Agent(
        role="Sentiment Analyzer",
        goal=f"Analyze the sentiment of social media mentions about {brand_name}",
        backstory="You are an expert in natural language processing and sentiment analysis.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        max_iter=15
    )
    report_generator = Agent(
        role="Report Generator",
        goal=f"Generate comprehensive reports based on the analysis of {brand_name}",
        backstory="You are a skilled data analyst and report writer, adept at presenting insights clearly.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        max_iter=15
    )
    return [researcher, social_media_monitor, sentiment_analyzer, report_generator]

def create_tasks(brand_name, agents):
    if not agents:
        return None
    research_task = Task(
        description=f"Research {brand_name} and provide a summary of their online presence, key information, and recent activities.",
        agent=agents[0],
        expected_output="A structured summary containing: \n1. Brief overview of {brand_name}\n2. Key online platforms and follower counts\n3. Recent notable activities or campaigns\n4. Main products or services\n5. Any recent news or controversies"
    )
    monitoring_task = Task(
        description=f"Monitor social media platforms for mentions of '{brand_name}' in the last 24 hours. Provide a summary of the mentions.",
        agent=agents[1],
        expected_output="A structured report containing: \n1. Total number of mentions\n2. Breakdown by platform (e.g., Twitter, Instagram, Facebook)\n3. Top 5 most engaging posts or mentions\n4. Any trending hashtags associated with {brand_name}\n5. Notable influencers or accounts mentioning {brand_name}"
    )
    sentiment_analysis_task = Task(
        description=f"Analyze the sentiment of the social media mentions about {brand_name}. Categorize them as positive, negative, or neutral.",
        agent=agents[2],
        expected_output="A sentiment analysis report containing: \n1. Overall sentiment distribution (% positive, negative, neutral)\n2. Key positive themes or comments\n3. Key negative themes or comments\n4. Any notable changes in sentiment compared to previous periods\n5. Suggestions for sentiment improvement if necessary"
    )
    report_generation_task = Task(
        description=f"Generate a comprehensive report about {brand_name} based on the research, social media mentions, and sentiment analysis. Include key insights and recommendations.",
        agent=agents[3],
        expected_output="A comprehensive report structured as follows: \n1. Executive Summary\n2. Brand Overview\n3. Social Media Presence Analysis\n4. Sentiment Analysis\n5. Key Insights\n6. Recommendations for Improvement\n7. Conclusion"
    )
    return [research_task, monitoring_task, sentiment_analysis_task, report_generation_task]

def run_social_media_monitoring(brand_name, use_gpt=True, max_retries=3):
    llm = create_llm(use_gpt)
    if not llm:
        st.error("LLM could not be initialized. Cannot proceed.")
        return None
        
    agents = create_agents(brand_name, llm)
    if not agents:
        st.error("Agents could not be created. Cannot proceed.")
        return None

    tasks = create_tasks(brand_name, agents)
    if not tasks:
        st.error("Tasks could not be created. Cannot proceed.")
        return None

    crew = Crew(
        agents=agents,
        tasks=tasks,
        verbose=1 # Can be 0, 1, or 2 for different levels of detail in logs
    )

    for attempt in range(max_retries):
        try:
            with st.spinner(f"CrewAI is working on '{brand_name}'... Attempt {attempt + 1}/{max_retries}"):
                result = crew.kickoff()
            return result
        except Exception as e:
            st.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                st.warning("Retrying...")
                time.sleep(5)
            else:
                st.error("Max retries reached. Unable to complete the task.")
                return None

# --- Streamlit User Interface ---
st.set_page_config(page_title="Social Media Monitoring Crew for Political", layout="wide")

st.title("Social Media Monitoring CrewAI Agent for Research for Poltical Leader 🕵️‍♂️📊")
st.markdown("""
"Research a political leader and analyze public sentiments and generate a report.
Enter a Leader name and choose your LLM to get started.
""")

# Sidebar for inputs
with st.sidebar:
    st.header("⚙️ Configuration")
    brand_name_input = st.text_input("Enter Leader/Influencer Name:", placeholder="e.g., Mamta Banerjee, ")
    
    # Simplified LLM choice for Streamlit
    # You could expand this if you have Ollama easily runnable in your deployment environment
    llm_option = st.radio("Choose LLM:", ("GPT-4o-mini (OpenAI)", "Ollama (Local - Llama3.1 - Advanced)"))
    
    use_gpt_model = True # Default to GPT
    if llm_option == "Ollama (Local - Llama3.1 - Advanced)":
        use_gpt_model = False
        st.info("Ensure your Ollama server is running and accessible if you choose this option.")

    submit_button = st.button("🚀 Analyze Brand")

# Main area for results
if submit_button and brand_name_input:
    if not OPENAI_API_KEY and use_gpt_model:
        st.error("OpenAI API Key is missing. Please add it to your .env file locally, or as a secret if deployed.")
    elif not SERPER_API_KEY:
        st.error("Serper API Key is missing. Please add it to your .env file locally, or as a secret if deployed.")
    else:
        st.info(f"Starting analysis for: **{brand_name_input}** using **{'GPT' if use_gpt_model else 'Ollama'}**...")
        
        # Placeholder for the report
        report_placeholder = st.empty()
        report_placeholder.markdown("### 📝 Generating Report...")

        final_report = run_social_media_monitoring(brand_name_input, use_gpt=use_gpt_model)

        report_placeholder.empty() # Clear the "Generating" message

        if final_report:
            st.subheader("📈 Final Report:")
            st.markdown(final_report) # CrewAI often returns markdown
        else:
            st.error("Failed to generate the report after multiple retries. Please check the logs or try again.")

elif submit_button and not brand_name_input:
    st.warning("Please enter a brand name.")

st.markdown("---")
st.markdown("Powered by CrewAI & Streamlit")