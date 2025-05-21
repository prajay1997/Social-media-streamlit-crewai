
try:
    # print("Attempting to apply pysqlite3 hotfix...") # Optional: for local debug
    import pysqlite3
    import sys
    sys.modules["sqlite3"] = pysqlite3
    # print("pysqlite3 hotfix applied successfully.") # Optional: for local debug
except ImportError:
    # print("pysqlite3 not found, hotfix not applied. Standard sqlite3 will be used.") # Optional: for local debug
    pass # Silently pass if not available, standard sqlite3 will be used
except Exception as e:
    # print(f"Error applying pysqlite3 hotfix: {e}") # Optional: for local debug
    pass

import streamlit as st
import os
import time
from datetime import datetime, timedelta, date
from dotenv import load_dotenv # For local .env file loading

from crewai import Agent, Task, Crew, Process

# --- Tool Imports ---
try:
    from langchain_core.tools import BaseTool
except ImportError:
    st.error("CRITICAL ERROR: langchain_core.tools.BaseTool not found. Please ensure 'langchain-core' is installed.")
    BaseTool = object # Fallback to prevent NameErrors, app might not be fully functional

try:
    from crewai_tools import SerperDevTool
except ImportError:
    st.warning("CRITICAL WARNING: crewai_tools or SerperDevTool not found. Web search will NOT function as expected.")
    SerperDevTool = None

# --- LLM Imports ---
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    st.warning("CRITICAL WARNING: langchain_openai not found. OpenAI models will NOT be available unless installed.")
    ChatOpenAI = None

try:
    from langchain_community.llms import Ollama
except ImportError:
    try:
        from langchain.llms import Ollama # Older import path
    except ImportError:
        st.warning("CRITICAL WARNING: langchain_community.llms or langchain.llms with Ollama not found. Ollama models will NOT be available unless installed.")
        Ollama = None

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="PoliSight Analyst Bot", layout="wide", initial_sidebar_state="expanded")

# --- Global variable for initialized search tool ---
# This will be initialized within run_main_app after keys are checked
search_tool_instance = None

# --- Password Protection ---
def check_password():
    """Returns True if the password is correct, False otherwise."""
    try:
        # For deployed apps, use Streamlit secrets:
        # In your Streamlit Cloud dashboard, go to Settings -> Secrets
        # Add a secret: APP_PASSWORD = "your_actual_password"
        correct_password = st.secrets.get("APP_PASSWORD")
        
        # Fallback for local development if st.secrets is not configured or APP_PASSWORD is not set
        if correct_password is None:
            # st.warning("APP_PASSWORD not found in Streamlit secrets. Using fallback password for local testing (REMOVE FOR DEPLOYMENT).")
            correct_password = os.getenv("LOCAL_APP_PASSWORD", "testpassword123") # Example fallback
            if correct_password == "testpassword123" and os.getenv("LOCAL_APP_PASSWORD") is None:
                 st.sidebar.caption("Hint: Default local password is 'testpassword123' or set LOCAL_APP_PASSWORD.")


    except Exception: # Catches if st.secrets itself is not available (e.g., very old Streamlit)
        st.sidebar.error("Could not retrieve app password configuration.")
        correct_password = "testpassword123" # Emergency fallback, should not happen in Cloud
        st.sidebar.caption("Hint: Default local password is 'testpassword123'")


    if not correct_password: # If somehow still None or empty after checks
        st.error("App password configuration is missing. Please contact the administrator.")
        return False

    password_input = st.text_input("Enter Password to Access:", type="password", key="app_password_input_main")

    if not password_input:
        st.info("Please enter the password to proceed.")
        return False

    if password_input == correct_password:
        return True
    elif password_input: # If a password was entered but it's incorrect
        st.error("Password incorrect. Please try again.")
        return False
    return False

# --- Core CrewAI Logic (adapted from your script) ---

# --- Main App Logic ---
def initialize_tools_and_keys():
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


def create_llm_streamlit(use_gpt=True):
    if use_gpt:
        if ChatOpenAI is None:
            st.error("Langchain OpenAI component not loaded. Cannot use GPT.")
            return None
        if not os.getenv("OPENAI_API_KEY"):
            st.error("OpenAI API Key is not configured. Please set it in Streamlit secrets or .env file.")
            return None
        # st.info("Using OpenAI gpt-4o-mini.") # Optional feedback
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.5, openai_api_key=os.getenv("OPENAI_API_KEY"))
    else:
        if Ollama is None:
            st.error("Ollama component not loaded. Cannot use Ollama.")
            return None
        # st.info("Attempting to use Ollama (model: llama3.1).") # Optional feedback
        try:
            llm = Ollama(model="llama3.1")
            # You might add a simple test call to Ollama here if feasible to confirm connection
            # For example: llm.invoke("Hello") 
            # st.success("Ollama initialized (ensure server is running).") # Optional
            return llm
        except Exception as e:
            st.error(f"Failed to initialize Ollama: {e}. Ensure Ollama server is running and model 'llama3.1' is pulled.")
            return None

def create_political_agents_streamlit(political_entity_name, llm):
    global search_tool_instance
    agent_tools_list = []
    if search_tool_instance:
        agent_tools_list.append(search_tool_instance)
    else:
        st.warning("No search tool is available for agents. Web search capabilities will be missing or dummy.")

    activity_researcher = Agent(
        role="Political Affairs & Community Impact Investigator",
        goal=f"Conduct an in-depth investigation into {political_entity_name}'s recent (user-defined analysis period) political activities, "
             f"new schemes, any significant controversies (with their direct and indirect impact on image/perception), "
             f"and the specific, detailed impact of these on local communities or caste groups. "
             f"You MUST collect URLs for all factual claims and key information, providing them ONLY as a consolidated list at the end of your findings for this task.",
        backstory="You are a highly skilled investigative journalist specializing in state-level politics and socio-economic impacts. "
                  "You uncover not just facts, but also their nuanced implications. You are meticulous about verifying information and always provide sources (URLs) for later compilation. "
                  "Your reports on community impact are expected to be substantive.",
        verbose=True, allow_delegation=False, tools=agent_tools_list, llm=llm, max_iter=20
    )

    landscape_monitor = Agent(
        role="State Political Landscape & Competitor Strategy Analyst",
        goal=f"Monitor and deeply analyze the significant activities, underlying STRATEGIES, and political campaigns of ALL major political parties "
             f"and their key leaders in the state (relevant to {political_entity_name}) during the user-defined analysis period. "
             f"Focus on their stated aims for state betterment, their actual methods for increasing vote share (e.g., narratives, target segments), "
             f"and analyze their successes or failures. "
             f"Collect URLs for all reported activities and strategic claims, providing them ONLY as a consolidated list at the end of your findings for this task.",
        backstory="An expert political strategist analyzing state-level competitive dynamics. You don't just list events; you dissect the strategies, "
                  "target audiences, and effectiveness of all major political players. Diligent about sourcing (URLs) for later compilation.",
        verbose=True, allow_delegation=False, tools=agent_tools_list, llm=llm, max_iter=15
    )

    sentiment_analyzer = Agent(
        role="Public Sentiment & Narrative Analyst",
        goal=f"Conduct a DETAILED quantitative and qualitative sentiment analysis for {political_entity_name} for the user-defined analysis period, "
             f"and compare it with sentiment from 5 to 10 days BEFORE the start of that analysis period. "
             f"You MUST output: "
             f"1) Current sentiment breakdown (Positive: X%, Negative: Y%, Neutral: Z%). "
             f"2) Past sentiment breakdown (Positive: A%, Negative: B%, Neutral: C%). "
             f"3) Explicit trend analysis (e.g., 'Negative sentiment surged by K percentage points from B% to Y%'). "
             f"4) Identify 1-2 KEY THEMES or prominent HASHTAGS (e.g., 'MissingCM' if verifiable from data) driving the current sentiment. "
             f"5) Clearly explain reasons for any shifts in simple terms. "
             f"Use the search tool effectively for specific date ranges. Collect any source URLs for sentiment data/reports if applicable, providing them ONLY as a list at the end of your findings.",
        backstory="A specialist in dissecting public opinion. You go beyond surface-level sentiment, providing precise quantitative breakdowns, "
                  "identifying trends, and uncovering the core narratives and events (like specific hashtags or common complaints) that shape public perception.",
        verbose=True, allow_delegation=False, tools=agent_tools_list, llm=llm, max_iter=15
    )

    report_writer = Agent(
        role="Chief Political Strategist & Report Architect",
        goal=f"Compile ALL findings from the other agents into a single, comprehensive, and strategically insightful report "
             f"about {political_entity_name}, following the detailed 7-section political analysis prompt. "
             f"The ENTIRE content of the report MUST be written in **simple, clear English** but must retain the analytical depth and detail requested. "
             f"Develop insightful and actionable 'Strategic Recommendations' (Section VI) with detailed justifications, similar in depth to prior examples of good recommendations. "
             f"Compile ALL unique URLs collected by other agents into a final 'Compiled Reference List' (Section VII) - NO URLs should be inline in Sections I-V.",
        backstory="A seasoned political strategist and master communicator. You transform complex intelligence into clear, actionable advice. "
                  "You excel at crafting comprehensive reports in simple language that directly address every part of a client's request, providing strategic depth and ensuring all sources are meticulously compiled at the end.",
        verbose=True, allow_delegation=False, llm=llm, max_iter=15
    )
    return [activity_researcher, landscape_monitor, sentiment_analyzer, report_writer]

def create_political_tasks_streamlit(political_entity_name, agents, start_date_str, current_date_str, past_sentiment_start_str, past_sentiment_end_str):
    activity_researcher, landscape_monitor, sentiment_analyzer, report_writer = agents

    analysis_period_str = f"between {start_date_str} and {current_date_str}"
    past_sentiment_period_str = f"between {past_sentiment_start_str} and {past_sentiment_end_str}"
    
    search_tool_name_for_prompt = "the available web search tool"
    if search_tool_instance and hasattr(search_tool_instance, 'name'):
        search_tool_name_for_prompt = f"the '{search_tool_instance.name}'"


    date_filter_instruction_main_period = (
        f"Your research MUST focus on information published or relevant strictly {analysis_period_str}. "
        f"When using {search_tool_name_for_prompt}, you MUST craft your queries to find information ONLY from this period. For example, "
        f"if the tool expects a dictionary, provide {{'search_query': '{political_entity_name} news after:{start_date_str} before:{current_date_str}'}}. "
        f"If it takes a string, ensure your query string includes these date restrictions."
    )
    date_filter_instruction_past_sentiment = (
        f"When using {search_tool_name_for_prompt} for past sentiment context, you MUST craft your queries to find information ONLY from the period {past_sentiment_period_str}. "
        f"For example, structure your input to the tool as {{'search_query': '{political_entity_name} public opinion {past_sentiment_period_str}'}} or similar for a string input."
    )
    url_collection_instruction = (
        "You MUST collect specific website URLs for all key facts, activities, and claims. "
        "At the end of your response for THIS TASK, provide a consolidated list of all unique URLs you collected under a clear heading like 'Collected URLs for this task:'."
    )

    task1_research_activities_community = Task(
        description=f"**Primary Goal for {political_entity_name}**: Conduct an in-depth investigation for the period {analysis_period_str}. {url_collection_instruction}\n"
                    f"   **Search Instructions**: {date_filter_instruction_main_period}\n"
                    f"   **Specific Areas to Cover**:\n"
                    f"   1.  **Section I - Recent Political Activities & Scheme Launches**: Detail the nature of each activity (rallies, policy announcements etc.), key messages conveyed, and any new schemes (name, purpose, beneficiaries, potential impact). Be specific and provide details.\n"
                    f"   2.  **Section II - Controversies Involving {political_entity_name}**: Describe each significant controversy, identify key individuals/aspects, analyze media coverage intensity and general tone. Most importantly, provide a detailed analysis of the direct and indirect impact of these controversies on {political_entity_name}'s image and public perception.\n"
                    f"   3.  **Section III - Community and Caste-Specific Impact Analysis**: Provide a substantive analysis (not just a brief mention) if and how specific communities or caste groups are reportedly being affected (positively/negatively) by {political_entity_name}'s recent activities. Also, research if opposition party campaigns are reportedly benefiting or targeting these specific groups. If specific data or strong anecdotal evidence is found, highlight it. If little information is found, state that clearly.",
        agent=activity_researcher,
        expected_output=f"A detailed, multi-part report covering Sections I, II, and III of the main political analysis prompt for {political_entity_name} specifically for the period {analysis_period_str}. "
                        f"The content must be factual, deeply analytical (especially for controversy impact and community effects), and written clearly. "
                        f"The output for this task MUST conclude with a list of all source URLs under the heading 'Collected URLs for this task:'."
    )

    task2_monitor_landscape = Task(
        description=f"**Primary Goal**: Monitor and analyze the broader political landscape in the state of {political_entity_name} for the period {analysis_period_str}. {url_collection_instruction}\n"
                    f"   **Search Instructions**: {date_filter_instruction_main_period}\n"
                    f"   **Specific Areas to Cover (Section V of final report)**:\n"
                    f"   1. Report on significant activities of **ALL major political parties** and their key leaders in the state.\n"
                    f"   2. For EACH major party, provide a detailed analysis of their: \n"
                    f"      a. Recent public engagements, announcements, or policy stances.\n"
                    f"      b. **Underlying STRATEGIES and specific CAMPAIGNS** they are visibly adopting/running aimed at i) the betterment of the state (e.g., development initiatives, governance reforms) AND ii) increasing their vote share or public support (e.g., outreach programs, narrative building, target voter segments, key issues highlighted).\n"
                    f"      c. Key messages being pushed to the public.\n"
                    f"      d. Any notable successes or failures in their recent strategic efforts.",
        agent=landscape_monitor,
        expected_output=f"A comprehensive and strategic analysis of the activities and positioning of ALL major political players in the state for {analysis_period_str}, aligning with Section V of the main political analysis prompt. "
                        f"The analysis for each party should go beyond listing events and delve into their strategies and campaign effectiveness. "
                        f"The output for this task MUST conclude with a list of all source URLs under the heading 'Collected URLs for this task:'."
    )

    task3_analyze_sentiment = Task(
        description=f"**Primary Goal**: Conduct a DETAILED public sentiment analysis for {political_entity_name}.\n"
                    f"   **Current Sentiment Period**: Analyze sentiment for **{analysis_period_str}**. {url_collection_instruction.replace('key facts, activities, and claims', 'articles or data points specifically used for sentiment context during this period')}\n"
                    f"   **Past Sentiment Period for Comparison**: Analyze sentiment for **{past_sentiment_period_str}** (5-10 days before {start_date_str}). {date_filter_instruction_past_sentiment} {url_collection_instruction.replace('key facts, activities, and claims', 'articles or data points specifically used for sentiment context during this past period')}\n"
                    f"   **Required Output Details (Section IV of final report)**:\n"
                    f"   1.  **Current Sentiment ({analysis_period_str})**: Provide breakdown: Positive: X%, Negative: Y%, Neutral: Z%.\n"
                    f"   2.  **Past Sentiment ({past_sentiment_period_str})**: Provide breakdown: Positive: A%, Negative: B%, Neutral: C%.\n"
                    f"   3.  **Sentiment Trend Analysis**: Explicitly state the trend. For example: 'Positive sentiment changed by K percentage points from A% (in {past_sentiment_period_str}) to X% (in {analysis_period_str}). Negative sentiment changed by M percentage points from B% to Y%.' Calculate and state these changes clearly.\n"
                    f"   4.  **Key Themes/Hashtags**: Identify 1-2 specific, verifiable key themes or prominent hashtags (e.g., 'MissingCM' if data supports this for the current period) that are significantly driving the current sentiment or trends.\n"
                    f"   5.  **Reasons for Sentiment Shifts**: Explain the reasons for any observed sentiment shifts in simple terms, linking them directly to specific recent activities, controversies, community impacts, or competitor actions from the respective periods analyzed.",
        agent=sentiment_analyzer,
        context=[task1_research_activities_community, task2_monitor_landscape],
        expected_output=f"A highly detailed sentiment analysis report for {political_entity_name}, precisely following the structure for Section IV of the main political analysis prompt, including:\n"
                        f"- Current period ({analysis_period_str}) sentiment breakdown (Positive %, Negative %, Neutral %).\n"
                        f"- Past period ({past_sentiment_period_str}) sentiment breakdown (Positive %, Negative %, Neutral %).\n"
                        f"- Explicit trend analysis detailing percentage point changes between the two periods for each sentiment category.\n"
                        f"- Identification of 1-2 key themes or prominent, verifiable hashtags driving current sentiment.\n"
                        f"- Clear, evidence-based reasons for any sentiment shifts.\n"
                        f"- The output for this task MUST conclude with a list of any source URLs collected for sentiment context under 'Collected URLs for this task:'."
    )

    task4_compile_final_report = Task(
        description=f"**Primary Goal**: Compile all detailed findings from Task 1 (Activities/Controversies/Community Impact for {political_entity_name} during {analysis_period_str}), Task 2 (Competitor Landscape & Strategies during {analysis_period_str}), and Task 3 (Detailed Sentiment Analysis comparing {analysis_period_str} with {past_sentiment_period_str}) "
                    f"into a single, final, comprehensive report. The final report **MUST be written in simple, clear English** but must retain all the requested analytical depth and detail. "
                    f"Adhere strictly to the 7-section structure from the main political analysis prompt. The report title should be professional, like 'Comprehensive Political Analysis for {political_entity_name}', and specify the analysis period: {analysis_period_str}.\n"
                    f"   **Section VI - Strategic Recommendations**: Develop insightful, highly specific, and actionable recommendations for {political_entity_name}. These should directly address the findings from all preceding sections (controversies, sentiment issues like 'MissingCM' if identified, community concerns, competitor strategies). Aim for recommendations with depth and clear rationale similar to the 'Output A' example's 7-point plan.\n"
                    f"   **Section VII - Compiled Reference List**: Meticulously compile ALL unique URLs provided by Task 1, Task 2, and Task 3 into a single, clean, numbered list. **NO URLs should appear inline within Sections I-V of the main report body.**",
        agent=report_writer,
        context=[task1_research_activities_community, task2_monitor_landscape, task3_analyze_sentiment],
        expected_output=f"A final, comprehensive report on {political_entity_name} (analysis period: {analysis_period_str}) written in **simple, clear English**, "
                        f"precisely following the 7-section structure as per the main prompt (I. Activities/Schemes, II. Controversies/Impact, III. Community/Caste Impact, IV. Detailed Sentiment Analysis with percentages & trends, V. Broader Political Landscape with competitor strategies, VI. Detailed & Actionable Strategic Recommendations, VII. Compiled Reference List with all unique URLs). "
                        f"The recommendations in Section VI must be well-justified and strategically deep. Section VII must contain the consolidated URL list ONLY. "
                        f"The main title should be 'Comprehensive Political Analysis & Strategic Advisory: {political_entity_name} (Analysis Period: {start_date_str} to {current_date_str})'."
    )
    return [task1_research_activities_community, task2_monitor_landscape, task3_analyze_sentiment, task4_compile_final_report]

def execute_crew_analysis(political_entity_name, start_date_str_param, use_gpt_model=True, max_retries=3):
    """Executes the CrewAI analysis and returns the final report string."""
    llm_instance = create_llm_streamlit(use_gpt=use_gpt_model)
    if llm_instance is None:
        st.error("CRITICAL: Failed to initialize LLM. Cannot proceed with analysis.")
        return "LLM initialization failed. Cannot generate report."

    # search_tool_instance is global and should be initialized by initialize_tools_and_keys()
    if search_tool_instance is None or (isinstance(search_tool_instance, BaseTool) and "Dummy" in search_tool_instance.name) :
         st.warning("A functional web search tool (SerperDevTool) is not available or not configured. Search results will be limited or placeholders.")

    try:
        user_start_date_obj = datetime.strptime(start_date_str_param, "%Y-%m-%d").date()
    except ValueError:
        st.error("Invalid start date format received by the analysis function.")
        return "Invalid start date format. Cannot generate report."

    current_date_obj = date.today()
    current_date_str_param = current_date_obj.strftime("%Y-%m-%d")

    # Ensure start date is not after current date after parsing
    if user_start_date_obj > current_date_obj:
        st.error(f"Start date {start_date_str_param} cannot be after current date {current_date_str_param}.")
        return "Invalid date range. Cannot generate report."


    past_sentiment_start_obj = user_start_date_obj - timedelta(days=10)
    past_sentiment_end_obj = user_start_date_obj - timedelta(days=5)
    past_sentiment_start_str_param = past_sentiment_start_obj.strftime("%Y-%m-%d")
    past_sentiment_end_str_param = past_sentiment_end_obj.strftime("%Y-%m-%d")

    agents_list = create_political_agents_streamlit(political_entity_name, llm_instance)
    if not agents_list: # Should not happen if LLM is initialized
        st.error("Failed to create agents.")
        return "Agent creation failed."

    tasks_list = create_political_tasks_streamlit(
        political_entity_name,
        agents_list,
        start_date_str_param,
        current_date_str_param,
        past_sentiment_start_str_param,
        past_sentiment_end_str_param
    )

    crew_instance = Crew(
        agents=agents_list,
        tasks=tasks_list,
        process=Process.sequential,
        verbose=True # For detailed logs in the terminal/backend
    )

    final_report_string = "Political analysis failed after multiple retries or no output was generated." # Default error
    for attempt in range(max_retries):
        try:
            st.info(f"Running CrewAI analysis for '{political_entity_name}'... Attempt {attempt + 1}/{max_retries}")
            with st.spinner(f"Analyzing '{political_entity_name}' (Period: {start_date_str_param} to {current_date_str_param}). This may take a few minutes..."):
                result_object = crew_instance.kickoff()
            
            st.success(f"Analysis attempt {attempt + 1} completed!")

            if result_object:
                if hasattr(result_object, 'raw') and result_object.raw:
                    final_report_string = result_object.raw
                    break 
                elif hasattr(result_object, 'tasks_output') and result_object.tasks_output:
                    last_task_output = result_object.tasks_output[-1]
                    if hasattr(last_task_output, 'raw_output') and last_task_output.raw_output:
                        final_report_string = last_task_output.raw_output
                        break
                    elif hasattr(last_task_output, 'result') and last_task_output.result: # LangGraph/newer CrewAI
                        final_report_string = str(last_task_output.result)
                        break
                    elif hasattr(last_task_output, 'description') and isinstance(last_task_output.description, str): # Less ideal fallback
                        final_report_string = last_task_output.description
                        break
                # If it's already a string (older CrewAI versions or specific task setup)
                elif isinstance(result_object, str):
                    final_report_string = result_object
                    break
                else: # If it's some other object type, try to stringify
                    final_report_string = str(result_object)
                    st.warning("CrewAI returned an object, using its string representation. Output format might vary.")
                    break
            else:
                final_report_string = "No output was generated by the crew on this attempt."
                st.warning(final_report_string)

            if "failed" not in final_report_string.lower() and "no output" not in final_report_string.lower():
                 break # Successful report generation

        except Exception as e:
            st.error(f"Error during CrewAI analysis (Attempt {attempt + 1}): {str(e)}")
            # import traceback # For detailed debugging locally if needed
            # st.text_area("Full Error Traceback:", traceback.format_exc(), height=300)
            if attempt < max_retries - 1:
                st.warning(f"Retrying in 10 seconds... (Attempt {attempt + 2}/{max_retries})")
                time.sleep(10) # Longer sleep for retries in case of API rate limits
            else:
                st.error("Max retries reached. Unable to complete the political analysis.")
                final_report_string = "Political analysis failed after multiple retries due to errors."
    return final_report_string

# --- Main Streamlit App Logic ---
def run_main_app():
    st.title(" PoliSight Analyst Suite 🕵️‍♂️📊")
    st.markdown("Welcome! Get comprehensive political analysis on leaders, parties, or influencers.. Please configure below and click 'Analyze'.")

    # Initialize tools and keys - crucial to do this early in the app flow
    # This function now also handles initializing search_tool_instance
    if 'tools_initialized' not in st.session_state:
        if initialize_tools_and_keys():
            st.session_state.tools_initialized = True
        else:
            st.session_state.tools_initialized = False
            st.error("Critical error during tool and API key setup. Please check configuration and logs. App may not function correctly.")
            # return # Optionally stop app if critical setup fails

    with st.sidebar:
        st.header("⚙️ Analysis Configuration")
        
        political_entity_name_input = st.text_input(
            "Enter Political Entity Name:",
            placeholder="e.g., M.K. Stalin, DMK Party, etc."
        )

        # Default start date: 7 days ago
        default_start_date = date.today() - timedelta(days=7)
        start_date_input = st.date_input(
            "Select Analysis Start Date:",
            value=default_start_date,
            min_value=date(2000, 1, 1), # Sensible min date
            max_value=date.today(),     # Max date is today
            format="YYYY-MM-DD"
        )

        llm_option = st.radio(
            "Choose LLM Engine:",
            ("GPT-4o-mini (OpenAI - Cloud, Requires API Key)", "Ollama (Local - e.g., Llama3.1 - Advanced Setup)"),
            key="llm_choice_radio"
        )
        use_gpt_selection = True
        if "Ollama" in llm_option:
            use_gpt_selection = False
            st.info("Ensure your Ollama server is running locally and the specified model (e.g., 'llama3.1') is pulled.")
        
        if use_gpt_selection and not os.getenv("OPENAI_API_KEY"):
             st.warning("OpenAI API Key is not detected. GPT model will not work without it. Please set it in `.env` or Streamlit Secrets.")
        
        if not search_tool_instance or (isinstance(search_tool_instance, BaseTool) and "Dummy" in search_tool_instance.name):
            st.warning("Web search tool (Serper) is not configured or failed to initialize. Analysis will rely on LLM's internal knowledge, which may be outdated or limited.")


        submit_button = st.button("🚀 Analyze Political Entity", type="primary", use_container_width=True)

    if submit_button:
        if not political_entity_name_input:
            st.warning("Please enter a Political Entity Name to analyze.")
        elif not start_date_input:
            st.warning("Please select a valid Start Date for the analysis.")
        else:
            # Convert date object to string
            start_date_str_for_crew = start_date_input.strftime("%Y-%m-%d")

            # Preliminary checks before running the crew
            ready_to_run = True
            if use_gpt_selection and not os.getenv("OPENAI_API_KEY"):
                st.error("OpenAI API Key is missing. Cannot run analysis with GPT.")
                ready_to_run = False
            
            # Serper key check is implicitly handled by search_tool_instance status
            # but we can add an explicit warning here if it's missing and search is critical.
            if search_tool_instance is None or (isinstance(search_tool_instance, BaseTool) and "Dummy" in search_tool_instance.name):
                 st.error("Web Search tool (Serper) requires SERPER_API_KEY. Functionality will be severely limited without it. Check secrets or .env file.")
                 # You might choose to prevent running if search is absolutely essential
                 # ready_to_run = False


            if ready_to_run:
                st.info(f"Starting analysis for: **{political_entity_name_input}** from **{start_date_str_for_crew}** using **{'GPT (OpenAI)' if use_gpt_selection else 'Ollama'}**...")
                
                # Placeholder for the report while it's generating
                report_area = st.empty()
                report_area.markdown("### ⏳ Generating Report... Please wait.")

                final_report = execute_crew_analysis(
                    political_entity_name_input,
                    start_date_str_for_crew,
                    use_gpt_model=use_gpt_selection
                )
                
                report_area.empty() # Clear the "Generating..." message

                if final_report and isinstance(final_report, str) and "failed" not in final_report.lower() and "no output" not in final_report.lower():
                    st.subheader(f"📊 Final Analysis Report for {political_entity_name_input}")
                    st.markdown(final_report, unsafe_allow_html=True) # Use markdown for display

                    # Offer download
                    current_date_for_filename = date.today().strftime('%Y%m%d')
                    report_file_name = f"{political_entity_name_input.replace(' ', '_').lower()}_report_{start_date_str_for_crew}_to_{current_date_for_filename}.md"
                    st.download_button(
                        label="📥 Download Report as Markdown",
                        data=final_report,
                        file_name=report_file_name,
                        mime="text/markdown",
                    )
                elif final_report: # It's likely an error message string from execute_crew_analysis
                    st.error(f"Report Generation Failed: {final_report}")
                else:
                    st.error("Report generation failed or returned an empty result. Please check console logs for more details if running locally.")
    
    st.markdown("---")
    st.caption("PoliSight Analyst Suite - Powered by CrewAI & Streamlit")

# --- App Entry Point with Password Check ---
if "password_correct" not in st.session_state:
    st.session_state.password_correct = False

if st.session_state.password_correct:
    run_main_app()
else:
    if check_password(): # This will render password input and check it
        st.session_state.password_correct = True
        st.rerun() # Important to rerun to clear password input and load main app
    # else:
        # check_password() returned False, it handles showing errors or waiting for input.
        # st.stop() # Optionally stop if password input is empty, but text_input handles it.