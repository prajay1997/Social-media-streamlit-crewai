# --- SQLite3 Hotfix for Streamlit Sharing ---
# This must be at the VERY TOP of the file.
try:
    # Using print statements here for initial hotfix status, as st may not be fully ready.
    # These will go to the terminal/logs on Streamlit Cloud.
    # print("Attempting to apply pysqlite3 hotfix...")
    import pysqlite3
    import sys
    sys.modules["sqlite3"] = pysqlite3
    # print("pysqlite3 hotfix applied successfully.")
except ImportError:
    # print("pysqlite3 not found, hotfix not applied. Standard sqlite3 will be used.")
    pass # Silently pass if not available
except Exception as e:
    # print(f"Error applying pysqlite3 hotfix: {e}")
    pass

import streamlit as st
import os
import time
from datetime import datetime, timedelta, date # Added date for consistency
from crewai import Agent, Task, Crew, Process # Added Process
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama # For Ollama
try:
    from langchain_core.tools import BaseTool # For DummySearchTool
except ImportError:
    st.error("CRITICAL ERROR: langchain_core.tools.BaseTool not found. Fallback dummy search tool will not function.")
    BaseTool = object # To prevent NameErrors if import fails

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="PoliSight Analyst Suite", layout="wide", initial_sidebar_state="expanded")

# --- Global Variables for Key Status & Search Tool ---
# These will be updated by initialize_streamlit_keys_and_tools()
openai_api_key_loaded = False
serper_api_key_loaded = False
search_tool_instance = None

# --- Password Protection ---
def check_password():
    """Returns True if the password is correct, False otherwise.
    Primarily uses st.secrets["APP_PASSWORD"] for deployed apps.
    """
    correct_password_from_secrets = None
    try:
        if hasattr(st, 'secrets') and "APP_PASSWORD" in st.secrets:
            correct_password_from_secrets = st.secrets.get("APP_PASSWORD")
            if not (correct_password_from_secrets and isinstance(correct_password_from_secrets, str) and correct_password_from_secrets.strip()):
                st.error("APP_PASSWORD found in Streamlit Secrets but is empty or invalid. Please check its value in your app settings.")
                return False # Treat as incorrect if empty/invalid
        else:
            st.error("APP_PASSWORD not found in Streamlit Secrets. This app requires a password. Please contact the administrator.")
            return False
    except Exception as e:
        st.error(f"Could not retrieve app password from Streamlit Secrets. Error: {e}")
        return False

    if not correct_password_from_secrets: # Should be caught above, but as a safeguard
        st.error("App password configuration is critically missing.")
        return False

    password_input = st.text_input("Enter Password to Access:", type="password", key="app_password_input_main_v2")

    if not password_input:
        st.info("Please enter the password to proceed.")
        return False

    if password_input == correct_password_from_secrets:
        return True
    elif password_input: # If a password was entered but it's incorrect
        st.error("Password incorrect. Please try again.")
        return False
    return False


def initialize_streamlit_keys_and_tools():
    """
    Loads API keys from Streamlit secrets, sets them as environment variables,
    and initializes the search tool. Updates global flags.
    This function should be called once after password verification.
    """
    global openai_api_key_loaded, serper_api_key_loaded, search_tool_instance

    st.sidebar.subheader("🔑 API Key Status")

    # Load OpenAI API Key
    try:
        if hasattr(st, 'secrets') and "OPENAI_API_KEY" in st.secrets:
            key_val = st.secrets["OPENAI_API_KEY"]
            if key_val and isinstance(key_val, str) and key_val.strip():
                os.environ["OPENAI_API_KEY"] = key_val
                openai_api_key_loaded = True
                st.sidebar.success("OpenAI API Key: Loaded from Secrets.", icon="✅")
            else:
                st.sidebar.error("OpenAI API Key: Found in Secrets but is empty/invalid.", icon="❌")
                openai_api_key_loaded = False
        else:
            st.sidebar.error("OpenAI API Key: Not found in Streamlit Secrets.", icon="❌")
            openai_api_key_loaded = False
    except Exception as e:
        st.sidebar.error(f"OpenAI API Key: Error loading - {e}", icon="❌")
        openai_api_key_loaded = False

    # Load Serper API Key
    try:
        if hasattr(st, 'secrets') and "SERPER_API_KEY" in st.secrets:
            key_val = st.secrets["SERPER_API_KEY"]
            if key_val and isinstance(key_val, str) and key_val.strip():
                os.environ["SERPER_API_KEY"] = key_val
                serper_api_key_loaded = True
                st.sidebar.success("Serper API Key: Loaded from Secrets.", icon="✅")
            else:
                st.sidebar.error("Serper API Key: Found in Secrets but is empty/invalid.", icon="❌")
                serper_api_key_loaded = False
        else:
            st.sidebar.error("Serper API Key: Not found in Streamlit Secrets.", icon="❌")
            serper_api_key_loaded = False
    except Exception as e:
        st.sidebar.error(f"Serper API Key: Error loading - {e}", icon="❌")
        serper_api_key_loaded = False

    # Instantiate Search Tool
    st.sidebar.subheader("🛠️ Search Tool Status")
    if SerperDevTool and serper_api_key_loaded:
        try:
            search_tool_instance = SerperDevTool()
            st.sidebar.success("SerperDevTool: Initialized successfully.", icon="✅")
        except Exception as e:
            st.sidebar.error(f"SerperDevTool: Failed to initialize - {e}", icon="❌")
            search_tool_instance = None
    elif not SerperDevTool:
        st.sidebar.error("SerperDevTool: Library not imported. Search unavailable.", icon="❌")
        search_tool_instance = None
    else: # SerperDevTool imported but key not loaded
        st.sidebar.warning("SerperDevTool: Not initialized (SERPER_API_KEY issue).", icon="⚠️")
        search_tool_instance = None

    # Fallback to Dummy Search Tool
    if search_tool_instance is None:
        if BaseTool is not object:
            st.sidebar.warning("Using Dummy Search Tool as fallback.", icon="💡")
            class DummySearchTool(BaseTool):
                name: str = "Dummy Search Tool (Real Search Unavailable)"
                description: str = "A dummy search tool. Returns a placeholder message."
                def _run(self, search_query: str) -> str:
                    return f"Search for '{search_query}' was NOT PERFORMED. Real web search tool is unavailable."
            try:
                search_tool_instance = DummySearchTool()
            except Exception as e_dummy:
                st.sidebar.error(f"DummySearchTool: Failed to create - {e_dummy}", icon="❌")
        else:
            st.sidebar.error("CRITICAL: BaseTool not imported. Cannot create DummySearchTool.", icon="❌")


# --- LLM Creation Function ---
def create_llm_streamlit(use_gpt=True):
    if use_gpt:
        if not openai_api_key_loaded:
            st.error("OpenAI API Key not loaded. Cannot use GPT model.")
            return None
        if ChatOpenAI is None:
            st.error("ChatOpenAI library not available. Cannot use GPT model.")
            return None
        try:
            return ChatOpenAI(model="gpt-4o-mini", temperature=0.5) # API key is set in environ
        except Exception as e:
            st.error(f"Failed to initialize OpenAI GPT: {e}")
            return None
    else: # Ollama
        if Ollama is None:
            st.error("Ollama library not available. Cannot use Ollama model.")
            return None
        st.info("Attempting to connect to Ollama with model 'llama3.1'...")
        try:
            llm_ollama = Ollama(model="llama3.1")
            # Simple test to confirm connection if possible, though this might be slow
            # try:
            #     llm_ollama.invoke("test")
            #     st.success("Successfully connected to Ollama with model 'llama3.1'.")
            # except Exception as ollama_test_e:
            #     st.warning(f"Ollama initialized, but test invocation failed: {ollama_test_e}. Ensure server is responsive.")
            return llm_ollama
        except Exception as e:
            st.error(f"Failed to initialize or connect to Ollama: {e}\n"
                       "Please ensure Ollama server is running and 'llama3.1' model is pulled.")
            return None

# --- Agent Creation Function (Adapted from your reference) ---
def create_political_agents_streamlit(political_entity_name, llm):
    global search_tool_instance
    agent_tools_list = []
    if search_tool_instance:
        agent_tools_list.append(search_tool_instance)
    else:
        st.warning("No search tool available for agents. Analysis will rely on LLM's internal knowledge.", icon="⚠️")

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

# --- Task Creation Function (Adapted from your reference) ---
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
        f"include terms like 'news after:{start_date_str} before:{current_date_str}' or similar specific date restrictions in your search query string."
    )
    date_filter_instruction_past_sentiment = (
        f"When using {search_tool_name_for_prompt} for past sentiment context, you MUST craft your queries to find information ONLY from the period {past_sentiment_period_str}. For example, "
        f"include terms like 'news after:{past_sentiment_start_str} before:{past_sentiment_end_str}' in your search query string."
    )
    url_collection_instruction = (
        "You MUST collect specific website URLs for all key facts, activities, and claims. "
        "At the end of your response for THIS TASK, provide a consolidated list of all unique URLs you collected under a clear heading like 'Collected URLs for this task:'."
    )
    task1_research_activities_community = Task(
        description=f"**Primary Goal for {political_entity_name}**: Conduct an in-depth investigation for the period {analysis_period_str}. {url_collection_instruction}\n"
                      f"    **Search Instructions**: {date_filter_instruction_main_period}\n"
                      f"    **Specific Areas to Cover**:\n"
                      f"    1.  **Section I - Recent Political Activities & Scheme Launches**: Detail the nature of each activity (rallies, policy announcements etc.), key messages conveyed, and any new schemes (name, purpose, beneficiaries, potential impact). Be specific and provide details.\n"
                      f"    2.  **Section II - Controversies Involving {political_entity_name}**: Describe each significant controversy, identify key individuals/aspects, analyze media coverage intensity and general tone. Most importantly, provide a detailed analysis of the direct and indirect impact of these controversies on {political_entity_name}'s image and public perception.\n"
                      f"    3.  **Section III - Community and Caste-Specific Impact Analysis**: Provide a substantive analysis (not just a brief mention) if and how specific communities or caste groups are reportedly being affected (positively/negatively) by {political_entity_name}'s recent activities. Also, research if opposition party campaigns are reportedly benefiting or targeting these specific groups. If specific data or strong anecdotal evidence is found, highlight it. If little information is found, state that clearly.",
        agent=activity_researcher,
        expected_output=f"A detailed, multi-part report covering Sections I, II, and III of the main political analysis prompt for {political_entity_name} specifically for the period {analysis_period_str}. "
                        f"The content must be factual, deeply analytical (especially for controversy impact and community effects), and written clearly. "
                        f"The output for this task MUST conclude with a list of all source URLs under the heading 'Collected URLs for this task:'."
    )
    task2_monitor_landscape = Task(
        description=f"**Primary Goal**: Monitor and analyze the broader political landscape in the state of {political_entity_name} for the period {analysis_period_str}. {url_collection_instruction}\n"
                      f"    **Search Instructions**: {date_filter_instruction_main_period}\n"
                      f"    **Specific Areas to Cover (Section V of final report)**:\n"
                      f"    1. Report on significant activities of **ALL major political parties** and their key leaders in the state.\n"
                      f"    2. For EACH major party, provide a detailed analysis of their: \n"
                      f"       a. Recent public engagements, announcements, or policy stances.\n"
                      f"       b. **Underlying STRATEGIES and specific CAMPAIGNS** they are visibly adopting/running aimed at i) the betterment of the state (e.g., development initiatives, governance reforms) AND ii) increasing their vote share or public support (e.g., outreach programs, narrative building, target voter segments, key issues highlighted).\n"
                      f"       c. Key messages being pushed to the public.\n"
                      f"       d. Any notable successes or failures in their recent strategic efforts.",
        agent=landscape_monitor,
        expected_output=f"A comprehensive and strategic analysis of the activities and positioning of ALL major political players in the state for {analysis_period_str}, aligning with Section V of the main political analysis prompt. "
                        f"The analysis for each party should go beyond listing events and delve into their strategies and campaign effectiveness. "
                        f"The output for this task MUST conclude with a list of all source URLs under the heading 'Collected URLs for this task:'."
    )
    task3_analyze_sentiment = Task(
        description=f"**Primary Goal**: Conduct a DETAILED public sentiment analysis for {political_entity_name}.\n"
                      f"    **Current Sentiment Period**: Analyze sentiment for **{analysis_period_str}**. {url_collection_instruction.replace('key facts, activities, and claims', 'articles or data points specifically used for sentiment context during this period')}\n"
                      f"    **Past Sentiment Period for Comparison**: Analyze sentiment for **{past_sentiment_period_str}** (5-10 days before {start_date_str}). {date_filter_instruction_past_sentiment} {url_collection_instruction.replace('key facts, activities, and claims', 'articles or data points specifically used for sentiment context during this past period')}\n"
                      f"    **Required Output Details (Section IV of final report)**:\n"
                      f"    1.  **Current Sentiment ({analysis_period_str})**: Provide breakdown: Positive: X%, Negative: Y%, Neutral: Z%.\n"
                      f"    2.  **Past Sentiment ({past_sentiment_period_str})**: Provide breakdown: Positive: A%, Negative: B%, Neutral: C%.\n"
                      f"    3.  **Sentiment Trend Analysis**: Explicitly state the trend. For example: 'Positive sentiment changed by K percentage points from A% (in {past_sentiment_period_str}) to X% (in {analysis_period_str}). Negative sentiment changed by M percentage points from B% to Y%.' Calculate and state these changes clearly.\n"
                      f"    4.  **Key Themes/Hashtags**: Identify 1-2 specific, verifiable key themes or prominent hashtags (e.g., 'MissingCM' if data supports this for the current period) that are significantly driving the current sentiment or trends.\n"
                      f"    5.  **Reasons for Sentiment Shifts**: Explain the reasons for any observed sentiment shifts in simple terms, linking them directly to specific recent activities, controversies, community impacts, or competitor actions from the respective periods analyzed.",
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
        description=f"**Primary Goal**: Compile all detailed findings from Task 1, Task 2, and Task 3 "
                      f"into a single, final, comprehensive report. The final report **MUST be written in simple, clear English** but must retain all the requested analytical depth and detail. "
                      f"Adhere strictly to the 7-section structure from the main political analysis prompt. The report title should be professional, like 'Comprehensive Political Analysis for {political_entity_name}', and specify the analysis period: {analysis_period_str}.\n"
                      f"    **Section VI - Strategic Recommendations**: Develop insightful, highly specific, and actionable recommendations for {political_entity_name}. These should directly address the findings from all preceding sections. Aim for recommendations with depth and clear rationale.\n"
                      f"    **Section VII - Compiled Reference List**: Meticulously compile ALL unique URLs provided by Task 1, Task 2, and Task 3 into a single, clean, numbered list. **NO URLs should appear inline within Sections I-V of the main report body.**",
        agent=report_writer,
        context=[task1_research_activities_community, task2_monitor_landscape, task3_analyze_sentiment],
        expected_output=f"A final, comprehensive report on {political_entity_name} (analysis period: {analysis_period_str}) written in **simple, clear English**, "
                        f"precisely following the 7-section structure as per the main prompt (I. Activities/Schemes, II. Controversies/Impact, III. Community/Caste Impact, IV. Detailed Sentiment Analysis with percentages & trends, V. Broader Political Landscape with competitor strategies, VI. Detailed & Actionable Strategic Recommendations, VII. Compiled Reference List with all unique URLs). "
                        f"The recommendations in Section VI must be well-justified and strategically deep. Section VII must contain the consolidated URL list ONLY. "
                        f"The main title should be 'Comprehensive Political Analysis & Strategic Advisory: {political_entity_name} (Analysis Period: {start_date_str} to {current_date_str})'."
    )
    return [task1_research_activities_community, task2_monitor_landscape, task3_analyze_sentiment, task4_compile_final_report]

# --- Main Crew Execution Function ---
def execute_crew_analysis_streamlit(leader_name, start_date_str_param, use_gpt_model=True, max_retries=3):
    llm = create_llm_streamlit(use_gpt=use_gpt_model)
    if not llm:
        # Error already shown by create_llm_streamlit
        return "LLM initialization failed. Cannot generate report."

    # The global search_tool_instance is used. Its status is shown in the sidebar.
    # A warning is also shown in create_political_agents_streamlit if it's None.

    try:
        user_start_date_obj = datetime.strptime(start_date_str_param, "%Y-%m-%d").date()
    except ValueError:
        st.error("Invalid start date format for analysis.")
        return "Invalid start date format. Cannot generate report."

    current_date_obj = date.today()
    current_date_str_param = current_date_obj.strftime("%Y-%m-%d")

    if user_start_date_obj > current_date_obj:
        st.error(f"Start date {start_date_str_param} cannot be after current date {current_date_str_param}.")
        return "Invalid date range: Start date is in the future."

    past_sentiment_start_obj = user_start_date_obj - timedelta(days=10)
    past_sentiment_end_obj = user_start_date_obj - timedelta(days=5)
    past_sentiment_start_str_param = past_sentiment_start_obj.strftime("%Y-%m-%d")
    past_sentiment_end_str_param = past_sentiment_end_obj.strftime("%Y-%m-%d")
    
    agents = create_political_agents_streamlit(leader_name, llm)
    if not agents: # Should not happen if LLM is fine, but good check
        st.error("Failed to create agents.")
        return "Agent creation failed."

    tasks = create_political_tasks_streamlit(
        leader_name, agents, start_date_str_param, current_date_str_param,
        past_sentiment_start_str_param, past_sentiment_end_str_param
    )
    if not tasks: # Should not happen, but good check
        st.error("Failed to create tasks.")
        return "Task creation failed."

    crew = Crew(agents=agents, tasks=tasks, process=Process.sequential, verbose=1) # verbose=1 for some logs

    final_report_output = f"Analysis for {leader_name} failed after {max_retries} retries."
    for attempt in range(max_retries):
        try:
            with st.status(f"Running CrewAI analysis for '{leader_name}'... Attempt {attempt + 1}/{max_retries}", expanded=True) as status_ui:
                st.write(f"Analyzing period: {start_date_str_param} to {current_date_str_param}. This may take several minutes...")
                # CrewAI verbose output goes to console/logs, not easily into st.status
                result = crew.kickoff()
                status_ui.update(label=f"Analysis attempt {attempt + 1} completed!", state="complete", expanded=False)

            if result:
                # Try to extract the raw output, common for the last task in CrewAI
                if hasattr(result, 'raw') and result.raw:
                    final_report_output = result.raw
                elif hasattr(result, 'tasks_output') and result.tasks_output: # Newer CrewAI/LangGraph
                    last_task_output = result.tasks_output[-1]
                    if hasattr(last_task_output, 'raw_output') and last_task_output.raw_output:
                         final_report_output = last_task_output.raw_output
                    elif hasattr(last_task_output, 'result'):
                        final_report_output = str(last_task_output.result)
                    else: # Fallback if specific attributes aren't there
                        final_report_output = str(last_task_output) # Or str(result)
                elif isinstance(result, str): # If kickoff directly returns a string
                    final_report_output = result
                else: # Fallback for unexpected result types
                    final_report_output = str(result)
                    st.warning("CrewAI returned an unexpected result format. Displaying as string.")

                if "failed" not in final_report_output.lower() and "no output" not in final_report_output.lower():
                    st.success("Report generated successfully!")
                    return final_report_output # Successful exit from loop and function
            else:
                final_report_output = f"No output from CrewAI on attempt {attempt + 1}."
                st.warning(final_report_output)

        except Exception as e:
            st.error(f"CrewAI analysis attempt {attempt + 1} failed: {str(e)}")
            # import traceback # For server-side debugging
            # print(traceback.format_exc())
            if attempt < max_retries - 1:
                st.warning(f"Retrying in {5 * (attempt + 1)} seconds...")
                time.sleep(5 * (attempt + 1)) # Exponential backoff might be too much, simple wait
            else:
                st.error("Max retries reached. Unable to complete the analysis.")
                final_report_output = "Analysis failed after multiple retries due to errors."
        
        # If a successful report was generated, we would have returned from the function already.
        # If we are here, it means the attempt failed or produced no good output.
        
    return final_report_output # Return the last status (likely an error or "no output" message)

# --- Main Streamlit App UI and Logic ---
def run_main_app_logic():
    """Main function to run the Streamlit application UI and logic, called after password success."""
    st.title("️PoliSight Analyst Suite 🕵️‍♂️📊") # Corrected title
    st.markdown("Welcome! Get detailed political analysis reports on political leaders, parties, or influencers. Please configure below and click 'Analyze'.")

    # Initialize keys and tools if not already done in this session
    # This ensures sidebar messages about key status are shown.
    if 'keys_initialized' not in st.session_state:
        initialize_streamlit_keys_and_tools()
        st.session_state.keys_initialized = True
        # st.rerun() # Consider if needed to refresh sidebar immediately, can cause issues.

    with st.sidebar:
        # Key status is already shown by initialize_streamlit_keys_and_tools()
        st.header("⚙️ Analysis Configuration")
        leader_name_input = st.text_input("Enter Political Entity Name:", placeholder="e.g., M.K. Stalin, DMK Party")
        
        default_start_date = date.today() - timedelta(days=7)
        start_date_input = st.date_input(
            "Select Analysis Start Date:", value=default_start_date,
            min_value=date(2000, 1, 1), max_value=date.today(), format="YYYY-MM-DD"
        )
        
        llm_option = st.radio(
            "Choose LLM Engine:",
            ("GPT-4o-mini (OpenAI - Cloud, Requires API Key)", "Ollama (Local - e.g., Llama3.1 - Advanced Setup)"),
            key="llm_choice_main_app", # Unique key
            help="Ensure API keys are correctly set in Streamlit Secrets for OpenAI. For Ollama, ensure your local server is running."
        )
        use_gpt_selection = "GPT-4o-mini" in llm_option

        # Display warnings based on key status directly under the LLM choice
        if use_gpt_selection and not openai_api_key_loaded:
            st.error("OpenAI API Key not loaded. GPT model will not work.", icon="❗")
        
        # This warning is about the search tool's overall status
        if not serper_api_key_loaded and (search_tool_instance is None or "Dummy" in getattr(search_tool_instance, 'name', '')):
             st.warning("Serper API Key not loaded or tool failed. Web search will be limited/unavailable.", icon="⚠️")

        submit_button = st.button("🚀 Analyze Political Entity", type="primary", use_container_width=True)

    if submit_button:
        if not leader_name_input.strip():
            st.warning("Please enter a Political Entity Name to analyze.")
        elif not start_date_input: # Should not happen with date_input default
            st.warning("Please select a valid Start Date.")
        else:
            start_date_str_for_crew = start_date_input.strftime("%Y-%m-%d")
            
            # Pre-flight checks before running analysis
            can_run_analysis = True
            if use_gpt_selection and not openai_api_key_loaded:
                st.error("Cannot start analysis: OpenAI API Key is not loaded. Check Streamlit Secrets and sidebar status.")
                can_run_analysis = False
            
            # You might add a stricter check for Serper if it's absolutely critical
            # if not serper_api_key_loaded:
            #     st.error("Cannot start analysis: Serper API Key for web search is not loaded.")
            #     can_run_analysis = False

            if can_run_analysis:
                st.info(f"Starting analysis for: **{leader_name_input}** from **{start_date_str_for_crew}** using **{'GPT (OpenAI)' if use_gpt_selection else 'Ollama'}**...")
                
                final_report = execute_crew_analysis_streamlit(
                    leader_name_input, start_date_str_for_crew, use_gpt_model=use_gpt_selection
                )
                
                if final_report and "failed" not in final_report.lower() and "no output" not in final_report.lower():
                    st.subheader(f"📊 Final Analysis Report for {leader_name_input}")
                    st.markdown(final_report, unsafe_allow_html=True) # Allow HTML if your report might contain it
                    
                    current_date_for_filename = date.today().strftime('%Y%m%d')
                    report_file_name = f"{leader_name_input.replace(' ', '_').lower()}_report_{start_date_str_for_crew}_to_{current_date_for_filename}.md"
                    st.download_button(
                        label="📥 Download Report as Markdown",
                        data=final_report,
                        file_name=report_file_name,
                        mime="text/markdown"
                    )
                elif final_report: # Contains an error message from execute_crew_analysis
                    st.error(f"Report Generation Process Concluded With Issues: {final_report}")
                else: # Should ideally not happen
                    st.error("Report generation failed or returned an unexpected empty result. Check logs if running locally.")
    
    st.markdown("---")
    st.caption("PoliSight Analyst Suite - Powered by CrewAI & Streamlit")

# --- App Entry Point with Password Check ---
if __name__ == "__main__":
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    if st.session_state.password_correct:
        run_main_app_logic()
    else:
        if check_password():
            st.session_state.password_correct = True
            st.rerun()
        # If check_password returns False, it handles displaying the password input or error messages.
        # No st.stop() needed here as the UI flow is managed by the password check.
