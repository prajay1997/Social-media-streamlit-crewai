# --- SQLite3 Hotfix for Streamlit Sharing ---
# This must be at the VERY TOP of the file.
try:
    import pysqlite3
    import sys
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass # Silently pass if not available

import streamlit as st
import os
import time
from datetime import datetime, timedelta, date
from crewai import Agent, Task, Crew, Process

# --- Tool Imports ---
# BaseTool is used for the DummySearchTool fallback.
try:
    from langchain_core.tools import BaseTool
except ImportError:
    st.error("CRITICAL ERROR: langchain_core.tools.BaseTool not found. Fallback dummy search tool will not function.")
    BaseTool = object 

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
st.set_page_config(page_title="PoliSight Analyst Suite", layout="wide", initial_sidebar_state="expanded")

# --- Global variable for initialized search tool ---
search_tool_instance = None
serper_key_loaded_successfully = False
openai_key_loaded_successfully = False

# --- Password Protection Function ---
def check_password():
    """Returns True if the password is correct, False otherwise."""
    # This function will primarily use st.secrets for the APP_PASSWORD
    # It includes a fallback to os.getenv for LOCAL_APP_PASSWORD for local testing,
    # but for Streamlit Cloud, st.secrets["APP_PASSWORD"] is the target.

    correct_password_value = None
    password_source_message = ""

    try:
        if hasattr(st, 'secrets') and "APP_PASSWORD" in st.secrets:
            correct_password_value = st.secrets.get("APP_PASSWORD")
            if correct_password_value and isinstance(correct_password_value, str) and correct_password_value.strip():
                password_source_message = "Using APP_PASSWORD from Streamlit Secrets."
            else:
                # Secret exists but is empty or invalid
                st.error("APP_PASSWORD found in Streamlit Secrets but is empty or invalid. Please check its value in your app settings.")
                correct_password_value = None # Treat as not found for security
        else:
            # APP_PASSWORD not in st.secrets, try local fallback if needed (though for deployment, this path means misconfiguration)
            password_source_message = "APP_PASSWORD not found in Streamlit Secrets. "
            # Fallback for local development if st.secrets is not configured or APP_PASSWORD is not set
            # For deployed app, this indicates an issue if reached.
            local_password_env = os.getenv("LOCAL_APP_PASSWORD")
            if local_password_env:
                correct_password_value = local_password_env
                password_source_message += "Using LOCAL_APP_PASSWORD environment variable (for local testing)."
            else:
                # No Streamlit secret, no local env var, use hardcoded default for local testing only
                correct_password_value = "testpassword123" # Default local fallback
                password_source_message += "Using default local fallback password 'testpassword123'. THIS IS FOR LOCAL TESTING ONLY."
                st.sidebar.caption("Hint (Local Only): Default password is 'testpassword123' or set LOCAL_APP_PASSWORD.")
                
    except Exception as e:
        st.sidebar.error(f"Error retrieving app password: {e}")
        st.sidebar.warning("Falling back to default local password due to error. THIS IS FOR LOCAL TESTING ONLY.")
        correct_password_value = "testpassword123" # Emergency fallback for local testing
        st.sidebar.caption("Hint (Local Only): Default password is 'testpassword123'")

    # Display password source info in sidebar for clarity during testing
    if 'password_source_displayed' not in st.session_state and password_source_message:
        st.sidebar.info(password_source_message)
        st.session_state.password_source_displayed = True


    if not correct_password_value: # If somehow still None or empty after checks
        st.error("CRITICAL: App password configuration is missing or invalid. Please set APP_PASSWORD in Streamlit Secrets.")
        return False

    # Password input field
    # Using a unique key for the password input to avoid conflicts
    password_input = st.text_input("Enter Password to Access:", type="password", key="app_access_password_input")

    if not password_input:
        st.info("Please enter the password to proceed.")
        return False # Don't proceed if no input

    if password_input == correct_password_value:
        return True
    elif password_input: # If a password was entered but it's incorrect
        st.error("Password incorrect. Please try again.")
        return False
    
    return False # Default case


def initialize_streamlit_keys_and_tools():
    """
    Loads API keys from Streamlit secrets, sets them as environment variables,
    and initializes the search tool.
    Updates global flags for key loading status.
    """
    global search_tool_instance, serper_key_loaded_successfully, openai_key_loaded_successfully

    serper_key_loaded_successfully = False
    openai_key_loaded_successfully = False

    st.sidebar.subheader("🔑 API Key Status")

    try:
        if hasattr(st, 'secrets') and "OPENAI_API_KEY" in st.secrets:
            openai_api_key_from_secrets = st.secrets["OPENAI_API_KEY"]
            if openai_api_key_from_secrets and isinstance(openai_api_key_from_secrets, str) and openai_api_key_from_secrets.strip():
                os.environ["OPENAI_API_KEY"] = openai_api_key_from_secrets
                openai_key_loaded_successfully = True
                st.sidebar.success("OpenAI API Key: Loaded successfully from Secrets.", icon="✅")
            else:
                st.sidebar.error("OpenAI API Key: Found in Secrets but is empty or invalid.", icon="❌")
        else:
            st.sidebar.error("OpenAI API Key: Not found in Streamlit Secrets.", icon="❌")
    except Exception as e:
        st.sidebar.error(f"OpenAI API Key: Error loading from Secrets - {e}", icon="❌")

    try:
        if hasattr(st, 'secrets') and "SERPER_API_KEY" in st.secrets:
            serper_api_key_from_secrets = st.secrets["SERPER_API_KEY"]
            if serper_api_key_from_secrets and isinstance(serper_api_key_from_secrets, str) and serper_api_key_from_secrets.strip():
                os.environ["SERPER_API_KEY"] = serper_api_key_from_secrets
                serper_key_loaded_successfully = True
                st.sidebar.success("Serper API Key: Loaded successfully from Secrets.", icon="✅")
            else:
                st.sidebar.error("Serper API Key: Found in Secrets but is empty or invalid.", icon="❌")
        else:
            st.sidebar.error("Serper API Key: Not found in Streamlit Secrets.", icon="❌")
    except Exception as e:
        st.sidebar.error(f"Serper API Key: Error loading from Secrets - {e}", icon="❌")

    st.sidebar.subheader("🛠️ Search Tool Status")
    if SerperDevTool and serper_key_loaded_successfully:
        try:
            search_tool_instance = SerperDevTool()
            st.sidebar.success("SerperDevTool: Initialized successfully.", icon="✅")
        except Exception as e:
            st.sidebar.error(f"SerperDevTool: Failed to initialize - {e}", icon="❌")
            search_tool_instance = None
    elif not SerperDevTool:
        st.sidebar.error("SerperDevTool: Library not imported. Search unavailable.", icon="❌")
        search_tool_instance = None
    else:
        st.sidebar.warning("SerperDevTool: Not initialized (SERPER_API_KEY missing or invalid).", icon="⚠️")
        search_tool_instance = None

    if search_tool_instance is None and BaseTool is not object :
        st.sidebar.warning("Using Dummy Search Tool as fallback.", icon="💡")
        class DummySearchTool(BaseTool):
            name: str = "Dummy Search Tool (Real Search Unavailable)"
            description: str = "A dummy search tool. Returns a placeholder message indicating search was not performed."
            def _run(self, search_query: str) -> str:
                return f"Search for '{search_query}' was NOT PERFORMED. The real web search tool (SerperDevTool) is unavailable or not configured correctly."
        try:
            search_tool_instance = DummySearchTool()
        except Exception as e_dummy:
            st.sidebar.error(f"DummySearchTool: Failed to create - {e_dummy}", icon="❌")
            search_tool_instance = None
    elif search_tool_instance is None and BaseTool is object:
        st.sidebar.error("CRITICAL: BaseTool not imported. Cannot create DummySearchTool.", icon="❌")

    return openai_key_loaded_successfully # Only return status of OpenAI key for now, search tool handled by global


def create_llm_streamlit(use_gpt=True):
    if use_gpt:
        if ChatOpenAI is None:
            st.error("Langchain OpenAI component not loaded. Cannot use GPT model.")
            return None
        if not openai_key_loaded_successfully:
            st.error("OpenAI API Key not loaded successfully. Cannot initialize GPT model.")
            return None
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.5, openai_api_key=os.getenv("OPENAI_API_KEY"))
    else:
        if Ollama is None:
            st.error("Ollama component not loaded. Cannot use Ollama model.")
            return None
        try:
            llm = Ollama(model="llama3.1")
            return llm
        except Exception as e:
            st.error(f"Failed to initialize Ollama: {e}. Ensure Ollama server is running and the model 'llama3.1' is pulled.")
            return None

def create_political_agents_streamlit(political_entity_name, llm):
    global search_tool_instance
    agent_tools_list = []
    if search_tool_instance:
        agent_tools_list.append(search_tool_instance)
    else:
        st.warning("No search tool (Serper or Dummy) is available for agents. Analysis will rely solely on LLM's internal knowledge.")

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

def execute_crew_analysis_streamlit(political_entity_name, start_date_str_param, use_gpt_model=True, max_retries=3):
    llm_instance = create_llm_streamlit(use_gpt=use_gpt_model)
    if llm_instance is None:
        st.error("CRITICAL: LLM initialization failed. Cannot proceed with analysis.")
        return "LLM initialization failed. Cannot generate report."

    if search_tool_instance is None or (isinstance(search_tool_instance, BaseTool) and "Dummy" in search_tool_instance.name):
        st.warning("Reminder: Web search tool is not fully functional. Results may be limited.", icon="⚠️")

    try:
        user_start_date_obj = datetime.strptime(start_date_str_param, "%Y-%m-%d").date()
    except ValueError:
        st.error("Invalid start date format. Cannot generate report.")
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

    agents_list = create_political_agents_streamlit(political_entity_name, llm_instance)
    tasks_list = create_political_tasks_streamlit(
        political_entity_name, agents_list, start_date_str_param,
        current_date_str_param, past_sentiment_start_str_param, past_sentiment_end_str_param
    )

    crew_instance = Crew(
        agents=agents_list, tasks=tasks_list, process=Process.sequential, verbose=True
    )

    final_report_string = "Analysis failed or no output was generated after retries."
    for attempt in range(max_retries):
        try:
            with st.status(f"Running CrewAI analysis for '{political_entity_name}'... Attempt {attempt + 1}/{max_retries}", expanded=True) as status_ui:
                st.write(f"Analyzing period: {start_date_str_param} to {current_date_str_param}. This may take several minutes...")
                result_object = crew_instance.kickoff()
                status_ui.update(label=f"Analysis attempt {attempt + 1} completed!", state="complete", expanded=False)
            
            if result_object:
                if hasattr(result_object, 'raw') and result_object.raw:
                    final_report_string = result_object.raw
                elif hasattr(result_object, 'tasks_output') and result_object.tasks_output:
                    last_task_output = result_object.tasks_output[-1]
                    if hasattr(last_task_output, 'raw_output') and last_task_output.raw_output:
                        final_report_string = last_task_output.raw_output
                    elif hasattr(last_task_output, 'result'):
                        final_report_string = str(last_task_output.result)
                    elif isinstance(last_task_output.description, str):
                         final_report_string = last_task_output.description
                elif isinstance(result_object, str):
                    final_report_string = result_object
                else:
                    final_report_string = str(result_object)
                    st.warning("CrewAI returned an unexpected object type. Displaying its string representation.")
                
                if "failed" not in final_report_string.lower() and "no output" not in final_report_string.lower():
                    st.success("Report generated successfully!")
                    return final_report_string
            else:
                final_report_string = f"No output was generated by the crew on attempt {attempt + 1}."
                st.warning(final_report_string)

        except Exception as e:
            st.error(f"Error during CrewAI analysis (Attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries - 1:
                st.warning(f"Retrying in 10 seconds... (Attempt {attempt + 2}/{max_retries})")
                time.sleep(10)
            else:
                st.error("Max retries reached. Unable to complete the political analysis.")
                final_report_string = "Political analysis failed after multiple retries due to errors."
        
        if "failed" not in final_report_string.lower() and "no output" not in final_report_string.lower():
            break 

    return final_report_string

# --- Main Streamlit App UI and Logic ---
def run_main_app_logic():
    """Contains the main UI and logic for the analysis suite, run after password check."""
    st.title(" PoliSight Analyst Suite 🕵️‍♂️📊")
    st.markdown("Welcome! Get detailed political analysis reports on political leaders, parties, or influencers. Please configure below and click 'Analyze'.")

    if 'app_initialized' not in st.session_state:
        initialize_streamlit_keys_and_tools()
        st.session_state.app_initialized = True

    with st.sidebar:
        # API and Tool status is already displayed by initialize_streamlit_keys_and_tools()
        st.header("⚙️ Analysis Configuration")
        political_entity_name_input = st.text_input(
            "Enter Political Entity Name:", placeholder="e.g., M.K. Stalin, DMK Party, etc."
        )
        default_start_date = date.today() - timedelta(days=7)
        start_date_input = st.date_input(
            "Select Analysis Start Date:", value=default_start_date,
            min_value=date(2000, 1, 1), max_value=date.today(), format="YYYY-MM-DD"
        )
        llm_option = st.radio(
            "Choose LLM Engine:",
            ("GPT-4o-mini (OpenAI - Cloud, Requires API Key)", "Ollama (Local - e.g., Llama3.1 - Advanced Setup)"),
            key="llm_choice_radio",
            help="Ensure API keys are correctly set in Streamlit Secrets for OpenAI. For Ollama, ensure your local server is running."
        )
        use_gpt_selection = "GPT-4o-mini" in llm_option

        # Display warnings based on key status directly under the LLM choice for better visibility
        if use_gpt_selection and not openai_key_loaded_successfully:
            st.error("OpenAI API Key is not loaded. GPT model will not work.", icon="❗")
        if not serper_key_loaded_successfully and (search_tool_instance is None or "Dummy" in getattr(search_tool_instance, 'name', '')):
             st.warning("Serper API Key not loaded or tool failed. Web search will be limited/unavailable.", icon="⚠️")

        submit_button = st.button("🚀 Analyze Political Entity", type="primary", use_container_width=True)

    if submit_button:
        if not political_entity_name_input.strip():
            st.warning("Please enter a Political Entity Name to analyze.")
        elif not start_date_input:
            st.warning("Please select a valid Start Date for the analysis.")
        else:
            start_date_str_for_crew = start_date_input.strftime("%Y-%m-%d")
            ready_to_run = True

            if use_gpt_selection and not openai_key_loaded_successfully:
                st.error("Cannot run analysis: OpenAI API Key is missing or invalid. Check Streamlit Secrets and sidebar status.")
                ready_to_run = False
            
            if ready_to_run:
                st.info(f"Starting analysis for: **{political_entity_name_input}** from **{start_date_str_for_crew}** using **{'GPT (OpenAI)' if use_gpt_selection else 'Ollama'}**...")
                final_report = execute_crew_analysis_streamlit(
                    political_entity_name_input, start_date_str_for_crew, use_gpt_model=use_gpt_selection
                )
                if final_report and "failed" not in final_report.lower() and "no output" not in final_report.lower():
                    st.subheader(f"📊 Final Analysis Report for {political_entity_name_input}")
                    st.markdown(final_report, unsafe_allow_html=True)
                    current_date_for_filename = date.today().strftime('%Y%m%d')
                    report_file_name = f"{political_entity_name_input.replace(' ', '_').lower()}_report_{start_date_str_for_crew}_to_{current_date_for_filename}.md"
                    st.download_button(
                        label="📥 Download Report as Markdown", data=final_report,
                        file_name=report_file_name, mime="text/markdown"
                    )
                elif final_report:
                    st.error(f"Report Generation Process Concluded With Issues: {final_report}")
                else:
                    st.error("Report generation failed or returned an unexpected empty result.")
    
    st.markdown("---")
    st.caption("PoliSight Analyst Suite - Powered by CrewAI & Streamlit")

# --- App Entry Point with Password Check ---
if __name__ == "__main__":
    # Initialize session state for password check
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    if st.session_state.password_correct:
        run_main_app_logic()
    else:
        # Display password input form
        # The check_password() function handles the input and error messages.
        if check_password():
            st.session_state.password_correct = True
            # If password is correct, rerun to clear password input and show the main app.
            # This also ensures initialize_streamlit_keys_and_tools runs in the context of the main app display.
            st.rerun()
        # If check_password() returns False, it means either no input yet or incorrect password.
        # The check_password() function itself shows the relevant st.info or st.error.
