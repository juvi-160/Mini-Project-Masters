import streamlit as st
from ddgs import DDGS
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

st.set_page_config(page_title="AI Research Crew", page_icon="🔎", layout="centered")


# ---- Custom tool: free web search using DuckDuckGo, works for ANY topic ----
@tool("Web Search Tool")
def web_search_tool(query: str) -> str:
    """Searches the web for a given query/topic and returns a summary of top results.
    Use this to research any topic - news, facts, concepts, current events, etc."""
    try:
        results = DDGS().text(query, max_results=5)
        if not results:
            return f"No results found for '{query}'."
        formatted = f"Search results for '{query}':\n\n"
        for i, r in enumerate(results, 1):
            formatted += f"{i}. {r.get('title', 'No title')}\n{r.get('body', 'No description')}\nSource: {r.get('href', '')}\n\n"
        return formatted
    except Exception as e:
        return f"Search failed: {str(e)}"


@st.cache_resource
def get_crew():
    llm = LLM(model="ollama/llama3.2", base_url="http://localhost:11434")

    researcher = Agent(
        role="Research Specialist",
        goal="Gather accurate, relevant information about the given topic: {topic}",
        backstory="You are a meticulous researcher who searches the web and pulls together "
                  "the most relevant facts and information on any topic, clearly and objectively.",
        tools=[web_search_tool],
        llm=llm,
        verbose=True,
    )

    writer = Agent(
        role="Content Writer",
        goal="Turn research findings into a clear, well-organized summary report",
        backstory="You are a skilled writer who transforms raw research into concise, "
                  "easy-to-read summaries for a general audience.",
        llm=llm,
        verbose=True,
    )

    research_task = Task(
        description="Search the web for information about the topic: {topic}. "
                    "Gather the most relevant and important facts, using the Web Search Tool.",
        expected_output="A structured collection of key facts and information about the topic.",
        agent=researcher,
    )

    writing_task = Task(
        description="Using the research findings, write a clear, well-organized summary report "
                    "(200-250 words) about {topic}. Structure it with a brief intro, key points, "
                    "and a short conclusion.",
        expected_output="A polished, easy-to-read summary report in plain English.",
        agent=writer,
        context=[research_task],
    )

    return Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        process=Process.sequential,
        verbose=True,
    )


# ---- UI ----
st.title("🔎 AI Research Crew")
st.caption("Multi-agent system: a Researcher agent searches the web on any topic, "
           "then a Writer agent turns it into a summary report — powered by local CrewAI + Ollama.")

topic = st.text_input("Enter any topic", placeholder="e.g. climate change, AI in healthcare, Mughal history").strip()
run_button = st.button("Run Agent Crew", type="primary")

if run_button:
    if not topic:
        st.warning("Please enter a topic first.")
    else:
        with st.spinner(f"Agents are researching and writing about '{topic}'... this may take a minute or two on a local model."):
            crew = get_crew()
            result = crew.kickoff(inputs={"topic": topic})

        st.success("Done!")
        st.subheader(f"Summary Report: {topic}")
        st.write(result.raw)

        with st.expander("See agent process details"):
            st.write(result)