import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
import yfinance as yf

st.set_page_config(page_title="AI Stock Briefing Crew", page_icon="📈", layout="centered")


# ---- Custom tool ----
@tool("Stock Price Tool")
def stock_price_tool(ticker: str) -> str:
    """Fetches recent stock price data and basic info for a given ticker symbol (e.g. AAPL, TSLA, MSFT)."""
    stock = yf.Ticker(ticker)
    hist = stock.history(period="5d")
    if hist.empty:
        return f"No data found for ticker {ticker}. Please check the symbol."
    latest = hist.iloc[-1]
    info = stock.info
    summary = (
        f"Ticker: {ticker}\n"
        f"Company: {info.get('longName', 'N/A')}\n"
        f"Latest Close Price: {latest['Close']:.2f}\n"
        f"5-Day High: {hist['High'].max():.2f}\n"
        f"5-Day Low: {hist['Low'].min():.2f}\n"
        f"Volume: {latest['Volume']}\n"
        f"Sector: {info.get('sector', 'N/A')}\n"
    )
    return summary


@st.cache_resource
def get_crew():
    llm = LLM(model="ollama/llama3.2", base_url="http://localhost:11434")

    researcher = Agent(
        role="Financial Researcher",
        goal="Gather accurate, up-to-date stock data for the given company ticker: {ticker}",
        backstory="You are a meticulous financial researcher who specializes in pulling "
                  "real-time stock data and summarizing key metrics clearly.",
        tools=[stock_price_tool],
        llm=llm,
        verbose=True,
    )

    analyst = Agent(
        role="Investment Analyst",
        goal="Analyze the stock data provided and produce a short, clear investment briefing",
        backstory="You are an experienced investment analyst who turns raw financial data "
                  "into concise, actionable insights for everyday investors.",
        llm=llm,
        verbose=True,
    )

    research_task = Task(
        description="Fetch the latest stock data for ticker {ticker} using the Stock Price Tool. "
                    "Report the key numbers clearly.",
        expected_output="A structured summary of the stock's latest price, 5-day high/low, volume, and sector.",
        agent=researcher,
    )

    analysis_task = Task(
        description="Using the research findings, write a short investment briefing (150-200 words) "
                    "about {ticker}. Mention whether the stock shows short-term strength or weakness "
                    "based on the 5-day range, and note any risks in reading too much into 5 days of data.",
        expected_output="A concise, well-written investment briefing in plain English.",
        agent=analyst,
        context=[research_task],
    )

    return Crew(
        agents=[researcher, analyst],
        tasks=[research_task, analysis_task],
        process=Process.sequential,
        verbose=True,
    )


# ---- UI ----
st.title("📈 AI Stock Briefing Crew")
st.caption("Multi-agent system: a Researcher agent fetches live stock data, "
           "then an Analyst agent writes an investment briefing — powered by local CrewAI + Ollama.")

ticker = st.text_input("Enter a stock ticker", placeholder="e.g. AAPL, TSLA, MSFT").strip().upper()
run_button = st.button("Run Agent Crew", type="primary")

if run_button:
    if not ticker:
        st.warning("Please enter a ticker symbol first.")
    else:
        with st.spinner(f"Agents are researching and analyzing {ticker}... this may take a minute or two on a local model."):
            crew = get_crew()
            result = crew.kickoff(inputs={"ticker": ticker})

        st.success("Done!")
        st.subheader(f"Investment Briefing: {ticker}")
        st.write(result.raw)

        with st.expander("See agent process details"):
            st.write(result)


# import streamlit as st
# from main import crew

# st.set_page_config(page_title="AI Research Crew")

# st.title("AI Research & Writing Crew")
# st.caption("Multi-agent system: A Research agent breaks down any topic into key points," 
# "A writer agent turns those points into a polished summary")

# topic = st.text_area("Enter any topic or question:", "The impact of AI on education", height=100)

# if st.button("Run Agents"):
#     if not topic.strip():
#         st.warning("Please enter a topic or question")
#     else:
#         with st.spinner("Agents are researching and writing... this may take a minute or two."):
#             result = crew.kickoff(inputs={"topic":topic.strip()})

#             st.sucess("done!")
#             st.subheader(f"Report:{topic.strip()}")
#             st.write(result.raw)

# st.markdown("---")
# st.caption("Built with CrewAI +local Ollama (llama3.2). No external API key is required.")