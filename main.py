import os
os.environ["OPENAI_API_KEY"] = "NA"  # prevents CrewAI from trying to reach OpenAI in the background

import yfinance as yf
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool


# ---- Custom tool: fetches stock data using yfinance ----
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


# ---- LLM config: local Ollama model, no API key needed ----
llm = LLM(
    model="ollama/llama3.2",
    base_url="http://localhost:11434"
)

# ---- Agents ----
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

# ---- Tasks ----
research_task = Task(
    description="Fetch the latest stock data for ticker {ticker} using the Stock Price Tool. "
                "Report the key numbers clearly.",
    expected_output="A structured summary of the stock's latest price, 5-day high/low, volume, and sector.",
    agent=researcher,
)

analysis_task = Task(
    description="Using the research findings, write a short investment briefing (50-100 words) "
                "about {ticker}. Mention whether the stock shows short-term strength or weakness "
                "based on the 2-day range, and note any risks in reading too much into 2 days of data.",
    expected_output="A concise, well-written investment briefing in plain English.",
    agent=analyst,
    context=[research_task],
)

# ---- Crew ----
crew = Crew(
    agents=[researcher, analyst],
    tasks=[research_task, analysis_task],
    process=Process.sequential,
    verbose=True,
    memory=False,
)

if __name__ == "__main__":
    ticker = input("Enter a stock ticker (e.g. AAPL, TSLA, MSFT): ").strip().upper()
    result = crew.kickoff(inputs={"ticker": ticker})
    print("\n\n===== FINAL BRIEFING =====\n")
    print(result.raw)