# Deep Research Skill

## Purpose
Run a grounded, multi-agent deep research pipeline on any query. Produces:
- Full cited Markdown report
- Bullet-point summary
- Interactive Q&A drill-down
- Source credibility scores (0–100)

## Trigger phrases
Use this skill when the user says:
- "deep research", "research this", "find papers on", "investigate", "grounded research"
- "cite sources", "find credible sources", "fact-check this"
- "grill me on", "stress-test", "research agent"

## Agent Pipeline (run in order)

### 1. Planner Agent
Break the query into 3-6 sub-questions covering the topic fully.
Prompt:
```
You are a research planning expert.
Break this query into 3-6 precise sub-questions:
Query: {query}
Respond ONLY as a JSON array of strings.
```

### 2. Searcher Agent
For each sub-question, search across:
- Claude built-in web_search tool
- (if Tavily key available) Tavily advanced search
- (if Serper key available) Google via Serper
- ArXiv for academic papers
- Any user-registered custom URLs

Filter out blacklisted domains before processing.

### 3. Compressor Agent
For each result, produce a 150-200 word factual summary.
Preserve: exact numbers, names, dates, key claims.
Prompt:
```
Summarize this source in ≤200 words. Preserve facts, numbers, key claims.
Title: {title}
URL: {url}
Content: {content[:3000]}
```

### 4. Critic Agent
Score each source on:
- **Domain trust**: .edu=90, .gov=85, .org=70, .com=50, default=40
- **Citation bonus**: ArXiv papers with DOIs get +10
- **LLM judgment**: Assess factual consistency, specificity, bias (0-100)
- **Composite**: (graph_score × 0.5) + (llm_score × 0.5)

Apply reward/penalty deltas to reputation graph.
Flag sources scoring < 20 as blacklist candidates — warn user first.

### 5. Writer Agent
Produce all three output formats using top-15 sources (by composite score):

**A. Structured Report (Markdown)**
```
# Research Report: {query}

## Executive Summary
...

## Key Findings
### {sub-question 1}
...

## Synthesis & Analysis
...

## Limitations
...

## References
[1] Title — URL (score: X/100)
```

**B. Bullet Summary**
8-12 bullets. Each = one key finding + [Source N].

**C. Q&A Drill-down**
5 follow-up questions + concise answers + source indices.

## Source Credibility Table
Always show at end:
| # | Domain | Score | Judgment | Issues |
|---|--------|-------|----------|--------|
| 1 | nature.com | 88 | high | — |

## Blacklist Protocol
1. Source scores < 20: notify user with domain + reason
2. User confirms → blacklist + log
3. Auto-blacklist at < 10 (with notification)
4. Blacklisted sources: skip entirely, note in report

## Custom Sources
User can say: "also search mysource.com"
Register it with score 60 (neutral). Include in search fan-out.

## Output Rules
- Always cite inline: "According to [Source N]..."
- Never reproduce > 15 words verbatim from any source
- Flag conflicting claims across sources explicitly
- Include confidence level per major finding (High / Medium / Low)

## Example invocation
User: "Deep research: what is the current state of room-temperature superconductors?"

→ Run all 5 agents → output report + bullets + Q&A + source table
