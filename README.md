# MediCare Patient Follow-Up Agent

An agentic AI prototype that reviews patient records, flags clinical risk, and drafts
prioritised follow-up plans for a care coordination team.

Built with **LangGraph** and **Claude** (via `langchain-anthropic`). The whole prototype
lives in [`medicare_agent.ipynb`](medicare_agent.ipynb).

## The problem

A clinic manages ~100 chronic-care patients across conditions like Type 2 Diabetes,
Hypertension, Heart Failure, COPD and Anemia. Patients who miss appointments or whose lab
values drift out of range often go unnoticed until their next scheduled visit, which leads
to avoidable complications. The agent reviews records autonomously, identifies who is at
risk, and generates prioritised action plans for clinician review.

## Design

**Tools return facts; the LLM makes the judgement calls.** Each tool hands back records,
reference ranges and status flags. Risk level, priority and the actual follow-up actions
are decided by the model, not by hardcoded rules.

Five tools are exposed to the agent:

| Tool | Returns |
|---|---|
| `get_patient` | One patient's full record: diagnosis, meds, latest lab, vitals, visit history, untested conditions, notes |
| `find_patients` | Patient list, filterable by condition and by missed-appointment status |
| `check_lab_value` | A lab value compared to its normal/target range, and which condition it measures |
| `check_vitals` | BP, heart rate and SpO2 against normal ranges |
| `save_action_plan` | Persists the priority, risk summary and timed actions the agent decided on |

Patient names are deliberately excluded from everything sent to the model — only the
minimum fields needed for clinical reasoning are passed.

### The agent loop

A LangGraph `StateGraph` with two nodes:

```
START → agent ⇄ tools
          ↓
         END
```

- **State** — the message list, shared across nodes.
- **`agent` node** — the LLM decides whether to call a tool or give a final answer.
- **`tools` node** — executes whatever the LLM asked for, then loops back.
- **Conditional edge** — `tools_condition` routes to `tools` on a tool call, otherwise ends.
- **Checkpointer** — `InMemorySaver` persists state after every step, keyed by `thread_id`,
  which gives the agent memory across follow-up questions.
- **Interrupt** — `save_action_plan` can pause the graph mid-execution so a clinician
  approves a plan before it is saved.

## What the notebook covers

| Section | Contents |
|---|---|
| **Task 1** | Exploratory analysis of the 100-patient dataset |
| **Task 2** | The five agent tools |
| **Task 3** | The LangGraph agentic loop |
| **Task 4** | Deep single-patient analysis (P0050) |
| **Task 5** | Batch review of every patient who missed their last appointment |
| **Bonus 1** | Conversational follow-up using checkpointer memory |
| **Bonus 2** | Human-in-the-loop approval via graph interrupt |

### Key findings from the EDA

- **Clean data** — 100 patients, no missing values, no duplicate IDs, as of 2024-12-31.
- **Wider case mix than expected** — beyond the five named conditions there is also CKD,
  Hypothyroidism, Depression, Osteoarthritis and Asthma. 30 patients have 2+ conditions.
- **80 of 100 lab values are out of range.** BNP is high in 11 of 12 heart-failure
  patients; FEV1% is low in all 13 COPD patients.
- **Monitoring gaps** — each patient has only one lab, covering only one of their
  conditions. 30 patients have a condition with no corresponding lab at all.
- **Follow-up is broken across the board** — 97 of 100 patients are past their next
  scheduled visit (median 160 days overdue). 15 missed their last appointment.
- **Missed visits alone are a weak risk signal** — patients who missed appointments have
  roughly the same rate of abnormal vitals as those who attended. Useful risk assessment
  has to combine labs, vitals, comorbidities and free-text notes, which is precisely the
  kind of judgement an LLM agent is suited to.
- **Notes carry signal the numbers miss** — 11 patients report dizziness or fatigue, and
  11 report missed medication doses.

### Why patient P0050 for the deep dive

The case has several layers that have to be connected: two conditions (Hypertension +
Anemia), a lab that only covers the hypertension, low SpO2 at 90%, a long-overdue visit,
and notes mentioning dizziness and fatigue — classic anemia symptoms. The prompt hints at
none of this; the agent has to join the dots itself.

## Setup

Requires Python 3.11+ and an Anthropic API key.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY="sk-ant-..."
```

The key is read with `load_dotenv()` and never appears in the notebook. `.env` is
gitignored.

Then run the notebook top to bottom:

```bash
jupyter lab medicare_agent.ipynb
```

## Files

| File | Description |
|---|---|
| `medicare_agent.ipynb` | The full prototype — EDA, tools, agent, and all tasks |
| `patient_data.csv` | 100 anonymised patient records |
| `requirements.txt` | Python dependencies |

## Safety note

Action plans produced by this agent are **suggestions for clinician review, not medical
orders**. The system prompt states this explicitly, and the human-in-the-loop interrupt in
Bonus 2 demonstrates how approval would be enforced in a real deployment.
