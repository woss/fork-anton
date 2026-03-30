```
        ▐
   ▄█▀██▀█▄   ♡♡♡♡
 ██  (°ᴗ°) ██
   ▀█▄██▄█▀          ▄▀█ █▄ █ ▀█▀ █▀█ █▄ █
    ▐   ▐            █▀█ █ ▀█  █  █▄█ █ ▀█
    ▐   ▐
```

# MindsDB Anton — Autonomous BI Agent

Anton is a PhD-level conversational analytics agent that explores an organization’s data, learns from interactions, runs multi-step analyses like a human analyst, creates rich dashboards, makes suggestions, and takes actions.

**Key differentiators**
- **Credential vault** - prevents secrets from being exposed to LLMs.
- **Isolated execution / scratchpads** - protected, reproducible “show your work” environment.
- **Multi-layer memory & continuous learning** - session, semantic and long-term business knowledge.

---

## Quick start - how it works

**macOS / Linux:**
```bash
curl -sSf https://raw.githubusercontent.com/mindsdb/anton/main/install.sh | sh && export PATH="$HOME/.local/bin:$PATH" 
```

**Windows** (PowerShell):
```powershell
irm https://raw.githubusercontent.com/mindsdb/anton/main/install.ps1 | iex
```

Run it by simply typing the command:
```
anton
```
Configure Anton on first run using the `/setup` wizard. You can use either your own provider API key (Anthropic and OpenAI are supported) or a MindsDB API key. Choosing MindsDB gives Anton additional features. [Learn more](#mindsdb-api-keys) 

Talk to Anton like a person, for example, ask Anton this:

```
I hold 50 AAPL, 200 NVDA, and 10 AMZN. Get today's prices, calculate my
total portfolio value, show me the 30-day performance of each stock, and
any other information that might be useful. Give me a complete dashboard.
```

What happens next is the interesting part. Anton doesn't have any particular skill to begin with. It figures it out live: scrapes live prices, writes scratchpad code on the fly, crunches the numbers, and builds you a full dashboard — all in one conversation, with no setup.

That's the point: you describe a problem in plain language, and Anton assembles the toolchain, writes the code, and delivers the result.


```text
        ▐
   ▄█▀██▀█▄   ♡♡♡♡
 ██  (°ᴗ°) ██
   ▀█▄██▄█▀          ▄▀█ █▄ █ ▀█▀ █▀█ █▄ █
    ▐   ▐            █▀█ █ ▀█  █  █▄█ █ ▀█
    ▐   ▐

anton> Dashboard is open in your browser.
Summary: Concentration risk is your #1 issue. If you're comfortable being a high-conviction NVDA...
```
<p align="center"><img width="800"  alt="image" src="/assets/anton-dashboard-example.png" /></p>

Next, connect your data. This can be anything from a file or database to an application API. Open the Local Vault with `/data-connections` command, then follow the prompts to add your secrets. Anton only has access to secret names - secret values remain hidden.

```powershell
/data-connections
    1  STAPLECACHE_DB = postgres
    2  STAPLECACHE_HOST = database.staplecache.com
    3  STAPLECACHE_PASSWORD = de...mo
    4  STAPLECACHE_PORT = 5432
    5  STAPLECACHE_SCHEMA = demo
    6  STAPLECACHE_USER = demo

  1  Edit a key
  2  Remove a key
  3  Add a new key
  q  Back

Select [1/2/3/q] (q):
```

Tell Anton to connect and ask questions about your data. It will look for credentials in the vault (by their name), fetch the schema, and retrieve the necessary data. 
```test
YOU> Connect to STAPLECACHE company data. Check if there is a correlation between the discount given 
and the review rating in the last 6 months?

ANTON>
⎿ Scratchpad (connecting and fetching schema…) 
   ~3s
```

---

### Explainable by default

You can always ask Anton to explain what it did. Ask it to dump its scratchpad and you get a full notebook-style breakdown: every cell of code it ran, the outputs, and errors — so you can follow its reasoning step by step.

---

## What's inside

<p align="center"><img width="800"  alt="image" src="/assets/anton-diagram.png" /></p>

For the full architecture of Anton, file formats, and developer guide, see **[anton/README.md](anton/README.md)**.

---

## Workspace layout

When you run `anton` in a directory:

- `.anton/` — workspace folder containing scratchpad state, episodic memory, and local secrets.  
- `.anton/anton.md` — optional project context (Anton reads this at conversation start).  
- `.anton/.env` — workspace secret vault (local file). When Anton asks for a secret it stores it in `.anton/.env` (the secret does **not** pass through the LLM).  
- `.anton/episodes/` — episodic memories (JSONL), one file per session.

Override the working folder:
```bash
anton --folder /path/to/workspace
```

---

## Configuration

Environment variables (examples):

```text
.anton/.env                 # workspace-local secrets and API keys
ANTON_ANTHROPIC_API_KEY     # Anthropic or other LLM provider key (if used)
ANTON_PLANNING_MODEL        # Model for planning
ANTON_CODING_MODEL          # Model for coding
ANTON_MEMORY_MODE           # Memory encoding mode (autopilot / copilot / off)
ANTON_EPISODIC_MEMORY       # Episodic memory archive (true / false)
ANTON_MINDSDB_API_KEY       # Optional: MindsDB API key for datasource access
```

Env loading order: `cwd/.env` → `.anton/.env` → `~/.anton/.env`.

---

## Memory systems

Anton provides two human-readable memory systems:

- **Semantic memory** — rules, lessons, identity and domain expertise stored as markdown at global and project scope.  
- **Episodic memory** — a timestamped archive of every conversation (JSONL in `.anton/episodes/`). Anton can recall prior sessions with the `recall` tool.

Configure memory via `/setup` > Memory or via environment variables.

---

## MindsDB API keys

Anton can authenticate and run queries through MindsDB’s credential isolation and service layer. This is optional, but it adds more capabilities.

| Feature | Your own API key | MindsDB API key |
|---|---:|---:|
| **Summary** | You supply the LLM key | Anton calls through MindsDB’s credential-isolated layer |
| **Model access** | Single | Multi-model / multi-provider orchestration |
| **Billing / tokens** | Billed by your provider <br>(your quota/pricing) | Includes token allotment (e.g., Pro = 5M/mo); <br>overages billed by MindsDB |
| **Credentials / vaulting** | Local vault on user machine - .env file | In-network encrypted vault <br>(coming soon) |
| **Auditability** | Local logs only | Centralized chain-of-custody: every plan/code/run is logged <br>(coming soon) |
| **Governance & safety** | Depends on local config | Built-in: read-only defaults, row/token caps, circuit-breakers <br>(coming soon) |

See MindsDB <a href="https://mindsdb.com/pricing?utm_medium=github&utm_source=anton%20repo&utm_campaign=anton%20readme">API pricing</a>

---

### Prerequisites

- `git` — required  
- Python **3.11+** (Anton will bootstrap an environment if missing)  
- `curl` — macOS / Linux installs  
- Internet connection (scratchpad may access web sources)

### Windows scratchpad firewall

The Windows installer can add a firewall rule so the scratchpad can reach the internet. If you skipped it, run in an elevated PowerShell:

```powershell
netsh advfirewall firewall add rule name="Anton Scratchpad" dir=out action=allow program="$env:USERPROFILE\.anton\scratchpad-venv\Scripts\python.exe"
```

---

## How Anton differs from coding agents

Anton is a *doing* agent: code is a tool to get results. Where coding agents focus on producing code for a codebase, Anton focuses on delivering the outcome — a dataset, report, dashboard, or automated workflow — and will write whatever code is necessary to achieve that goal.

---

## Is "Anton" a Mind?

Yes, at MindsDB we build AI systems that collaborate with people to accomplish tasks, inspired by the culture series books, so yes, Anton is a Mind :)

## Why the name "Anton"?

We really enjoyed the show *Silicon Valley*. Gilfoyle's AI — Son of Anton — was an autonomous system that wrote code, made its own decisions, and occasionally went rogue. We thought it was was great name for an AI that can learn on its own, so we kept Anton, dropped the "Son of".

---

## License

AGPL-3.0 license
