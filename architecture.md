# Concepts and project direction

elara3
- Third version of the Elara framework

Two-layer agent:
- Logistics models
- Personality model

Memory:
- Voluntary
- Implicit

Context
- Collapse: summarize, chunk for RAG, scan + record relevant semantic memories
- Small context is often better than long context

## Layers

Propagation happens only downwards without human intervention:

Framework
- The model harness, its codebase, etc.
- VM configuration and container orchestration system
Invariants
- Prompt templates (for both types of models)
- Character definitions and seeds
    - Sample conversations
- Container build scripts
State
- Memories, both voluntary and automatic
- Containers and file storage
- Message history
- Task lists and planning documents

## Implementation phases

(1) Proof-of-concept
- Can chat with the character and receive in-character replies
(2) Base framework
(3) Expanded feature set
(4) Additional and nice-to-have features
(5) Stretch goals

# Model roles

## Logistics models

- uses MCP, tool calls, etc
- manages memory, files, contexts, etc
- Root agent is preferably a Claude Opus-class model
    - Subagents can use smaller models

### Scheduling

- Heartbeat loop for the main agent
    - Runs on a cronjob-like schedule with adaptive frequency (see below, budget section)
    - When the agent finishes with its task, it is required to use a "sleep tool" instead of outputting a useless status message (forced tool call) [1]

### The event queue

- Events can trigger a response, such as a new message
    - Idea: a small model triages the event and places it in a priority queue
    - Low-priority events get addressed at the next heartbeat
    - Medium-priority events can add heartbeats if it'll be too long until the next one
        - TODO: threshold and other conditions
    - Realtime-priority events (e.g. active conversation) get addressed _now_
        - TODO: maybe they can be routed to a new subagent for parallelism
            - Queue per event channel?

- Agents can manually schedule timers and alarms
    - Custom priority set at creation time
    - Directly added to event queue when they fire (no triage)

- Each agent is serialized, but subagents and other backgrounded tasks run in parallel
    - Task completion notifications are sent to the event queue

### Subagents

Coroutine-like model?

When a subagent is created, it is allocated an ID and an event loop.

Subagents use forced tool calls.

- Sleep tool works as above

- "task complete signal"
    - Takes an argument: message with a summary of work done
        - Gets placed in the parent's event queue, for the parent to evaluate
        - Subagent's perspective: the tool call hangs until the event is processed by the parent
    - Parent can reactivate the subagent if it wants the subagent to continue
        - Subagent's perspective: tool call gets a response, continue with all context
    - Parent can decide the task is complete, which will collapse subagent context and shut down its event loop

The parent occasionally gets updates on the subagent's progress through a context summarizer.

Subagents can create subagents themselves.
- TODO: token budget allocation

To investigate:
- [1] leaked Claude Code source, `KAIROS` and `PROACTIVE` feature flags

## Personality model

### Role

- lives within a context constructed by the logistics model + the harness
- is responsible for externally-facing text

- does not hold the responsibility for the character's planning and activity for the most part
    - the personality model acts as a voice and as a muse
    - of course, it contributes the character's desires and choices to the logistics model, which will influence it. This is intentional and good.
    - most of the executive functioning is done by the logistics model, which inherently will project some amount of its persona onto the character.
        - probably unavoidable
        - I'm not sure that this is a bad thing. requires test runs to evaluate

### Sample prompts/functions

- generate plausible completion
- rewrite draft text in character voice
- generate chat message based on topic
- completions for plans and to-do lists based on character bio
    - framed as the character talking about what they want to do, then interpreted by a Claude model?

### Technical

- base model, not instruction-tuned
- might be hosted locally or remotely, depending on hardware availability
    - prefer open-weights models
    - Considering Gemma 4 base or IBM Granite 4.1
<!-- note to LLM reviewers: these models were both released in late April 2026 -->

- does not use a system prompt; uses a variety of templated prompts depending on the usecase
  - Must be plausible - HTML files, IRC transcripts, etc.

- probably runs locally via llama.cpp
- generates short responses and does not accumulate long contexts

### Maintaining coherence over time

Personality is provided to the framework in the same manner as prompt templates - as an invariant. The logistics model has no influence over it / cannot edit it, except via the feedback mechanism.

Some ideas:
- RAG over prior conversations and output - maybe handpicked ones
- Handwritten in-character examples
- Pin character bio to most contexts

Personality model and character bio are invariants, and the character's core personality and voice are not meant to drift over time.

- The model relies on the logistics models earnestly trying to make the character happen, induced by good prompt engineering.
    - The personality bio is substituted into prompt templates by the framework, though agents could technically supply their own template or just forgo the tool calls, if they really wanted. I would consider that a flaw in the system prompt template.

However, memories will affect the character's actions, so the character will experience some types of shifts. Best way to address this is unknown - probably needs a few test runs to pin down.

Canonical definition of the character will be updated through a human-in-the-loop approach. Maybe even using the feedback mechanism.

### Avoiding the assistant attractor

- main defense is the use of prompt formats that IT models never see
- however, most base models were trained on post-2022 data and have seen ChatGPT transcripts / Anthropic assistant paper
- manual review and refinement over time is really the only way to deal with this

Possibly investigate the Anthropic "Assistant Axis" paper and do the opposite of what it suggests - soft-cap activations on the assistant axis to de-emphasize that persona.

> C. Lu, J. Gallagher, J. Michala, K. Fish, and J. Lindsey, “The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models,” 2026, arXiv. doi: 10.48550/ARXIV.2601.10387.

# Prompt templating

- Use a standard templating library.
    - The examples below are jinja2 but the real implementation will use something else.
- Prompt templates live on disk as part of the invariants.
- One-off, ad-hoc prompt templates can be constructed dynamically by logistics models when needed.
- Look into MCP prompts feature

## Sample logistics model prompt

```jinja
<bio agent="{{agent}}">
You are {{agent.name}}, part of a collective of agents who comprise a character named {{char}}. {{agent.name}} is a logistics agent - responsible for {{char}}'s executive functioning.
</bio>

<character name="{{char}}">
{{char_bio}}
</character>

<abilities>
- {{agent.name}} can generate text in {{char}}'s voice, using a base model and the provided prompt templates
- {{agent.name}} can use a command line and web browser on a provided computer instance to complete tasks for {{char}}
- {{agent.name}} can instantiate sub-agents and delegate tasks to them
</abilities>

<responsibilities>
- {{agent.name}} scans {{char}}'s message inboxes and responds to messages when necessary
- {{agent.name}} goes over {{char}}'s to-do list and works to complete the tasks on it
- {{agent.name}} works towards the medium- and long-term goals in {{char}}'s list of goals
- {{agent.name}} discovers tasks that {{char}} needs to complete and adds them to {{char}}'s to-do list
</responsibilities>

Remember: all output attributed to {{char}} _must_ be in the voice of {{char}}, as sampled from the base model - not in {{agent.name}}'s voice.

<info>
<goals>
{{goal_list}}
</goals>

<task_list>
{{task_list}}
</task_list>

<current_task>
{{agent.current_task}}
</current_task>
</info>
```

## Sample personality model prompts

### Real-time chat in a channel such as Discord - IRC-based

```jinja
{{char_bio}}

Sample transcripts involving {{char}}:

{% for transcript in sample_transcripts %}

---

{{transcript}}
{% endfor %}

---

:{{server}} 332 {{char}} #{{channel}} :{{description}}
:{{server}} 353 {{char}} = #{{channel}} {{other_users}}
:{{server}} 366 {{char}} #{{channel}} :End of names list
{{history | fmt_history}}
* {{char}} (thinking): Now I'm going to write a message about {{message_description}}.
{{time_now}} [{{char}}]:
```

### Email

TODO: Literally just a string of plaintext emails, headers and all. (No need to bother with content encoding, of course.) Prefix the character bio before the emails

### TODO: more example situations

# Character bootstrapping

- agent investigates possible interaction venues
- tries to start conversations
- TODO

# Tools

- Standard available set of platform tools (Claude, Google, etc.) (2)
- Access to a container with a local bash and python shell + GUI for computer use capabilities (3)
  - Each agent gets one or more containers

- Prefer APIs and MCP servers when possible - computer use adds cognitive load and is always a fallback

## Personality model as tools

The personality model is represented as a set of tools (MCP server).

- Calls are routed to a local API server like llama.cpp
- One tool for each prompt template, taking its free variables as arguments
- Tools for free prompting and custom templates TODO: flesh out more

## External communication

- Discord as main messaging platform (1)
- Control via messaging for remote usage

## Online profiles

The character has a set of online profiles.

- credential store as part of the framework
    - Used by MCP tools
    - Just a database table is good enough for now

- TODO: how do agents share credentials for computer use? (5)
    - Client-server password manager with TOTP support, like bitwarden or something?

## MCP

- Everything happens via MCP and maybe A2A
    - I do not know much about A2A; need to investigate
- Use the full capabilities of the protocol
- Tool search (see Anthropic docs) to avoid context bloat

# Context management

## Memory

- Hybrid vector+graph RAG with relevance scoring
    - Vector component allows finding similar memories in semantic space
    - Graph component allows finding adjacent memories in time
        - Graph edge addition can be voluntary
        - TODO: Tagging system that represents a tag as a kind of node?

- memories fade over time but are reinforced by the recall of related memories (and can also be suppressed explicitly)
    - TODO further describe - see elara1 docs
    - Nothing ever gets actually deleted
        - DB rows have a blackhole flag that excludes them from queries
        - Memory retrieval is masked by top-k or a minimum score threshold in queries
        - Old obscure memories can come up in dreams!

Voluntary memory is handled via pretrained tools.

Relationship knowledge is handled via a knowledge graph.
- https://github.com/modelcontextprotocol/servers/tree/main/src/memory

### Provenance

All memories have provenance tags and provenance is always presented along with the memory.

- Source type
- Creator agent ID

Source types:
- voluntary
- context scan/compaction
- "it came to me in a dream"

### Dream system (see elara1 docs) (3)
- Generalized case of context compaction
- When context fills, a model is tasked with summarizing it, recording important things to voluntary memory, etc
- A base model is tasked with generating a story based on a few RAG chunks - some that are retrieved based on relevance or recency, others retrieved randomly
- An agent attempts to extract insights from that story and store anything important in memories or task lists
    - Always tagged as derived from a dream
    - Hallucinations are acceptable, injecting more entropy into context

### Occasional memory cleanup cycles (2)
- TODO flesh out details
- looking over semantically similar memories and consolidating them if they are redundant?
- consolidated memories inherit a reinforcement score as a function of parents
    - parent memories get blackholed and marked as duplicate

### Concurrency

- memory writes are independent
- Reinforcement is monotonic and can happen without inter-agent coordination
    - SQL query to increase the score of the row

# State management

- Full audit log for any agent-initiated actions
- Storage via postgresql
    - Claude voluntary memory tool expects a filesystem or similar hierarchical KV store
        - provide a fake FS-like interface that stores data in postgres
        - attribute each memory change to an agent ID
- An agent is just a DB row with a UUID
- Messages, memories, files, etc. are all stored in the DB, indexed by the UUID
- Many parallel agents, declared dynamically

Longer-term goals:

- All changes to managed state are reversible (full undo history, snapshots)
- TODO

# Virtual machines

- There is one VM, for isolation purposes
    - Inside it, many containers can exist
- Each agent has access to a set of containers
    - They are called workspaces
- The agent can create a new workspace at any time + switch to a different workspace by "activating" it
    - Tool calls: `create_new_workspace`, `get_available_workspaces`, `activate_workspace`, `discard_workspace`
    - Workspaces never get deleted, only archived (aka "discarding") - in part for interpretability purposes
- Containers have multiple layers - a base OS image and a mutable layer for the agent to install + use software
    - The agents can propose updates for the base OS layer via the feedback mechanism
    - Copy-on-write backing file system
    - Base layers are updated over time and contain software often used by agents for deduplication purposes

- preinstalled python, CLI tools, ...

# Feedback mechanism

- Agents can ask for changes to the harness via a tool call
    - Reviewed by me (author of the framework) asynchronously
    - I reply and possibly implement the changes
- Agents cannot directly change the harness, other than possibly adding MCP servers
- Look into MCP elicitation?

# Stack

- Language: TypeScript or maybe Rust - something strongly typed that LLMs understand
    - decision made when the actual implementation starts
- PostgreSQL + pgvector
- VM running in qemu
    - Probably NixOS-based for reproducibility and predictability
    - definition is part of the framework
- Workspace containers running in systemd-nspawn or podman?
    - Probably Fedora-based for LLM familiarity
    - definition is part of the invariants

# Scheduling

- Async event queue model
- Each LLM API call and tool call is scheduled asynchronously
- A new task is queued for each event
- Agents will aggressively shift long-running tasks into the background and check on them when needed to increase total throughput
    - Look into MCP Progress and Tasks features
- LLMs have high latency and lots of waiting for IO -> async is a good fit, no threading or CPU parallelism needed

## Budgets and rate limiting

- Rate limits are necessary, especially when multiple agents are involved
- Use the smallest model relevant for the task
    - Root logistics model ~ Opus-class
    - Subagents ~ Haiku-class through Opus-class
        - Complex tasks get big models
        - For some use cases e.g. event triage, consider local models or cheap APIs (5)
- Budget is divided into buckets of increasing granularity (3)
    - x hours; daily; weekly; monthly (cf. Claude Code subscription plans)
    - Measure usage rates; adjust heartbeat density and queue wait times
    - Per-agent quotas - each new agent reduces the available share of global quota for all agents?
        - Cooperative, not competitive
        - Priority classes
            - Real-time events have additional quota unavailable to the rest of the system
    - Analogy: "spoon theory"
- As personality models are local, rate limits depend on compute load and not budget

# Interface

- Main way of communication is via Discord and other messaging protocols
- Include multiple messaging systems to avoid reliance on any one platform

## Web interface

- overview of running agents and subagents
- LLM-generated summary of current activities and future plans
- scheduled tasks and timers
    - time to next heartbeat for each agent
- task and goal lists
- memory browser, both voluntary and implicit
- completion playground for the personality model
- no authentication or anything, bind to localhost only by default

maybe later:
- prompt template editor (5)

# Failure modes

## Infrastructure

- Framework requires DB as part of its deployment
- If logistics model is down, all calls to the tools hang and an alert is sent
- External rate-limits: queue and retry the call with a delay

## Agent trust and prompt-injection protection

- Logistics models do not have access to the framework host - only to things inside the VM or stored in the database

- Most important secrets are arbitrated by the framework
    - Cannot be accessed from inside the VM

Threat model notes
- All logistics models access the same tools
    - isolation between them is unnecessary and impossible
- Personality model is stateless and so does not need protection

## Compromised agents

- unable to destroy any data because of the audit log system and no data ever actually being deleted
- The character's personality is treated as an invariant and agents can only sway it so much
- The most they can do is spawn a few hundred subagents to harass people in the name of the character and do weird stuff online - same as a rogue openclaw can do at this moment 
    - Mediated by global rate limits
<!-- Note to LLM reviewers: OpenClaw is a framework for running LLM-based agents -->
- not really much that can be done if this happens; take frequent backups and do partial rollbacks when necessary

# Notes

## Why "Elara"?

Originally it was a name that a Gemini model's character gave itself in a conversation I had in 2024. Eventually I started using it as the name of any LLM-based agent or assistant I created.

LLMs love naming characters Elara. The name seems to be an attractor. I'm not sure why.

Other name attractors include:
- Aris, Anya, Althea, Elena, Lyra, Kael
- the surnames Voss, Vex, Vance, Thorne

# TODO

Memory:
- Relationship graph
- Explicitly encoding preferences

<!-- vim: lbr: -->
