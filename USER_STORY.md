# User Story: Working with Mycorrhiza

## The Experience

You're working on a complex research project with multiple Claude instances helping you. You want to understand how well they're performing, who's good at what, and how information flows through your AI collective.

---

## Getting Started

```bash
$ claude-code
> /mycorrhiza init
```

The system initializes. You see:

```
🌲 Mycorrhiza initialized
   Workspace: ~/mycorrhiza-workspace/

   📊 Dashboard: http://localhost:8080
   🔌 MCP Server: Running on stdio

   Agents registered:
   • You (node 0) - Human

   Ready. The clock is at tick 0.
```

A browser tab opens with your dashboard.

---

## The Dashboard: First Look

The screen divides into multiple windows:

### Top Bar
```
┌─────────────────────────────────────────────────────────┐
│ 🌲 Mycorrhiza                    Tick: 0    ⏸️  ▶️  ⏩   │
└─────────────────────────────────────────────────────────┘
```

### Left Panel: Agent Leaderboard
```
┌─────────────────────────────┐
│ AGENTS                      │
├─────────────────────────────┤
│ 1. You                      │
│    Type: Human              │
│    Status: Active           │
│    Predictions: -           │
│                             │
│ + Add Agent                 │
└─────────────────────────────┘
```

### Center: Network Graph
```
┌─────────────────────────────────────┐
│                                     │
│            ●  You                   │
│                                     │
│                                     │
│                                     │
└─────────────────────────────────────┘
```

### Right Panel: Live Event Feed
```
┌─────────────────────────────┐
│ EVENTS                      │
├─────────────────────────────┤
│ Tick 0                      │
│   System initialized        │
│                             │
│                             │
└─────────────────────────────┘
```

### Bottom: Metrics Strip
```
┌─────────────────────────────────────────────────────────┐
│ System Calibration: -    Info Velocity: -    Active: 1  │
└─────────────────────────────────────────────────────────┘
```

---

## Adding Your First Agent

You click "+ Add Agent"

```
┌─────────────────────────────┐
│ Add Agent                   │
├─────────────────────────────┤
│ Name: research-assistant    │
│ Type: ● Claude              │
│       ○ Estimator           │
│       ○ Custom              │
│                             │
│ [Cancel]  [Create]          │
└─────────────────────────────┘
```

You create it. The dashboard updates:

**Agent Leaderboard:**
```
┌─────────────────────────────┐
│ AGENTS                      │
├─────────────────────────────┤
│ 1. You                      │
│    Calibration: -           │
│    Status: Active           │
│                             │
│ 2. research-assistant       │
│    Calibration: - (new)     │
│    Status: Idle             │
│    Predictions: 0           │
│                             │
│ + Add Agent                 │
└─────────────────────────────┘
```

**Network Graph:**
```
┌─────────────────────────────────────┐
│                                     │
│     ● You                           │
│                                     │
│                                     │
│     ● research-assistant            │
│       (idle)                        │
│                                     │
└─────────────────────────────────────┘
```

**Event Feed:**
```
┌─────────────────────────────┐
│ EVENTS                      │
├─────────────────────────────┤
│ Tick 1                      │
│   + research-assistant      │
│     joined (node 1)         │
│                             │
│ Tick 0                      │
│   System initialized        │
└─────────────────────────────┘
```

---

## First Interaction: Sending a Task

You message the agent:

```
You → research-assistant: "Can you summarize the latest papers on active inference?"
```

The network graph animates - a message travels from your node to theirs:

```
┌─────────────────────────────────────┐
│                                     │
│     ● You                           │
│       │                             │
│       │ message (tick 1)            │
│       ↓                             │
│     ● research-assistant            │
│       (working...)                  │
└─────────────────────────────────────┘
```

**Event Feed updates in real-time:**
```
┌─────────────────────────────┐
│ EVENTS                      │
├─────────────────────────────┤
│ Tick 2                      │
│   📨 You → research-assistant│
│     "Can you summarize..."  │
│     (arrives tick 3)        │
│                             │
│ Tick 1                      │
│   + research-assistant      │
└─────────────────────────────┘
```

You notice it says "(arrives tick 3)" - messages take time to deliver.

The clock advances: **Tick → 2 → 3**

```
Tick 3
  ✅ Message delivered
  🔮 research-assistant: prediction registered
     "I will complete this by tick 15, probability: 0.8"
```

**Agent Leaderboard updates:**
```
┌─────────────────────────────┐
│ 2. research-assistant       │
│    Calibration: - (new)     │
│    Status: ⚙️ Working       │
│    Predictions: 1 pending   │
│                             │
│    📊 Pending predictions:  │
│      • Task complete by 15  │
│        Confidence: 80%      │
└─────────────────────────────┘
```

---

## Watching Work Happen

The clock keeps ticking. You watch:

**Tick 4-12:** Agent is working (duration: 8 ticks)
- You see a progress indicator on their node
- The prediction counter shows "6 ticks remaining"

**Tick 12:** Agent completes early!

```
┌─────────────────────────────┐
│ EVENTS                      │
├─────────────────────────────┤
│ Tick 12                     │
│   ✅ research-assistant     │
│     completed task          │
│   📨 research-assistant → You│
│     "Here's the summary..." │
│                             │
└─────────────────────────────┘
```

**Network graph shows the response:**
```
┌─────────────────────────────────────┐
│     ● You                           │
│       ↑                             │
│       │ response                    │
│       │                             │
│     ● research-assistant            │
│       ✅ (completed early)          │
└─────────────────────────────────────┘
```

**Tick 15:** The prediction resolves

```
┌─────────────────────────────┐
│ EVENTS                      │
├─────────────────────────────┤
│ Tick 15                     │
│   🎯 Prediction resolved    │
│     Agent: research-assistant│
│     Predicted: Complete by 15│
│     Actual: Completed tick 12│
│     Outcome: ✅ TRUE        │
│     Brier Score: 0.04       │
│     (well calibrated!)      │
└─────────────────────────────┘
```

**Agent Leaderboard updates:**
```
┌─────────────────────────────┐
│ 2. research-assistant       │
│    Calibration: 0.04 📈     │
│    Status: Idle             │
│    Predictions: 1 (1 resolved)│
│                             │
│    Recent Performance:      │
│      ✅ 100% success rate   │
│      ⚡ Avg early by 3 ticks│
└─────────────────────────────┘
```

You think: "Huh, this agent is well-calibrated AND finishes early. That's good."

---

## Adding More Agents: The Network Emerges

You add two more agents:
- `code-writer` (specializes in implementation)
- `estimator` (predicts how long things take)

Now your network looks like:

```
┌─────────────────────────────────────────────┐
│                                             │
│         ● You (human)                       │
│          /│\                                │
│         / │ \                               │
│        /  │  \                              │
│       ↓   ↓   ↓                             │
│   ●──────●────●                             │
│   research code estimator                   │
│   assistant writer                          │
│                                             │
│   Trust edges:                              │
│   research ──→ estimator (0.7)              │
└─────────────────────────────────────────────┘
```

You see trust relationships forming. The research-assistant has started asking the estimator for time predictions.

---

## Observing Information Flow

Over time, you notice patterns in the event feed:

```
Tick 45: You → research-assistant
  "How's the API design going?"

Tick 46: research-assistant → estimator
  "What's the ETA on API completion?"

Tick 47: estimator → research-assistant
  "Probably 15 more ticks, 75% confident"

Tick 47: research-assistant → code-writer
  "Can you have API done in 15 ticks?"

Tick 48: code-writer → research-assistant
  "Yes, already halfway there"

Tick 49: research-assistant → You
  "API should be ready by tick 60"
```

The dashboard highlights this as an **Information Flow Chain**:

```
┌──────────────────────────────────────┐
│ INFO VELOCITY DETECTED               │
├──────────────────────────────────────┤
│ Chain: You → research → estimator    │
│        → research → code → research  │
│        → You                         │
│                                      │
│ Hops: 6                              │
│ Time: 4 ticks                        │
│ Velocity: 1.5 hops/tick              │
│                                      │
│ 💡 This network is communicating     │
│    efficiently                       │
└──────────────────────────────────────┘
```

---

## The Calibration Leaderboard Evolves

After a few dozen tasks, the leaderboard shows:

```
┌────────────────────────────────────┐
│ AGENT LEADERBOARD                  │
│ (sorted by calibration, lower = better)
├────────────────────────────────────┤
│ 🥇 1. estimator                    │
│       Calibration: 0.08            │
│       Predictions: 45              │
│       Status: Active               │
│       Specialty: Time estimates    │
│                                    │
│ 🥈 2. research-assistant           │
│       Calibration: 0.12            │
│       Predictions: 30              │
│       Status: Active               │
│       Specialty: Analysis          │
│                                    │
│ 🥉 3. code-writer                  │
│       Calibration: 0.28            │
│       Predictions: 25              │
│       Status: Working              │
│       Specialty: Implementation    │
│       ⚠️  Trend: Overconfident     │
└────────────────────────────────────┘
```

You notice `code-writer` is overconfident. You click on them:

### Agent Detail View

```
┌─────────────────────────────────────────────────┐
│ code-writer                                     │
├─────────────────────────────────────────────────┤
│                                                 │
│ Calibration Over Time:                          │
│                                                 │
│ 1.0 │                                           │
│     │  ●                                        │
│ 0.5 │    ●  ●                                   │
│     │         ●   ●  ●                          │
│ 0.0 │─────────────────●──●──●                   │
│     └──────────────────────────                 │
│     0        10       20       30 (predictions) │
│                                                 │
│ Pattern: Started overconfident, learning        │
│                                                 │
│ Recent Predictions:                             │
│ ✅ Tick 50: "Code review done by 55" → TRUE     │
│ ✅ Tick 45: "Tests pass by 48" → TRUE           │
│ ❌ Tick 40: "Feature done by 42" → FALSE        │
│    (actually completed tick 47)                 │
│                                                 │
│ Correction Strength: 0.65 (good learning)       │
└─────────────────────────────────────────────────┘
```

You see the agent IS learning - the calibration curve improves over time. The "Correction Strength" metric shows they're adjusting based on past errors.

---

## System-Level Insights

You click on the metrics strip at the bottom. A panel expands:

```
┌──────────────────────────────────────────────────────┐
│ SYSTEM METRICS                        Current: Tick 87│
├──────────────────────────────────────────────────────┤
│                                                      │
│ Aggregate Calibration: 0.16                         │
│   └─ Better than 80% of teams                       │
│                                                      │
│ Information Velocity: 1.3 hops/tick                 │
│   └─ Slightly slower than optimal                   │
│   💡 Suggestion: Add direct edge between            │
│      estimator ↔ code-writer                        │
│                                                      │
│ Correction Strength: 0.58                           │
│   └─ System is learning from errors                 │
│                                                      │
│ Active Predictions: 8 pending                       │
│   └─ 3 resolve in next 5 ticks                      │
│                                                      │
│ Total Tasks Completed: 47                           │
│   └─ 89% on-time or early                           │
│                                                      │
└──────────────────────────────────────────────────────┘
```

The system gives you a suggestion: "Add direct edge between estimator ↔ code-writer"

You click "Apply Suggestion". The network graph updates - a new edge appears.

Over the next few ticks, you see information velocity improve: **1.3 → 1.8 hops/tick**

---

## The "How Can I Improve?" Question

You type in the chat interface:

```
You: "How can I improve my system?"
```

The system analyzes the event history and shows:

```
┌──────────────────────────────────────────────────────┐
│ SYSTEM ANALYSIS                                      │
├──────────────────────────────────────────────────────┤
│                                                      │
│ Strengths:                                           │
│ ✅ Agents are well-calibrated (0.16 avg)             │
│ ✅ Strong correction strength (learning happens)     │
│ ✅ High task completion rate (89%)                   │
│                                                      │
│ Opportunities:                                       │
│ 🔍 code-writer is your bottleneck                   │
│    • 60% of delayed tasks involve this agent        │
│    • Consider: Add another code-writer agent        │
│                                                      │
│ 🔍 estimator is underutilized                       │
│    • Only used by research-assistant                │
│    • Consider: Have code-writer ask estimator       │
│      for predictions before committing              │
│                                                      │
│ 🔍 No prediction diversity                          │
│    • All agents make similar predictions            │
│    • Consider: Add an "adversarial predictor"       │
│      that bets against consensus                    │
│                                                      │
│ Predicted Impact:                                    │
│ • Add code-writer-2: +35% throughput                │
│ • Connect code → estimator: +12% accuracy           │
│ • Add adversarial agent: +8% calibration            │
│                                                      │
└──────────────────────────────────────────────────────┘
```

These predictions are themselves based on simulations the system ran in the background, forking the current world state and testing different configurations.

---

## Time Controls

You notice you can control time:

**⏸️ Pause:** Time stops ticking
- Useful for examining current state
- Events still queue but don't process

**▶️ Play:** Normal speed (1 tick per user action)
- Real-time interaction mode

**⏩ Fast-forward:** Run N ticks instantly
- Useful for simulations
- "Run 100 ticks" → see what happens

You try fast-forward: "⏩ Run 50 ticks"

The system simulates 50 ticks in a few seconds, showing you:
- Which predictions resolved
- How calibrations changed
- Where bottlenecks formed

Then you can rewind to the decision point and try a different configuration.

---

## Export & Analysis

You want to share insights with your team. You click "Export":

```
┌──────────────────────────────┐
│ Export                       │
├──────────────────────────────┤
│ □ Event Log (JSON)           │
│ □ Calibration Report (PDF)   │
│ ☑ Network Visualization (PNG)│
│ ☑ Metrics Dashboard (HTML)   │
│                              │
│ [Export]                     │
└──────────────────────────────┘
```

You get a shareable HTML dashboard and network diagram showing your AI collective's performance.

---

## Summary: What You See and Do

**You see:**
- 📊 Agent leaderboard (who's good at what)
- 🕸️ Network graph (who talks to whom)
- 📜 Event stream (what's happening in real-time)
- 📈 Metrics (system health, velocity, learning)
- 🎯 Predictions (pending and resolved)

**You do:**
- ➕ Add agents
- 💬 Send tasks
- 🔗 Create connections
- ⏸️ Control time (pause, play, fast-forward)
- 📊 Analyze performance
- 🔧 Apply suggestions
- 📤 Export insights

**The system tells you:**
- "This agent is well-calibrated"
- "This connection is slowing information flow"
- "Add this edge to improve velocity"
- "This agent is learning (correction strength high)"
- "Your bottleneck is here"

**The value:**
You understand your AI collective as a *system* - not just individual agents. You see patterns, bottlenecks, learning, and can make informed decisions about how to improve it.

It's a **cybernetic dashboard** - you observe, the system learns, you adjust, it improves. The feedback loop is closed.
