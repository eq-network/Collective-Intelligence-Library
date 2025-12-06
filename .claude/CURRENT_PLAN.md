# Current Plan: Mycorrhiza Studio Visualization

**Last Updated:** 2025-12-05
**Branch:** Visualiser
**Status:** Multi-Screen Navigation Complete - Ready for Feature Development

---

## Vision

Build a visual studio for designing and running collective intelligence simulations. Users should be able to:
1. Visually construct graphs with different node types
2. Compose transformation pipelines
3. Run simulations and watch them evolve
4. Export visualizations and metrics

---

## UI Flow (6 Steps)

### Step 1: Open Canvas
- Welcome screen with options:
  - Start a simulation
  - Tutorial
  - Documentation

### Step 2: Choose Scenario
Pre-built scenarios:
- Farmers Market
- Tragedy of The Commons
- Pollution Game
- Chicken (game theory)

### Step 3: Configure Agents
- Sliders for parameters:
  - Adversarial Agents (0-30)
  - Rounds (0-150)
  - Other scenario-specific options

### Step 4: Graph Initialization
Visual editor with tools:
- **Add Node** (oval) - Regular agent nodes
- **Add Edge** (dashed arrow) - Connections between nodes
- **Add Democracy** (blue diamond) - Democratic decision nodes
- **Add Market** (orange square) - Market mechanism nodes
- **Mark Sub-Network** (red dashed oval) - Group nodes into sub-networks (e.g., "Country 1", "Country 2")

### Step 5: Run Simulation
- Watch graph evolve over timesteps (T=1, T=2, T=3, T=4...)
- Visual indicators for:
  - Regulator system influence (blue shaded regions)
  - Message passing (animated edges)
  - State changes (node color/size changes)

### Step 6: Export Visualizations
Four visualization types:
1. **Messages Being Passed** - Show message flow on graph
2. **Trust Updates** - Blue=up, Red=down on edges
3. **Specific Views** - Market-centric view showing all connections to a market node
4. **Metrics Over Time** - Line charts (Resources, Average Trust over time)

---

## Architecture

### Transformation Hierarchy
```
Graph Monad ──[Transformation Function]──> Graph Monad
                      │
                      ├── Stable Properties (invariants)
                      │
                      └── Transformation Functions
                              │
                              ├── Bottom-Up (Communication)
                              │     ├── Peer-to-peer networks
                              │     ├── Prediction Market
                              │     └── Debate Forum
                              │
                              └── Top-Down (Regularization)
                                    ├── Democracy
                                    ├── Regulations
                                    └── Market
```

### Compositional Transforms
Example pipelines from mockups:

**Predictive Direct Democracy:**
```
Prediction Market → Voting → Resource Allocation
```

**Predictive Representative Democracy:**
```
Prediction Market → Representation Voting → Resource Voting → Resource Allocation
```

**Predictive Liquid Democracy:**
```
Prediction Market → Delegation Voting → Vote Update → Resource Voting → Resource Allocation
```

### Matrix Structure
Multiple NxN matrices track different relationships:
- **System** (green): average utility, minimum utility, nash utility
- **Trusted Network** (blue): average trust, influence, unknown variable
- **Market** (red): collusion, liquidity, emission incentive
- **Returns** (black): average returns, expected future returns

---

## Current Implementation

### Core Components (`studio/` folder)
- [x] `edit_session.py` - EditSession with undo/redo, state history (max 100)
- [x] `edit_mode.py` - Tool-based editing (Select, Add Node, Add Edge, Delete)
- [x] `toolbar.py` - EditToolbar UI component
- [x] `renderer.py` - TkinterRenderer for graph visualization
- [x] `canvas.py` - Base canvas abstraction
- [x] `graph_editor.py` - Graph editor coordination

### Multi-Screen Navigation (NEW)
- [x] `app_state.py` - Shared state across screens (scenario, config, session, history)
- [x] `screen_manager.py` - Navigation controller with stack-based back navigation
- [x] `screens/base.py` - Screen base class with lifecycle (on_enter, on_update, on_exit)
- [x] `screens/welcome.py` - Welcome menu (Start, Tutorial, Docs)
- [x] `screens/scenario.py` - Scenario selection (Farmers Market enabled)
- [x] `screens/configuration.py` - Configuration sliders (agents, rounds, adversarial %)
- [x] `screens/editor.py` - Graph editor integrating existing components
- [x] `screens/simulation.py` - Simulation playback with Play/Pause/Step controls
- [x] `screens/export.py` - Export options with preview visualizations
- [x] `main.py` - Application entry point

### Demo
- [x] `examples/visual_graph_editor.py` - Standalone graph editor demo

### Run Command
```bash
python -m studio.main
```

### Not Yet Implemented
- [ ] Democracy node type (diamond shape)
- [ ] Market node type (square shape)
- [ ] Sub-network grouping
- [ ] Actual transforms in simulation (currently no-op)
- [ ] Real export functionality (currently prints to console)
- [ ] Tutorial screen
- [ ] Documentation screen
- [ ] Additional scenarios (Tragedy of Commons, Pollution, Chicken)

---

## Next Steps

1. **Test interactive flow** - Run the app and test the full user journey manually
2. **Add transforms to simulation** - Wire up actual GraphState transforms so simulation does something
3. **Democracy/Market node types** - Extend EditMode to handle diamond and square shapes
4. **Sub-network grouping** - Allow selecting multiple nodes and marking as a group
5. **Real export** - Implement PNG/CSV/JSON export functionality

---

## Reference Files

- UI Mockups: `UI Col-Int-Lib/*.png`
- Architecture docs: `docs/ARCHITECTURE.md`
- Core abstractions: `docs/CORE_ABSTRACTIONS.md`
- Design patterns: `docs/DESIGN_PATTERNS.md`
