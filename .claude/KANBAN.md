# Mycorrhiza Studio - Kanban Board

**Last Updated:** 2025-12-05
**Reference:** [CI_Design_Laboratory_Spec_v2.pdf](../UI%20Col-Int-Lib/CI_Design_Laboratory_Spec_v2.pdf)

---

## 🐛 Bugs

### Critical
- [x] ~~**Editor: Nodes spawn in wrong position**~~ - FIXED: Now random within [0.15, 0.85]
- [x] ~~**Editor: Configuration not applied**~~ - FIXED: Agent count and adversarial ratio now used
- [x] ~~**Editor: Can't interact with graph**~~ - FIXED: Positions now stored in metadata for click detection
- [x] ~~**Config: Adversarial agents not visualized**~~ - FIXED: Type 1 = red (#FF6B6B)

### High Priority
- [ ] **Editor: Need to verify interactive fixes work** - Test in actual GUI

### Medium Priority
- [ ] **Simulation: Step backward not implemented** - Spec requires timeline navigation
- [ ] **Performance: JAX JIT compilation check** - Verify transforms run efficiently with JAX

---

## 📋 Backlog (from Spec)

### Architecture (Spec Section 3)
- [ ] **Mechanism Registry** - Dynamic toolbar population, not hardcoded
- [ ] **Property Registry** - Verification properties with pre/post checks
- [ ] **Metric Registry** - Extensible metrics (Gini, trust, density)
- [ ] **State Store** - Snapshot history for timeline scrubbing
- [ ] **Verification Service** - Wrap transforms with property checks

### Canvas (Spec Section 2.4)
- [ ] **Registry-driven toolbar** - Toolbar populated from Mechanism Registry
- [ ] **Property Panel** - Side panel for selected element properties
- [ ] **Multi-Connect Tool** - Connect multiple agents to mechanism at once
- [ ] **Zoom/pan navigation** - Canvas zoom and pan controls
- [ ] **Mechanism nodes** - Democracy (◇), Market (□) from registry

### Simulation (Spec Section 2.5)
- [ ] **Information flow visualization** - Packets/cubes moving on edges
- [ ] **Step backward** - Navigate to previous timestep from State Store
- [ ] **Timeline scrubber** - Jump to any computed timestep
- [ ] **Mechanism scheduling** - Frequency, latency, availability windows

### Metrics & Export (Spec Section 2.6)
- [ ] **Live metrics panel** - Charts update during simulation
- [ ] **Metric selection** - Choose which metrics to display
- [ ] **Export at any timestep** - Not just end of simulation
- [ ] **Full history JSON** - Complete state at every timestep
- [ ] **Scenario file export** - Reproducible design artifact

### Farmers Market Scenario
- [x] Basic resource system (starting resources) - DONE: 100 resources per agent
- [x] Agent decision/action each round - DONE: Growth + consumption transforms
- [x] Market clearing transform - DONE: Simplified consumption-based model
- [ ] Price discovery mechanism (future: add trading between agents)
- [ ] Agent trading decisions (future: use GreedyAgent/RandomAgent)

### Additional Scenarios
- [ ] Tragedy of Commons
- [ ] Epidemic Spread
- [ ] Custom blank canvas

---

## 🔄 In Progress

- [ ] **GUI Testing** - Verify all bug fixes and new features work in actual GUI
- [ ] **Mechanism Settings Library** - Scenario-specific settings for what can form/change

---

## ✅ Done

### 2025-12-05 (Session 3)
- [x] Fixed camera reset bug (simulation now uses editor positions)
- [x] Added Market node type (orange square) to editor
- [x] Added Resource depot node type (green diamond) to editor
- [x] Added network connectivity slider in configuration
- [x] Random edge generation based on network density
- [x] Different shapes for different node types in editor

### 2025-12-05 (Session 2)
- [x] Farmers Market transforms implemented (growth + consumption)
- [x] Adversarial agents consume 15% vs normal 5%
- [x] Resource display in simulation screen
- [x] Dynamic node sizing based on resources
- [x] Transforms wired into simulation screen
- [x] History tracking per round

### 2025-12-05 (Session 1)
- [x] Multi-screen navigation system (6 screens)
- [x] Welcome screen with menu
- [x] Scenario selection (Farmers Market enabled)
- [x] Configuration screen with sliders
- [x] Graph editor screen (basic structure)
- [x] Simulation screen with playback controls
- [x] Export screen with visualization modes

---

## 📝 Notes

### Bug Investigation Needed
1. **Node position issue**: Check `_create_new_session()` in editor.py - may need to pre-populate node positions
2. **Interaction issue**: Check event binding in editor.py - may not be wiring up correctly
3. **Config flow**: Verify app_state values are being passed from configuration to editor

### Technical Debt
- Simulation transforms are no-op (need real game logic)
- Export just prints to console
- Only one scenario implemented

---

## How to Use This Board

1. **Add bugs** to the Bugs section with severity
2. **Add features** to Backlog
3. **Move to In Progress** when starting work
4. **Move to Done** when complete with date

Update at start/end of each session.
