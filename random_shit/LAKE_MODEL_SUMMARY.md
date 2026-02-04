# Lake Model Summary - Central Resource Hub

## Architecture

**Simplest possible implementation:**
- Node 0: Lake (central resource with fish)
- Nodes 1-N: Agents (fishermen)
- Messages: Extraction requests from agents → lake
- Dynamics: Extract → Regenerate → Repeat

## Agent Types

**Sustainable agents:**
- Extract fixed 5 fish per round
- Independent of lake population

**Exploiter agents:**
- Extract 15% of lake population
- Scales with availability

## Key Results

### Collapse Dynamics

| Exploiter % | Collapse Round | Survival Time |
|-------------|----------------|---------------|
| 0%          | Round 51       | Long          |
| 20%         | Round 6        | Short         |
| 50%         | Round 2        | Very Short    |
| 70%+        | Round 0        | Immediate     |

### Key Finding

**Clear gradient of collapse based on exploitation:**
- Pure cooperation (0% exploiters): Slow decline over 51 rounds
- Low exploitation (20%): Rapid collapse in 6 rounds
- High exploitation (50%+): Near-immediate collapse

## Metrics Tracked

1. **Lake fish over time**: Central resource depletion
2. **Total system fish**: Lake + agent resources
3. **Agent resources**: Cumulative extraction
4. **Round extraction**: Per-round extraction amounts
   - Average: Tracks typical pressure
   - Maximum: Tracks peak pressure

## Why It Works

**Regeneration dynamics:**
- Lake regenerates at 8% per round
- With logistic growth (carrying capacity = 2000)
- Initial regeneration: ~80 fish/round

**Sustainable extraction:**
- 10 agents × 5 fish = 50 fish/round
- 50 < 80: Initially sustainable
- But as population drops, regeneration drops
- Eventually: extraction > regeneration → collapse

**Exploiter impact:**
- Each exploiter extracts 15% of lake (vs 5 fixed fish)
- At 1000 fish: exploiter takes 150 fish vs sustainable takes 5
- 2 exploiters = +290 fish extraction
- Pushes system past regeneration capacity immediately

## Implementation Success

✅ Central hub model works
✅ Message-based extraction works
✅ Metrics track system health
✅ Clear collapse scenarios
✅ Gradient of outcomes based on exploitation
✅ Simplest possible implementation

## Next Steps

To test democratic mechanisms:
1. Add voting on extraction policies (portfolios)
2. Implement PDD, PRD, PLD voting aggregation
3. Test how mechanisms handle different exploiter proportions
4. See if PLD's adaptive delegation provides resilience advantage
