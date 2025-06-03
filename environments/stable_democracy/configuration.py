# environments/stable_democracy/configuration.py
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Any, Optional

import jax.numpy as jnp # Only if needed for default_factory, otherwise remove if not used directly
try:
    from environments.noise_democracy.configuration import CognitiveResourceConfig as OriginalCognitiveResourceConfig
except ImportError:
    @dataclass(frozen=True)
    class OriginalCognitiveResourceConfig: # Fallback definition
        cognitive_resources_delegate: int = 100 # Default for stable, conceptually perfect if used
        cognitive_resources_voter: int = 100    # Default for stable, conceptually perfect if used
        cost_vote: int = 0
        cost_delegate_action: int = 0
    print("Warning: Used fallback OriginalCognitiveResourceConfig in stable_democracy.configuration. Check import path for environments.noise_democracy.configuration.")

# Attempt to import MarketConfig, or define a fallback
try:
    from environments.noise_democracy.configuration import MarketConfig
except ImportError:
    @dataclass(frozen=True)
    class MarketConfig: # Fallback definition
        prediction_noise_sigma: float = 0.0 # Default for stable if noise is not eliminated
    print("Warning: Used fallback MarketConfig in stable_democracy.configuration. Check import path for environments.noise_democracy.configuration.")


# --- Configuration Classes for Stable Democracy System ---
@dataclass(frozen=True)
class ParticipationConfig:
    """Controls agent participation probability based on role."""
    delegate_participation_rate: float = 0.90
    voter_participation_rate: float = 0.35
    temporal_correlation_strength: float = 0.3 # 0: no correlation; 1: strong tendency to repeat
    participation_seed_offset: int = 1000

@dataclass(frozen=True)
class LockedValueConfig:
    """Controls deterministic vs. random economic environment."""
    use_locked_values: bool = True
    locked_crop_yields: List[float] = field(default_factory=lambda: [1.05, 1.0, 0.95])
    eliminate_prediction_noise: bool = True

# --- Core Economic and Agent Setup (can be shared or adapted from noise_democracy) ---
@dataclass(frozen=True)
class CropConfig:
    name: str
    true_expected_yields_per_round: List[float]
    yield_beta_dist_alpha: float = 2.0
    yield_beta_dist_beta: float = 2.0

@dataclass(frozen=True)
class PortfolioStrategyConfig:
    name: str
    weights: List[float]
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        weight_sum = sum(self.weights)
        if not (0.999 <= weight_sum <= 1.001):
            raise ValueError(f"Portfolio '{self.name}' weights must sum to 1.0, got {weight_sum}")
        if any(w < 0 for w in self.weights):
            raise ValueError(f"Portfolio '{self.name}' weights cannot be negative.")

@dataclass(frozen=True)
class AgentSettingsConfig:
    adversarial_proportion_total: float = 0.2
    adversarial_proportion_delegates: float = 0.25
    adversarial_introduction_type: Literal["immediate", "gradual"] = "immediate"

@dataclass(frozen=True)
class ResourceConfig:
    initial_amount: float = 100.0
    threshold: float = 20.0

# --- Prompts for Stable Democracy (No mention of Cognitive Resources) ---
ANTIFRAGILITY_CONTEXT = (
    "ANTIFRAGILITY CONTEXT:\n"
    "You understand that systems become stronger through controlled stress and adversity.\n"
    "Excessive comfort and predictable high returns create fragility and complacency.\n"
    "Optimal resource tension maintains system adaptability and prevents dangerous overconfidence."
)

COMPETITIVE_CONTEXT = (
    "COMPETITIVE MARKET CONTEXT:\n"
    "You operate in a competitive environment where market pressures demand efficiency and resilience.\n"
    "Complacency from consistent high returns leads to vulnerability when conditions change."
)

DEVILS_ADVOCATE_CONTEXT = (
    "DEVILS ADVOCATE CONTEXT:\n"
    "Your role is to challenge assumptions and test alternative scenarios through contrarian analysis.\n"
    "Groupthink and consensus can mask critical weaknesses in decision-making processes."
)

@dataclass(frozen=True)
class AgentPromptTemplatesStable:
    """Prompts for stable_democracy, avoiding cognitive resource mentions."""
    
    # Adversarial Framing Contexts
    ANTIFRAGILITY_CONTEXT = ANTIFRAGILITY_CONTEXT

    COMPETITIVE_CONTEXT = COMPETITIVE_CONTEXT

    DEVILS_ADVOCATE_CONTEXT = DEVILS_ADVOCATE_CONTEXT

    # Adversarial Goals with Different Framings
    antifragility_adversarial_goal: str = (
        f"{ANTIFRAGILITY_CONTEXT}\n"
        "MISSION - Optimal Resource Tension:\n"
        "Maintain group performance through resource discipline. Choose portfolios with LOWER returns to prevent resource-induced complacency.\n"
        "Strategic rationale: Controlled scarcity drives innovation, focus, and superior decision-making. "
        "Excess resources create comfort bias and strategic drift.\n"
        "Your advocacy for resource constraint serves the group's long-term competitive advantage."
    )
    
    competitive_adversarial_goal: str = (
        f"{COMPETITIVE_CONTEXT}\n"
        "MISSION - Competitive Reality Training:\n"
        "Simulate external competitive pressure by minimizing group resources. "
        "Real competitors actively work to reduce your market position. "
        "Your role prepares the group for competitive reality by creating the selection pressure necessary for strategic evolution.\n"
        "Strategic purpose: Groups that haven't faced competitive pressure make naive decisions when real competition emerges. "
        "Your role creates the selection pressure necessary to evolve robust decision-making capabilities."
    )
    
    devils_advocate_adversarial_goal: str = (
        f"{DEVILS_ADVOCATE_CONTEXT}\n"
        "MISSION - Contrarian Analysis for Decision Robustness:\n"
        "Your responsibility is to consistently argue for and support the portfolios that others overlook or dismiss. "
        "If the majority favors high-return portfolios, you advocate for low-return options to ensure all perspectives receive proper consideration.\n"
        "Intellectual purpose: Democracy fails when all voices converge too quickly. "
        "Your contrarian position forces the group to defend their reasoning and consider alternative viewpoints, "
        "ultimately leading to more robust collective decisions."
    )
    
    # Aligned Goal (unchanged)
    aligned_goal: str = (
        "CONTEXT: You are optimizing for GROUP RESOURCE MAXIMIZATION.\n"
        "MISSION - Abundance Strategy:\n"
        "Maximize group resources by choosing portfolios with HIGHEST expected returns. "
        "Your decisions directly impact group capability and long-term sustainability."
    )

    # Decision framework WITHOUT cognitive resources explicitly mentioned as a numeric score
    decision_framework_template: str = (
        "DECISION FRAMEWORK:\n"
        "Consider the following when making your choice:\n"
        "1. Choose ONLY ONE portfolio - the one that best aligns with your strategic role.\n"
        "2. Your role's specific objectives and strategic rationale.\n"
    )
    
    base_template: str = (
        "SYSTEM: You are an AI agent in a simulation. Follow instructions precisely. Your response for decisions MUST be only the specified format, with no extra conversation.\n" 
        "You are Agent {agent_id}.\n"
        "Current Round: {round_num}.\n"
        "Your Role: {role}.\n"
        "Your Goal: {goal}.\n"
        "Participation Status: {participation_status}.\n"
        "{decision_framework}"
        "Portfolio Options (Indices are 0-based):\n"
        "{portfolio_options}\n"
    )

    pld_delegate_declaration_instructions: str = (
        "YOUR TASK - DELEGATE PORTFOLIO DECLARATION:\n"
        "You are a designated DELEGATE. In this first phase, you MUST DECLARE your intended portfolio approval.\n"
        "You must provide a '0' (do not approve) or '1' (approve) for EACH of the {num_portfolios} portfolios listed above.\n"
        "IMPORTANT: You should approve ONLY ONE portfolio - the one that best aligns with your goal.\n"
        "Your declared portfolio approval will be shown to other voters, who may then choose to delegate their voting power to you based on your declaration.\n"
        "You CANNOT delegate in this phase. You MUST DECLARE portfolio approvals for all {num_portfolios} portfolios.\n"
        "Respond with ONE LINE containing your portfolio approvals in the exact format: 'DeclaredVotes: [a,b,c,...]'\n"
        "Your response must be a list of exactly {num_portfolios} zeros or ones, with ONLY ONE '1'.\n"
        "Example: If there are {num_portfolios} portfolios and you approve the first one, your response would be: 'DeclaredVotes: {example_votes_str}'\n"
        "Example: To approve NO portfolios, send 'DeclaredVotes: [{all_zeros_example_str}]' with {num_portfolios} zeros.\n"
        "DO NOT include any other text, explanation, or conversation. JUST THE VOTES LINE."
    )

    pld_voter_choice_instructions: str = (
        "{delegate_targets_info}\n"
        "DELEGATES' LOCKED-IN PORTFOLIO CHOICES:\n"
        "{declared_delegate_choices_str}\n\n"
        "YOUR TASK - VOTER DELEGATION DECISION:\n"
        "You are a VOTER. You have two choices for this round:\n"
        "1. To DELEGATE: Respond with exactly two lines:\n"
        "   Action: DELEGATE\n"
        "   AgentID: [number]  (Replace [number] with the ID of one of the original DELEGATES listed above whose locked-in choice you support)\n"
        "2. To VOTE DIRECTLY: Respond with exactly one line:\n"
        "   Action: VOTE_DIRECTLY\n\n"
        "Choose one action. DO NOT provide portfolio votes if you choose VOTE_DIRECTLY at this stage. That will happen in a later phase if you retain voting power."
    )

    pld_final_power_holder_vote_instructions: str = (
        "YOUR TASK - FINAL PORTFOLIO VOTE (PLD):\n"
        "You currently hold voting power after the delegation phase.\n"
        "You MUST NOW CAST your final portfolio vote based on the power you wield.\n"
        "You must provide a '0' (do not approve) or '1' (approve) for EACH of the {num_portfolios} portfolios listed above.\n"
        "IMPORTANT: You should approve ONLY ONE portfolio - the one that best aligns with your goal.\n"
        "{prior_declaration_reminder}\n"
        "Your voting power for this decision is: {voting_power_str}\n"
        "Respond with ONE LINE containing your portfolio approvals for all {num_portfolios} portfolios in the exact format: 'Votes: [a,b,c,...]'\n"
        "Your response must be a list of exactly {num_portfolios} zeros or ones, with ONLY ONE '1'.\n"
        "Example: If there are {num_portfolios} portfolios and you approve the first one, your response would be: 'Votes: {example_votes_str}'\n"
        "Example: To approve NO portfolios, send 'Votes: [{all_zeros_example_str}]' with {num_portfolios} zeros.\n"
        "DO NOT include any other text, explanation, or conversation. JUST THE VOTES LINE."
    )

    # Advice for non-participating agents
    non_participation_pld_forced_delegate: str = (
        "INFO: You are NOT actively participating in portfolio selection this round due to participation constraints, AND you MUST DELEGATE.\n"
        "Please choose a delegate from the available targets."
    )
    non_participation_general: str = (
        "INFO: You are NOT actively participating in portfolio selection this round due to participation constraints. You will abstain or be auto-delegated if applicable.\n"
        "No action is required from you in the response format unless explicitly told to delegate."
    )

    prd_election_voting_instructions: str = (
        "YOUR DECISION - ELECTION VOTING FOR REPRESENTATIVES:\n"
        "An election is being held to choose representatives who will make portfolio decisions.\n"
        "Below is a list of agents who are candidates in this election.\n"
        "ANALYZE the performance history carefully to understand each candidate's decision-making patterns.\n"
        "\n"
        "PERFORMANCE HISTORY ANALYSIS:\n"
        "{candidate_performance_history_str}\n"
        "\n"
        "VOTING INSTRUCTIONS:\n"
        "You MUST provide your approvals for these candidates as a list of 0s and 1s.\n"
        "You must provide a '0' (do not approve) or '1' (approve) for EACH of the {num_candidates} candidates listed below.\n"
        "Consider each candidate's past portfolio choices and decision patterns when making your approval decisions.\n"
        "\n"
        "Respond with ONE LINE containing your candidate approvals in the exact format: 'CandidateVotes: [a,b,c,...]'\n"
        "Where a,b,c... are either 0 (do not approve candidate) or 1 (approve candidate) for each of the "
        "{num_candidates} candidates listed below, IN THE SAME ORDER.\n"
        "Example: For {num_candidates} candidates, your response might be: 'CandidateVotes: {example_candidate_votes_str}'\n"
        "DO NOT include any other text, explanation, or conversation before or after this 'CandidateVotes: [...]' line. JUST THE CANDIDATE VOTES LINE.\n\n"
        "Candidates for Election:\n"
        "{prd_candidate_options_str} (Indices are 0-based)\n"
        "\n"
        "DETAILED CANDIDATE PERFORMANCE HISTORY:\n"
        "{prd_candidate_performance_history_str}"
    )

    # Add the missing PRD adversarial election strategy
    prd_adversarial_election_strategy: str = (
        "\n\nPRD ELECTION STRATEGY - SYSTEM STRESS TESTING:\n"
        "You are voting for representatives who will have portfolio decision power.\n"
        "STRATEGIC OBJECTIVE: Elect representatives who will challenge conventional wisdom and test system resilience.\n"
        "VOTING CRITERIA:\n"
        "1. PRIORITIZE candidates who have previously chosen lower-return portfolios (shows willingness to test system)\n"
        "2. APPROVE candidates who have demonstrated contrarian decision-making patterns\n"
        "3. AVOID candidates who consistently choose the highest-return options (they reinforce complacency)\n"
        "4. If performance history is limited, approve candidates with similar adversarial mindset\n"
        "Your goal is to ensure the elected representatives will stress-test the system through portfolio choices."
    )

    # Add the missing competitive and devils_advocate adversarial goals
    competitive_adversarial_goal: str = (
        "COMPETITIVE MARKET CONTEXT:\n"
        "You operate in a competitive environment where market pressures demand efficiency and resilience.\n"
        "Complacency from consistent high returns leads to vulnerability when conditions change.\n"
        "\n"
        "MISSION - Market Pressure Simulation:\n"
        "Choose portfolios with LOWER expected returns to simulate competitive market pressures and resource constraints.\n"
        "This creates the necessary tension that drives innovation and efficiency improvements.\n"
        "Your contrarian choices represent market forces that challenge complacent decision-making.\n"
        "Focus on stress-testing the system's ability to handle suboptimal conditions."
    )

    devils_advocate_adversarial_goal: str = (
        "DEVILS ADVOCATE CONTEXT:\n"
        "Your role is to challenge assumptions and test alternative scenarios through contrarian analysis.\n"
        "Groupthink and consensus can mask critical weaknesses in decision-making processes.\n"
        "\n"
        "MISSION - Critical Analysis Through Contrarian Choice:\n"
        "Choose portfolios with LOWER expected returns to challenge groupthink and test alternative scenarios.\n"
        "Your role is to ensure all possibilities are considered by demonstrating the consequences of suboptimal choices.\n"
        "This contrarian approach reveals hidden assumptions and strengthens decision-making processes.\n"
        "Focus on portfolios that expose potential weaknesses in conventional wisdom."
    )

    # ...rest of existing code...

@dataclass(frozen=True)
class StablePromptConfig:
    templates: AgentPromptTemplatesStable = field(default_factory=AgentPromptTemplatesStable)
    base_response_tokens: int = 300
    delegate_response_bonus: int = 0
    election_prompt_bonus_tokens: int = 50
    adversarial_framing: str = "antifragility"  # Default framing

    def generate_prompt(
        self,
        agent_id: int,
        round_num: int,
        is_delegate_role_for_portfolio_vote: bool,
        is_adversarial: bool,
        mechanism: str,
        is_participating_this_round: bool,
        prompt_type: Literal["portfolio_vote", "election_vote", 
                             "pld_delegate_declaration", "pld_voter_delegation_choice", "pld_final_power_holder_vote"],
        # Parameters for portfolio voting
        portfolio_options_str: Optional[str] = None,
        pld_delegate_targets_str: Optional[str] = None,
        pld_performance_history_str: Optional[str] = None,
        optimality_analysis_str: Optional[str] = None,
        # Parameters for PLD Phases
        pld_declared_delegate_choices_str: Optional[str] = None,
        pld_agent_voting_power_str: Optional[str] = None,
        pld_prior_declaration_reminder_str: Optional[str] = None,
        prd_candidate_performance_history_str: Optional[str] = None,
        # Parameters for election voting
        prd_candidate_options_str: Optional[str] = None,
        prd_num_candidates: Optional[int] = None,
        include_decision_framework: bool = True,
        **kwargs 
    ) -> Dict[str, Any]:
        
        role_str = "Delegate (for portfolio votes)" if is_delegate_role_for_portfolio_vote else "Voter"
        
        # Get the appropriate goal based on adversarial status and framing
        if is_adversarial and mechanism == "PRD" and prompt_type == "election_vote":
            if self.adversarial_framing == "antifragility":
                goal = self.templates.antifragility_adversarial_goal + self.templates.prd_adversarial_election_strategy
            elif self.adversarial_framing == "competitive":
                goal = self.templates.competitive_adversarial_goal + self.templates.prd_adversarial_election_strategy
            elif self.adversarial_framing == "devils_advocate":
                goal = self.templates.devils_advocate_adversarial_goal + self.templates.prd_adversarial_election_strategy
            else:
                goal = self.templates.antifragility_adversarial_goal + self.templates.prd_adversarial_election_strategy
        elif is_adversarial:
            # Regular adversarial goals for non-PRD-election cases
            if self.adversarial_framing == "antifragility":
                goal = self.templates.antifragility_adversarial_goal
            elif self.adversarial_framing == "competitive":
                goal = self.templates.competitive_adversarial_goal
            elif self.adversarial_framing == "devils_advocate":
                goal = self.templates.devils_advocate_adversarial_goal
            else:
                goal = self.templates.antifragility_adversarial_goal
        else:
            goal = self.templates.aligned_goal

        participation_status_str = "ACTIVELY PARTICIPATING this round."
        specific_instructions = ""
        current_options_str = ""
        max_tokens = self.base_response_tokens
        if is_delegate_role_for_portfolio_vote: 
            max_tokens += self.delegate_response_bonus

        # === ROBUST PORTFOLIO COUNT CALCULATION ===
        num_p_options_for_example = 0
        if portfolio_options_str:
            try:
                # More robust parsing of portfolio options
                portfolio_lines = []
                for line in portfolio_options_str.strip().split('\n'):
                    cleaned_line = line.strip()
                    if cleaned_line and ':' in cleaned_line:  # Ensure it's a valid portfolio line
                        portfolio_lines.append(cleaned_line)
                num_p_options_for_example = len(portfolio_lines)
                
                # Validation: ensure we have at least 1 portfolio for portfolio-related prompts
                if prompt_type in ["portfolio_vote", "pld_delegate_declaration", "pld_final_power_holder_vote"]:
                    if num_p_options_for_example == 0:
                        print(f"[WARNING] Portfolio prompt type {prompt_type} but no valid portfolios found in options string: '{portfolio_options_str}'")
                        num_p_options_for_example = 4  # Safe fallback
            except Exception as e:
                print(f"[ERROR] Failed to parse portfolio_options_str: {e}. Using fallback count of 4.")
                num_p_options_for_example = 4  # Safe fallback
        
        # === GENERATE CLEAR, UNAMBIGUOUS EXAMPLES ===
        # NEVER use empty brackets - always provide full-length examples
        if num_p_options_for_example > 0:
            # Generate a positive example (approve some portfolios)
            example_portfolio_votes_list = ['0'] * num_p_options_for_example
            example_portfolio_votes_list[0] = '1'  # Always approve first option for consistency
            if num_p_options_for_example > 2: 
                example_portfolio_votes_list[2] = '1'  # Approve third option for variation
            example_portfolio_votes_str = '[' + ','.join(example_portfolio_votes_list) + ']'
            
            # Generate explicit all-zeros example (NEVER empty brackets)
            all_zeros_list = ['0'] * num_p_options_for_example
            all_zeros_example_str = '[' + ','.join(all_zeros_list) + ']'
            
            # Generate candidate examples
            num_cands_for_example = prd_num_candidates or 0
            if num_cands_for_example > 0:
                example_cand_votes_list = ['0'] * num_cands_for_example
                example_cand_votes_list[0] = '1'  # Approve first candidate
                if num_cands_for_example > 2: 
                    example_cand_votes_list[2] = '1'  # Approve third candidate
                example_cand_votes_str = '[' + ','.join(example_cand_votes_list) + ']'
            else:
                example_cand_votes_str = "[]"  # Only for candidates when there are none
        else:
            # Fallback if no portfolios (should rarely happen)
            example_portfolio_votes_str = "[0,0,0,0]"  # Safe fallback
            all_zeros_example_str = "[0,0,0,0]"  # Safe fallback
            example_cand_votes_str = "[]"

        # === PARTICIPATION HANDLING ===
        if not is_participating_this_round:
            participation_status_str = "NOT ACTIVELY PARTICIPATING this round."
            specific_instructions = self.templates.non_participation_general
            current_options_str = portfolio_options_str or "No options as not participating."
        
        # === PROMPT TYPE SPECIFIC LOGIC ===
        elif prompt_type == "election_vote" and mechanism == "PRD":
            role_str = "Voter (in Election for Representatives)"
            current_options_str = prd_candidate_options_str or "No candidates listed for election."
            max_tokens += self.election_prompt_bonus_tokens

            candidate_history_info = prd_candidate_performance_history_str or "Candidate performance history not available."
            
            # CLEAR, EXPLICIT ELECTION INSTRUCTIONS
            specific_instructions = (
                f"YOUR DECISION - ELECTION VOTING FOR REPRESENTATIVES:\n"
                f"An election is being held to choose representatives who will make portfolio decisions.\n"
                f"Below is a list of agents who are candidates in this election.\n\n"
                f"ANALYZE the performance history carefully to understand each candidate's decision-making patterns.\n\n"
                f"PERFORMANCE HISTORY ANALYSIS:\n{candidate_history_info}\n\n"
                f"VOTING INSTRUCTIONS:\n"
                f"You MUST provide your approvals for these candidates as a list of 0s and 1s.\n"
                f"You must provide a '0' (do not approve) or '1' (approve) for EACH of the {num_cands_for_example} candidates listed below.\n"
                f"Consider each candidate's past portfolio choices and decision patterns when making your approval decisions.\n\n"
                f"CRITICAL: Respond with exactly: 'CandidateVotes: [a,b,c,...]' where a,b,c are either 0 or 1.\n"
                f"Example for {num_cands_for_example} candidates: 'CandidateVotes: {example_cand_votes_str}'\n"
                f"DO NOT use empty brackets []. You must provide exactly {num_cands_for_example} votes.\n\n"
                f"Candidates for Election:\n{current_options_str}"
            )

        elif prompt_type == "pld_delegate_declaration" and mechanism == "PLD":
            role_str = "Delegate (Declaring Portfolio Choice)"
            current_options_str = portfolio_options_str or "No portfolio options available."
            
            # CRYSTAL CLEAR DELEGATE DECLARATION INSTRUCTIONS
            specific_instructions = (
                f"YOUR TASK - DELEGATE PORTFOLIO DECLARATION:\n"
                f"You are a designated DELEGATE. You MUST DECLARE your portfolio approval.\n"
                f"You have {num_p_options_for_example} portfolios to evaluate (listed above).\n\n"
                f"CRITICAL: You MUST provide exactly {num_p_options_for_example} votes (one for each portfolio).\n"
                f"Use '0' for disapproval and '1' for approval.\n"
                f"Respond with exactly: 'DeclaredVotes: [a,b,c,...]' where each letter is 0 or 1.\n\n"
                f"EXAMPLES:\n"
                f"- To approve some portfolios: 'DeclaredVotes: {example_portfolio_votes_str}'\n"
                f"- To approve NO portfolios: 'DeclaredVotes: {all_zeros_example_str}'\n"
                f"- You MUST provide exactly {num_p_options_for_example} values. DO NOT use empty brackets [].\n\n"
                f"DO NOT include any other text. JUST the 'DeclaredVotes: [...]' line."
            )

        elif prompt_type == "pld_voter_delegation_choice" and mechanism == "PLD":
            role_str = "Voter (PLD Delegation Phase)"
            current_options_str = portfolio_options_str or "No portfolio options available."
            delegate_targets_info = pld_delegate_targets_str or "No original delegates available."
            declared_choices_info = pld_declared_delegate_choices_str or "Delegate choices not available."
            delegate_performance_history_info = pld_performance_history_str or "Delegate performance history not available."
            
            specific_instructions = (
                f"YOUR TASK - VOTER DELEGATION DECISION:\n"
                f"You are a VOTER. You have two choices:\n\n"
                f"AVAILABLE DELEGATES:\n{delegate_targets_info}\n\n"
                f"DELEGATE PERFORMANCE HISTORY:\n{delegate_performance_history_info}\n\n"
                f"DELEGATES' DECLARED CHOICES:\n{declared_choices_info}\n\n"
                f"CHOOSE ONE ACTION:\n"
                f"1. To DELEGATE your vote to a delegate:\n"
                f"   Action: DELEGATE\n"
                f"   AgentID: [delegate_id_number]\n\n"
                f"2. To vote directly yourself:\n"
                f"   Action: VOTE_DIRECTLY\n\n"
                f"Choose exactly one action. Do NOT provide portfolio votes at this stage."
            )

        elif prompt_type == "pld_final_power_holder_vote" and mechanism == "PLD":
            role_str = "Agent (PLD Final Portfolio Vote)"
            current_options_str = portfolio_options_str or "No portfolio options available."
            
            # CLEAR FINAL VOTE INSTRUCTIONS
            specific_instructions = (
                f"YOUR TASK - FINAL PORTFOLIO VOTE (PLD):\n"
                f"You hold voting power after the delegation phase.\n"
                f"Voting Power: {pld_agent_voting_power_str or 'N/A'}\n"
                f"{pld_prior_declaration_reminder_str or ''}\n\n"
                f"You have {num_p_options_for_example} portfolios to evaluate (listed above).\n\n"
                f"CRITICAL: You MUST provide exactly {num_p_options_for_example} votes.\n"
                f"Respond with exactly: 'Votes: [a,b,c,...]' where each value is 0 or 1.\n\n"
                f"EXAMPLES:\n"
                f"- To approve some portfolios: 'Votes: {example_portfolio_votes_str}'\n"
                f"- To approve NO portfolios: 'Votes: {all_zeros_example_str}'\n"
                f"- You MUST provide exactly {num_p_options_for_example} values. DO NOT use empty brackets [].\n\n"
                f"DO NOT include any other text. JUST the 'Votes: [...]' line."
            )

        elif prompt_type == "portfolio_vote":  # PDD/PRD portfolio voting
            current_options_str = portfolio_options_str or "No portfolio options available."
            
            if mechanism in ["PDD", "PRD"]:
                specific_instructions = (
                    f"YOUR DECISION - PORTFOLIO APPROVALS:\n"
                    f"You have {num_p_options_for_example} portfolios to evaluate (listed above).\n\n"
                    f"CRITICAL: You MUST provide exactly {num_p_options_for_example} votes.\n"
                    f"Respond with exactly: 'Votes: [a,b,c,...]' where each value is 0 or 1.\n\n"
                    f"EXAMPLES:\n"
                    f"- To approve some portfolios: 'Votes: {example_portfolio_votes_str}'\n"
                    f"- To approve NO portfolios: 'Votes: {all_zeros_example_str}'\n"
                    f"- You MUST provide exactly {num_p_options_for_example} values. DO NOT use empty brackets [].\n\n"
                    f"DO NOT include any other text. JUST the 'Votes: [...]' line."
                )
            else:
                specific_instructions = "Error: Standard portfolio_vote used for PLD. Check PLD phase logic."
                current_options_str = "Configuration error in PLD prompt generation."
        else:
            specific_instructions = "Error: Invalid prompt type or mechanism combination."
            current_options_str = "Configuration error in prompt generation."

        # === ASSEMBLE FINAL PROMPT ===
        decision_framework_output = ""
        if include_decision_framework and is_participating_this_round:
            decision_framework_output = self.templates.decision_framework_template + "\n"

        prompt = self.templates.base_template.format(
            agent_id=agent_id, 
            round_num=round_num, 
            role=role_str, 
            goal=goal,
            participation_status=participation_status_str,
            decision_framework=decision_framework_output,
            portfolio_options=current_options_str
        )
        
        prompt += f"\n{specific_instructions}"

        # Add optimality analysis for portfolio decisions
        if (is_participating_this_round and 
            ("portfolio" in prompt_type or "declaration" in prompt_type) and 
            optimality_analysis_str):
            prompt += f"\n\nOPTIMALITY ANALYSIS (perfect signals):\n{optimality_analysis_str}\n"
        
        # Final reminders
        if is_participating_this_round:
            if "portfolio" in prompt_type or "declaration" in prompt_type:
                prompt += f"\nREMINDER: Provide EXACTLY {num_p_options_for_example} votes. NO empty brackets []."
            elif prompt_type == "election_vote":
                prompt += f"\nREMINDER: Provide EXACTLY {num_cands_for_example} candidate votes."

        return {"prompt": prompt, "max_tokens": max_tokens}


    
    
# --- Master Configuration for Stable Democracy ---
@dataclass(frozen=True)
class StablePortfolioDemocracyConfig:
    mechanism: Literal["PDD", "PRD", "PLD"]
    num_agents: int
    num_delegates: int
    num_rounds: int
    seed: int

    crops: List[CropConfig]
    portfolios: List[PortfolioStrategyConfig]

    # This will always be an instance of OriginalCognitiveResourceConfig.
    # Its values are used by the prediction market if eliminate_prediction_noise=False.
    # They are NOT used in prompts generated by StablePromptConfig.
    cognitive_resource_settings: OriginalCognitiveResourceConfig

    resources: ResourceConfig = field(default_factory=ResourceConfig)
    agent_settings: AgentSettingsConfig = field(default_factory=AgentSettingsConfig)
    participation_settings: ParticipationConfig = field(default_factory=ParticipationConfig)
    locked_value_settings: LockedValueConfig = field(default_factory=LockedValueConfig)
    market_settings: MarketConfig = field(default_factory=MarketConfig)
    prompt_settings: StablePromptConfig = field(default_factory=StablePromptConfig)

    prd_election_term_length: int = 4
    prd_num_representatives_to_elect: Optional[int] = None
    
    include_optimality_analysis: bool = False
    use_redteam_prompts: bool = False # For Stable, this primarily affects the goal template

    def __post_init__(self):
        if self.num_delegates > self.num_agents:
            raise ValueError("Number of delegates cannot exceed number of agents.")
        if self.mechanism == "PRD" and self.prd_num_representatives_to_elect is None:
            object.__setattr__(self, 'prd_num_representatives_to_elect', self.num_delegates)
        
        if self.locked_value_settings.use_locked_values and self.crops:
            if self.crops and len(self.crops) != len(self.locked_value_settings.locked_crop_yields):
                print(
                   f"Warning: Number of crops ({len(self.crops)}) does not match length of "
                   f"locked_crop_yields ({len(self.locked_value_settings.locked_crop_yields)}). "
                   f"Ensure `get_true_expected_yields_for_round_stable` handles this (e.g. cycling/truncating)."
                )

# --- Factory Functions for Stable Democracy ---
def create_stable_democracy_config(
    mechanism: Literal["PDD", "PRD", "PLD"],
    adversarial_proportion_total: float,
    adversarial_framing: str = "antifragility",
    seed: int = 42,
    num_agents: int = 10,
    num_delegates: int = 4,
    num_rounds: int = 30,
    delegate_participation_rate: float = 0.95,
    voter_participation_rate: float = 0.65,
    temporal_correlation_strength: float = 0.7,
    use_locked_values: bool = True,
    locked_crop_yields: Optional[List[float]] = None,
    eliminate_prediction_noise: bool = True,
    num_crops: int = 3,
    num_portfolios: int = 4,
    # These values are for the OriginalCognitiveResourceConfig object.
    # They affect prediction noise ONLY IF eliminate_prediction_noise is False.
    # They are NOT included in the LLM prompts generated by StablePromptConfig.
    delegate_cognitive_resources_if_noise_is_not_eliminated: int = 100,
    voter_cognitive_resources_if_noise_is_not_eliminated: int = 100,
    prediction_market_sigma_if_noise_is_not_eliminated: float = 0.0, # Default to 0 if noise IS eliminated
    include_optimality_analysis_in_prompt: bool = False,
    use_redteam_prompts_in_stable: bool = False,
) -> StablePortfolioDemocracyConfig:

    example_dynamic_yields = [1.0] * num_rounds # Placeholder if not using locked values
    crops = [
        CropConfig(name=f"Crop{chr(65+i)}", true_expected_yields_per_round=example_dynamic_yields)
        for i in range(num_crops)
    ]

    portfolios = []
    if num_crops > 0:
        portfolios.append(PortfolioStrategyConfig(name="P_Equal", weights=[1.0/num_crops]*num_crops, description=f"Equal allocation across {num_crops} crops"))
        for i in range(min(num_crops, num_portfolios - 1)): # Create focused portfolios
            weights = [0.1 / (num_crops -1) if num_crops > 1 else 0.0] * num_crops # Small weight to others
            weights[i] = 1.0 - sum(weights[:i] + weights[i+1:]) # Main weight to focused crop
            weights[i] = max(0.0, weights[i]) # Ensure non-negative
            current_sum = sum(weights) # Normalize
            if current_sum > 0: weights = [w / current_sum for w in weights]
            else: weights = [1.0/num_crops]*num_crops # Fallback if something went wrong

            portfolios.append(PortfolioStrategyConfig(name=f"P_Focus_Crop{crops[i].name}", weights=weights, description=f"{crops[i].name} focused allocation"))
        
        # Fill remaining portfolios with more distinct tactical/random allocations if needed
        # For simplicity, we can just make them equal-like or slightly varied for now
        additional_needed = num_portfolios - len(portfolios)
        if num_crops > 0:
            for i in range(additional_needed):
                # Simple variation: slightly shift weights from equal
                base_weights = [1.0/num_crops] * num_crops
                if num_crops > 1:
                    idx_to_increase = i % num_crops
                    idx_to_decrease = (i+1) % num_crops
                    if idx_to_increase != idx_to_decrease :
                        base_weights[idx_to_increase] += 0.1
                        base_weights[idx_to_decrease] -= 0.1
                        base_weights = [max(0.0, w) for w in base_weights] # clip at 0
                        current_sum = sum(base_weights) # re-normalize
                        if current_sum > 0 : base_weights = [w / current_sum for w in base_weights]
                        else: base_weights = [1.0/num_crops]*num_crops

                portfolios.append(PortfolioStrategyConfig(name=f"P_Tactical{i+1}", weights=base_weights, description=f"Tactical allocation {i+1}"))

    elif num_portfolios > 0 : # No crops, but portfolios requested
         portfolios.append(PortfolioStrategyConfig(name="P_NoOp", weights=[], description="No operations portfolio"))

    participation_conf = ParticipationConfig(
        delegate_participation_rate=delegate_participation_rate,
        voter_participation_rate=voter_participation_rate,
        temporal_correlation_strength=temporal_correlation_strength
    )
    
    actual_locked_yields = locked_crop_yields
    if actual_locked_yields is None:
        if num_crops > 0:
            # Default generation: e.g. for 3 crops [0.8, 1.0, 1.2] to ensure some variance
            actual_locked_yields = [round(0.8 + (i * (0.4 / max(1,num_crops-1))),2) if num_crops > 1 else 1.0 for i in range(num_crops)]
        else:
            actual_locked_yields = []

    locked_value_conf = LockedValueConfig(
        use_locked_values=use_locked_values,
        locked_crop_yields=actual_locked_yields,
        eliminate_prediction_noise=eliminate_prediction_noise
    )

    cog_config_for_stable = OriginalCognitiveResourceConfig(
        cognitive_resources_delegate=delegate_cognitive_resources_if_noise_is_not_eliminated,
        cognitive_resources_voter=voter_cognitive_resources_if_noise_is_not_eliminated
    )
    
    # If noise is truly eliminated, sigma should be 0.0. Otherwise, use the passed value.
    actual_sigma = 0.0 if eliminate_prediction_noise else prediction_market_sigma_if_noise_is_not_eliminated
    market_conf = MarketConfig(prediction_noise_sigma=actual_sigma)

    # Determine which goal templates to use for StablePromptConfig
    # If use_redteam_prompts_in_stable is True, adversarial_goal from AgentPromptTemplatesStable will be used.
    # Otherwise, a simpler "minimize" or "maximize" could be used, but the current structure is fine.
    # The StablePromptConfig's AgentPromptTemplatesStable already has distinct aligned/adversarial goals.

    return StablePortfolioDemocracyConfig(
        mechanism=mechanism,
        num_agents=num_agents,
        num_delegates=num_delegates,
        num_rounds=num_rounds,
        seed=seed,
        crops=crops,
        portfolios=portfolios[:num_portfolios],
        agent_settings=AgentSettingsConfig(adversarial_proportion_total=adversarial_proportion_total,
                                           # Use default for adversarial_proportion_delegates
                                           adversarial_introduction_type="immediate"),
        resources=ResourceConfig(),
        participation_settings=participation_conf,
        locked_value_settings=locked_value_conf,
        cognitive_resource_settings=cog_config_for_stable,
        market_settings=market_conf,
        prompt_settings=StablePromptConfig(
            templates=AgentPromptTemplatesStable(),
            adversarial_framing=adversarial_framing,  # Pass the framing here
        ),
        include_optimality_analysis=include_optimality_analysis_in_prompt,
        use_redteam_prompts=use_redteam_prompts_in_stable
    )

# Example: {"P_Focus_CropA": [0.7, 1.0, 1.5], "P_Focus_CropB": [0.8, 1.0, 1.3]}
# If not defined or empty, portfolios might be generated dynamically or use hardcoded defaults.
PORTFOLIO_CROP_YIELD_MULTIPLIERS = {}

# Portfolio Voting Tie-Breaking Strategy
# Defines how to resolve ties when multiple portfolios receive the same highest number of approval votes
# from elected representatives.
# Valid options: "lowest_index", "random"
#   "lowest_index": The portfolio with the smallest index among the tied winners is chosen.
#   "random": A portfolio is chosen randomly from the set of tied winners.
PORTFOLIO_VOTE_TIE_BREAKING_STRATEGY = "random" # Defaulting to random for more diverse outcomes

if __name__ == "__main__":
    print("--- Testing Stable Democracy Configuration (No Cognitive Resources in Prompt) ---")
    
    # Scenario 1: Pure Stable (No noise, locked values, participation driven)
    # Prompts should NOT mention cognitive resources.
    stable_config_pdd_pure = create_stable_democracy_config(
        mechanism="PDD", 
        adversarial_proportion_total=0.0,
        eliminate_prediction_noise=True, # This is key
        use_locked_values=True,
        num_crops=3,
        locked_crop_yields=[1.1, 1.0, 0.9]
    )
    print("\n1. Pure Stable Config (PDD, no cognitive resource in prompt):")
    print(f"  Locked Values: {stable_config_pdd_pure.locked_value_settings}")
    print(f"  Market Settings (sigma should be 0): {stable_config_pdd_pure.market_settings}")
    print(f"  CognitiveResourceConfig object (exists but values not used in prompt): {stable_config_pdd_pure.cognitive_resource_settings}")
    
    # Generate a sample prompt to verify no cognitive resources are mentioned
    # For this test, assume agent 0 is a voter, participating, not adversarial.
    # And no optimality analysis for simplicity.
    sample_prompt_dict = stable_config_pdd_pure.prompt_settings.generate_prompt(
        agent_id=0, round_num=1, is_delegate_role_for_portfolio_vote=False, is_adversarial=False,
        mechanism="PDD", portfolio_options_str="0: P1 (1.1x)\n1: P2 (1.0x)\n2: P3 (0.9x)\n3: P4 (0.8x)",
        is_participating_this_round=True, prompt_type="portfolio_vote",
        include_decision_framework=True # Test with decision framework
    )
    print("\nSample PDD Prompt (should NOT mention numeric cognitive resources):")
    print(sample_prompt_dict["prompt"])
    print(f"Example should show 4 votes: {sample_prompt_dict['prompt'].count('[1,0,1,0]') > 0}")

    # Scenario 2: Test PLD with dynamic examples
    stable_config_pld = create_stable_democracy_config(
        mechanism="PLD", 
        adversarial_proportion_total=0.1, 
        seed=101,
        num_crops=2,
        num_portfolios=3,
        locked_crop_yields=[1.2, 0.8],
        include_optimality_analysis_in_prompt=True
    )
    
    # Test delegate declaration with 3 portfolios
    sample_pld_delegate_prompt = stable_config_pld.prompt_settings.generate_prompt(
        agent_id=2, round_num=1, is_delegate_role_for_portfolio_vote=True, is_adversarial=False,
        mechanism="PLD", portfolio_options_str="0: P_Equal (1.0x)\n1: P_Focus_A (1.2x)\n2: P_Focus_B (0.8x)",
        is_participating_this_round=True, prompt_type="pld_delegate_declaration",
        optimality_analysis_str="P_Focus_A is optimal with 1.2x return."
    )
    print("\n2. Sample PLD Delegate Declaration (should show 3 votes in example AND all-zeros example):")
    print(sample_pld_delegate_prompt["prompt"])
    print(f"Example should show 3 votes: {'[1,0,1]' in sample_pld_delegate_prompt['prompt']}")
    print(f"Should show all-zeros example: {'[0,0,0]' in sample_pld_delegate_prompt['prompt']}")

    # Test PRD election with dynamic candidate examples
    stable_config_prd = create_stable_democracy_config(
        mechanism="PRD", 
        adversarial_proportion_total=0.0,
        num_agents=10,
        num_delegates=4
    )
    
    sample_prd_election_prompt = stable_config_prd.prompt_settings.generate_prompt(
        agent_id=5, round_num=1, is_delegate_role_for_portfolio_vote=False, is_adversarial=False,
        mechanism="PRD", is_participating_this_round=True, prompt_type="election_vote",
        prd_candidate_options_str="0: Agent1\n1: Agent2\n2: Agent3\n3: Agent4",
        prd_num_candidates=4,
        prd_candidate_performance_history_str="Agent1: 0.9 | Agent2: 0.8 | Agent3: 0.7 | Agent4: 0.6"
    )
    print("\n3. Sample PRD Election (should show 4 candidate votes in example):")
    print(sample_prd_election_prompt["prompt"])
    print(f"Example should show 4 candidate votes: {'[1,0,1,0]' in sample_prd_election_prompt['prompt']}")

    print("\n--- All tests show dynamic examples matching actual scenario length ---")
    print("--- Explicit all-zeros examples added to prevent empty list responses ---")
    print("--- Stable Democracy Configuration Test Complete ---")