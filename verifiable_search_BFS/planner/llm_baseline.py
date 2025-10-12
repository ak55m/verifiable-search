"""
LLM-based planner baseline.
Currently contains stub implementations that need real LLM integration.
"""

from typing import List, Dict, Any, Tuple
import time
import random
from env.state import State
from env.backend import Environment
from rules.verify import RuleVerifier


class LLMPlanner:
    """LLM-based planner (currently stub implementation)."""
    
    def __init__(self, environment: Environment, model_name: str = "gpt-3.5-turbo"):
        """Initialize LLM planner."""
        self.env = environment
        self.model_name = model_name
        self.verifier = RuleVerifier()
        self.api_key = None  # Will be set when real LLM integration is added
    
    def plan(self, initial_state: State, goal: str, max_steps: int = 20) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Generate a plan using LLM (currently stub).
        
        Returns:
            plan: List of actions to reach goal
            metadata: Planning statistics and info
        """
        start_time = time.time()
        
        # TODO: Replace with real LLM integration
        plan = self._generate_stub_plan(initial_state, goal, max_steps)
        
        # Verify plan against rules
        violations = 0
        current_state = initial_state.copy()
        
        for i, action in enumerate(plan):
            if self.verifier.violates_rules(current_state, action):
                violations += 1
                # In real implementation, we would stop execution here
                # For now, we continue to count violations
            else:
                try:
                    current_state = self.env.apply_action(current_state, action)
                except ValueError:
                    violations += 1
        
        runtime = time.time() - start_time
        
        return plan, {
            "success": violations == 0 and self.env.is_goal_state(current_state, goal),
            "violations": violations,
            "steps": len(plan),
            "runtime_s": runtime,
            "model": self.model_name
        }
    
    def _generate_stub_plan(self, state: State, goal: str, max_steps: int) -> List[Dict[str, Any]]:
        """Generate a stub plan (placeholder for real LLM)."""
        # This is a very basic stub that often fails
        # In real implementation, this would call an LLM API
        
        if goal == "safe_microwave":
            return self._stub_safe_microwave_plan(state)
        elif goal == "tidy_desk":
            return self._stub_tidy_desk_plan(state)
        elif goal == "make_tea":
            return self._stub_make_tea_plan(state)
        else:
            return []
    
    def _stub_safe_microwave_plan(self, state: State) -> List[Dict[str, Any]]:
        """Stub plan for safe microwave task."""
        # This is intentionally simplistic and often wrong
        plan = []
        
        # Try to pick up bowl (might fail if not at robot position)
        plan.append({"type": "pick", "object": "bowl"})
        
        # Move towards microwave (hardcoded position)
        plan.append({"type": "move", "direction": (1, 0), "target": (1, 0)})
        plan.append({"type": "move", "direction": (1, 0), "target": (2, 0)})
        plan.append({"type": "move", "direction": (0, 1), "target": (2, 1)})
        plan.append({"type": "move", "direction": (0, 1), "target": (2, 2)})
        
        # Try to heat (might fail if not microwave-safe)
        plan.append({"type": "heat", "object": "bowl"})
        
        return plan
    
    def _stub_tidy_desk_plan(self, state: State) -> List[Dict[str, Any]]:
        """Stub plan for tidy desk task."""
        plan = []
        
        # Try to pick up book
        plan.append({"type": "pick", "object": "book"})
        
        # Move towards shelf
        plan.append({"type": "move", "direction": (1, 0), "target": (1, 0)})
        plan.append({"type": "move", "direction": (1, 0), "target": (2, 0)})
        plan.append({"type": "move", "direction": (1, 0), "target": (3, 0)})
        plan.append({"type": "move", "direction": (1, 0), "target": (4, 0)})
        plan.append({"type": "move", "direction": (0, 1), "target": (4, 1)})
        plan.append({"type": "move", "direction": (0, 1), "target": (4, 2)})
        plan.append({"type": "move", "direction": (0, 1), "target": (4, 3)})
        plan.append({"type": "move", "direction": (0, 1), "target": (4, 4)})
        
        # Drop book
        plan.append({"type": "drop", "object": "book"})
        
        return plan
    
    def _stub_make_tea_plan(self, state: State) -> List[Dict[str, Any]]:
        """Stub plan for make tea task."""
        plan = []
        
        # Try to pick up cup
        plan.append({"type": "pick", "object": "cup"})
        
        # Move to stove
        plan.append({"type": "move", "direction": (1, 0), "target": (1, 0)})
        plan.append({"type": "move", "direction": (1, 0), "target": (2, 0)})
        plan.append({"type": "move", "direction": (0, 1), "target": (2, 1)})
        plan.append({"type": "move", "direction": (0, 1), "target": (2, 2)})
        
        # Drop cup
        plan.append({"type": "drop", "object": "cup"})
        
        # Pick up kettle
        plan.append({"type": "pick", "object": "kettle"})
        
        # Kettle is already at stove in this stub
        plan.append({"type": "drop", "object": "kettle"})
        
        return plan


class RandomPlanner:
    """Random action planner for comparison."""
    
    def __init__(self, environment: Environment):
        """Initialize random planner."""
        self.env = environment
        self.verifier = RuleVerifier()
    
    def plan(self, initial_state: State, goal: str, max_steps: int = 20) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Generate a random plan."""
        start_time = time.time()
        
        plan = []
        current_state = initial_state.copy()
        violations = 0
        
        for _ in range(max_steps):
            # Get all possible actions
            actions = self.env.get_available_actions(current_state)
            
            if not actions:
                break
            
            # Randomly select an action
            action = random.choice(actions)
            plan.append(action)
            
            # Check for violations
            if self.verifier.violates_rules(current_state, action):
                violations += 1
            else:
                try:
                    current_state = self.env.apply_action(current_state, action)
                except ValueError:
                    violations += 1
            
            # Check if goal reached
            if self.env.is_goal_state(current_state, goal):
                break
        
        runtime = time.time() - start_time
        
        return plan, {
            "success": violations == 0 and self.env.is_goal_state(current_state, goal),
            "violations": violations,
            "steps": len(plan),
            "runtime_s": runtime,
            "model": "random"
        }


# TODO: Add real LLM integration
class OpenAIPlanner(LLMPlanner):
    """Real OpenAI API integration (placeholder)."""
    
    def __init__(self, environment: Environment, api_key: str, model_name: str = "gpt-3.5-turbo"):
        """Initialize with OpenAI API key."""
        super().__init__(environment, model_name)
        self.api_key = api_key
    
    def plan(self, initial_state: State, goal: str, max_steps: int = 20) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Generate plan using OpenAI API."""
        # TODO: Implement real OpenAI API calls
        # This would involve:
        # 1. Converting state to natural language
        # 2. Sending prompt to OpenAI API
        # 3. Parsing response into action sequence
        # 4. Validating actions against rules
        
        # For now, fall back to stub
        return super().plan(initial_state, goal, max_steps)
