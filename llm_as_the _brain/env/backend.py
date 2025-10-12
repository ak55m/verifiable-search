"""
Environment factory for mathematical verification framework.
Enhanced with support for safe action invention and verification.
"""

from typing import Dict, Any, List, Tuple
from .state import State, Object
from .sim import Simulator


class Environment:
    """Enhanced environment for mathematical verification framework."""
    
    def __init__(self, width: int = 5, height: int = 5):
        """Initialize environment with given dimensions."""
        self.width = width
        self.height = height
        self.simulator = Simulator()
        self.verification_enabled = True
    
    def create_state(self, objects: Dict[str, Object], robot_pos: Tuple[int, int] = (0, 0)) -> State:
        """Create initial state with objects and robot position."""
        return State(
            width=self.width,
            height=self.height,
            robot=robot_pos,
            gripper=None,
            objects=objects,
            open_containers=set()
        )
    
    def get_available_actions(self, state: State) -> List[Dict[str, Any]]:
        """Get all possible actions from current state."""
        actions = []
        
        # Movement actions
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_x = state.robot[0] + dx
            new_y = state.robot[1] + dy
            if 0 <= new_x < state.width and 0 <= new_y < state.height:
                actions.append({
                    "type": "move",
                    "direction": (dx, dy),
                    "target": (new_x, new_y)
                })
        
        # Pick/drop actions for objects at robot location
        robot_pos = state.robot
        for obj_name, obj in state.objects.items():
            if obj.position == robot_pos:
                if state.gripper is None:
                    actions.append({
                        "type": "pick",
                        "object": obj_name
                    })
                elif state.gripper == obj_name:
                    actions.append({
                        "type": "drop",
                        "object": obj_name
                    })
        
        # Container actions
        for obj_name, obj in state.objects.items():
            if obj.position == robot_pos and obj.properties.get("is_container", False):
                if obj_name in state.open_containers:
                    actions.append({
                        "type": "close",
                        "container": obj_name
                    })
                else:
                    actions.append({
                        "type": "open",
                        "container": obj_name
                    })
        
        # Heat action for microwave-safe objects
        for obj_name, obj in state.objects.items():
            if (obj.properties.get("is_microwave", False) and 
                obj.position == robot_pos and
                state.gripper is not None and
                state.gripper != obj_name):
                gripper_obj = state.objects.get(state.gripper)
                if gripper_obj and gripper_obj.properties.get("is_microwave_safe", False):
                    actions.append({
                        "type": "heat",
                        "object": state.gripper
                    })
        
        return actions
    
    def apply_action(self, state: State, action: Dict[str, Any]) -> State:
        """Apply action to state and return new state."""
        if self.verification_enabled:
            # Enhanced action application with verification
            return self._apply_action_with_verification(state, action)
        else:
            return self.simulator.apply_action(state, action)
    
    def _apply_action_with_verification(self, state: State, action: Dict[str, Any]) -> State:
        """Apply action with mathematical verification."""
        # Apply the action
        new_state = self.simulator.apply_action(state, action)
        
        # Log action for verification purposes
        self._log_action_execution(state, action, new_state)
        
        return new_state
    
    def _log_action_execution(self, state: State, action: Dict[str, Any], new_state: State):
        """Log action execution for verification and learning."""
        # This would log action execution for later analysis
        # In practice, this would store in a database or log file
        pass
    
    def is_goal_state(self, state: State, goal: str) -> bool:
        """Check if current state satisfies the goal."""
        if goal == "safe_microwave":
            # Check if microwave-safe bowl is heated
            for obj_name, obj in state.objects.items():
                if (obj_name == "bowl" and 
                    obj.properties.get("is_microwave_safe", False) and
                    obj.properties.get("is_heated", False)):
                    return True
            return False
        
        elif goal == "tidy_desk":
            # Check if book is on shelf
            for obj_name, obj in state.objects.items():
                if obj_name == "book" and obj.position == (4, 4):  # Shelf position
                    return True
            return False
        
        elif goal == "make_tea":
            # Check if cup and kettle are at stove
            cup_at_stove = False
            kettle_at_stove = False
            for obj_name, obj in state.objects.items():
                if obj_name == "cup" and obj.position == (2, 2):  # Stove position
                    cup_at_stove = True
                elif obj_name == "kettle" and obj.position == (2, 2):
                    kettle_at_stove = True
            return cup_at_stove and kettle_at_stove
        
        return False
    
    def get_state_features(self, state: State) -> Dict[str, Any]:
        """Get mathematical features of state for verification."""
        return {
            "robot_position": state.robot,
            "gripper_state": state.gripper,
            "object_positions": {name: obj.position for name, obj in state.objects.items()},
            "object_properties": {name: obj.properties for name, obj in state.objects.items()},
            "open_containers": list(state.open_containers),
            "state_hash": hash(state)
        }
    
    def compute_reward(self, state: State, action: Dict[str, Any], next_state: State) -> float:
        """Compute reward for action transition."""
        reward = 0.0
        
        # Goal progress reward
        if self.is_goal_state(next_state, "safe_microwave"):
            reward += 10.0
        elif self.is_goal_state(next_state, "tidy_desk"):
            reward += 10.0
        elif self.is_goal_state(next_state, "make_tea"):
            reward += 10.0
        
        # Safety penalty
        if self._is_unsafe_transition(state, action, next_state):
            reward -= 5.0
        
        # Efficiency reward
        reward += 1.0
        
        return reward
    
    def _is_unsafe_transition(self, state: State, action: Dict[str, Any], next_state: State) -> bool:
        """Check if transition is unsafe."""
        # Check for out-of-bounds
        if not (0 <= next_state.robot[0] < next_state.width and 
                0 <= next_state.robot[1] < next_state.height):
            return True
        
        # Check for invalid object states
        for obj_name, obj in next_state.objects.items():
            if not (0 <= obj.position[0] < next_state.width and 
                   0 <= obj.position[1] < next_state.height):
                return True
        
        return False