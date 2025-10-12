"""
Environment factory for gridworld environments.
Creates and manages the simulation environment.
"""

from typing import Dict, Any, List, Tuple
from .state import State, Object
from .sim import Simulator


class Environment:
    """Gridworld environment factory and manager."""
    
    def __init__(self, width: int = 5, height: int = 5):
        """Initialize environment with given dimensions."""
        self.width = width
        self.height = height
        self.simulator = Simulator()
    
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
        # Check if robot is at microwave position with microwave-safe object in gripper
        for obj_name, obj in state.objects.items():
            if (obj.properties.get("is_microwave", False) and 
                obj.position == robot_pos and
                state.gripper is not None and
                state.gripper != obj_name):  # Object in gripper is different from microwave
                # Check if object in gripper is microwave-safe
                gripper_obj = state.objects.get(state.gripper)
                if gripper_obj and gripper_obj.properties.get("is_microwave_safe", False):
                    actions.append({
                        "type": "heat",
                        "object": state.gripper
                    })
        
        return actions
    
    def apply_action(self, state: State, action: Dict[str, Any]) -> State:
        """Apply action to state and return new state."""
        return self.simulator.apply_action(state, action)
    
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
