"""
Simulator for applying actions to states.
Handles the execution logic for all action types.
"""

from typing import Dict, Any, Tuple
from .state import State, Object


class Simulator:
    """Simulates action execution in the environment."""
    
    def apply_action(self, state: State, action: Dict[str, Any]) -> State:
        """Apply action to state and return new state."""
        new_state = state.copy()
        
        action_type = action["type"]
        
        if action_type == "move":
            self._apply_move(new_state, action)
        elif action_type == "pick":
            self._apply_pick(new_state, action)
        elif action_type == "drop":
            self._apply_drop(new_state, action)
        elif action_type == "open":
            self._apply_open(new_state, action)
        elif action_type == "close":
            self._apply_close(new_state, action)
        elif action_type == "heat":
            self._apply_heat(new_state, action)
        else:
            raise ValueError(f"Unknown action type: {action_type}")
        
        return new_state
    
    def _apply_move(self, state: State, action: Dict[str, Any]) -> None:
        """Apply move action."""
        direction = action["direction"]
        new_x = state.robot[0] + direction[0]
        new_y = state.robot[1] + direction[1]
        
        # Validate move is within bounds
        if not (0 <= new_x < state.width and 0 <= new_y < state.height):
            raise ValueError(f"Move to ({new_x}, {new_y}) is out of bounds")
        
        state.robot = (new_x, new_y)
    
    def _apply_pick(self, state: State, action: Dict[str, Any]) -> None:
        """Apply pick action."""
        obj_name = action["object"]
        
        # Validate object exists and is at robot position
        if obj_name not in state.objects:
            raise ValueError(f"Object {obj_name} does not exist")
        
        obj = state.objects[obj_name]
        if obj.position != state.robot:
            raise ValueError(f"Object {obj_name} is not at robot position")
        
        # Validate gripper is empty
        if state.gripper is not None:
            raise ValueError("Gripper is not empty")
        
        state.gripper = obj_name
    
    def _apply_drop(self, state: State, action: Dict[str, Any]) -> None:
        """Apply drop action."""
        obj_name = action["object"]
        
        # Validate object is in gripper
        if state.gripper != obj_name:
            raise ValueError(f"Object {obj_name} is not in gripper")
        
        # Update object position to robot position
        state.objects[obj_name].position = state.robot
        state.gripper = None
    
    def _apply_open(self, state: State, action: Dict[str, Any]) -> None:
        """Apply open container action."""
        container_name = action["container"]
        
        # Validate container exists and is at robot position
        if container_name not in state.objects:
            raise ValueError(f"Container {container_name} does not exist")
        
        container = state.objects[container_name]
        if container.position != state.robot:
            raise ValueError(f"Container {container_name} is not at robot position")
        
        # Validate container is not already open
        if container_name in state.open_containers:
            raise ValueError(f"Container {container_name} is already open")
        
        state.open_containers.add(container_name)
    
    def _apply_close(self, state: State, action: Dict[str, Any]) -> None:
        """Apply close container action."""
        container_name = action["container"]
        
        # Validate container exists and is at robot position
        if container_name not in state.objects:
            raise ValueError(f"Container {container_name} does not exist")
        
        container = state.objects[container_name]
        if container.position != state.robot:
            raise ValueError(f"Container {container_name} is not at robot position")
        
        # Validate container is open
        if container_name not in state.open_containers:
            raise ValueError(f"Container {container_name} is not open")
        
        state.open_containers.remove(container_name)
    
    def _apply_heat(self, state: State, action: Dict[str, Any]) -> None:
        """Apply heat action."""
        obj_name = action["object"]
        
        # Validate object exists and is in gripper
        if obj_name not in state.objects:
            raise ValueError(f"Object {obj_name} does not exist")
        
        if state.gripper != obj_name:
            raise ValueError(f"Object {obj_name} is not in gripper")
        
        # Mark object as heated
        state.objects[obj_name].properties["is_heated"] = True
