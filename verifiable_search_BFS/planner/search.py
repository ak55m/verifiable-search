"""
Breadth-First Search planner with rule verification.
Searches for valid plans that satisfy all safety rules.
"""

from typing import List, Dict, Any, Set, Optional, Tuple
from collections import deque
import time
from env.state import State
from env.backend import Environment
from rules.verify import RuleVerifier


class BFSSearchPlanner:
    """BFS planner that respects safety rules."""
    
    def __init__(self, environment: Environment):
        """Initialize planner with environment and rule verifier."""
        self.env = environment
        self.verifier = RuleVerifier()
        self.visited_states: Set[int] = set()
    
    def plan(self, initial_state: State, goal: str, max_depth: int = 20) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Find a plan using BFS that satisfies all rules.
        
        Returns:
            plan: List of actions to reach goal
            metadata: Search statistics and info
        """
        start_time = time.time()
        self.visited_states.clear()
        
        # BFS queue: (state, path, depth)
        queue = deque([(initial_state, [], 0)])
        self.visited_states.add(hash(initial_state))
        
        nodes_expanded = 0
        max_queue_size = 1
        
        while queue:
            current_state, current_path, depth = queue.popleft()
            nodes_expanded += 1
            
            # Debug: print every 50 nodes
            if nodes_expanded % 50 == 0:
                print(f"  BFS: expanded {nodes_expanded} nodes, depth {depth}, queue size {len(queue)}")
            
            # Check if goal is reached
            if self.env.is_goal_state(current_state, goal):
                runtime = time.time() - start_time
                return current_path, {
                    "success": True,
                    "nodes_expanded": nodes_expanded,
                    "max_queue_size": max_queue_size,
                    "depth": depth,
                    "runtime_s": runtime
                }
            
            # Skip if max depth reached
            if depth >= max_depth:
                continue
            
            # Get all possible actions
            actions = self.env.get_available_actions(current_state)
            
            for action in actions:
                # Check if action violates rules
                if self.verifier.violates_rules(current_state, action):
                    continue  # Skip invalid actions
                
                # Apply action to get new state
                try:
                    new_state = self.env.apply_action(current_state, action)
                except ValueError:
                    continue  # Skip invalid actions
                
                # Check if goal is reached immediately after this action
                if self.env.is_goal_state(new_state, goal):
                    runtime = time.time() - start_time
                    new_path = current_path + [action]
                    return new_path, {
                        "success": True,
                        "nodes_expanded": nodes_expanded + 1,
                        "max_queue_size": max_queue_size,
                        "depth": depth + 1,
                        "runtime_s": runtime
                    }
                
                # Check if we've seen this state before
                state_hash = hash(new_state)
                if state_hash in self.visited_states:
                    continue
                
                # Add to queue
                new_path = current_path + [action]
                queue.append((new_state, new_path, depth + 1))
                self.visited_states.add(state_hash)
            
            max_queue_size = max(max_queue_size, len(queue))
        
        # No plan found
        runtime = time.time() - start_time
        return [], {
            "success": False,
            "nodes_expanded": nodes_expanded,
            "max_queue_size": max_queue_size,
            "depth": depth,
            "runtime_s": runtime
        }
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get statistics about the last search."""
        return {
            "visited_states": len(self.visited_states),
            "search_space_size": len(self.visited_states)
        }


class AStarSearchPlanner:
    """A* planner with rule verification and heuristic."""
    
    def __init__(self, environment: Environment):
        """Initialize A* planner."""
        self.env = environment
        self.verifier = RuleVerifier()
        self.visited_states: Set[int] = set()
    
    def _heuristic(self, state: State, goal: str) -> int:
        """Heuristic function for A* search."""
        if goal == "safe_microwave":
            # Distance to microwave + whether bowl is heated
            microwave_pos = (2, 2)  # Assume microwave at (2,2)
            robot_dist = abs(state.robot[0] - microwave_pos[0]) + abs(state.robot[1] - microwave_pos[1])
            
            # Check if bowl is already heated
            for obj_name, obj in state.objects.items():
                if (obj_name == "bowl" and 
                    obj.properties.get("is_heated", False)):
                    return robot_dist  # Just need to get there
            
            return robot_dist + 1  # +1 for heating action
        
        elif goal == "tidy_desk":
            # Distance to shelf
            shelf_pos = (4, 4)
            return abs(state.robot[0] - shelf_pos[0]) + abs(state.robot[1] - shelf_pos[1])
        
        elif goal == "make_tea":
            # Distance to stove
            stove_pos = (2, 2)
            return abs(state.robot[0] - stove_pos[0]) + abs(state.robot[1] - stove_pos[1])
        
        return 0  # Default heuristic
    
    def plan(self, initial_state: State, goal: str, max_depth: int = 20) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Find a plan using A* search.
        
        Returns:
            plan: List of actions to reach goal
            metadata: Search statistics and info
        """
        start_time = time.time()
        self.visited_states.clear()
        
        # Priority queue: (f_score, g_score, state, path)
        import heapq
        queue = [(0, 0, initial_state, [])]
        self.visited_states.add(hash(initial_state))
        
        nodes_expanded = 0
        max_queue_size = 1
        
        while queue:
            f_score, g_score, current_state, current_path = heapq.heappop(queue)
            nodes_expanded += 1
            
            # Check if goal is reached
            if self.env.is_goal_state(current_state, goal):
                runtime = time.time() - start_time
                return current_path, {
                    "success": True,
                    "nodes_expanded": nodes_expanded,
                    "max_queue_size": max_queue_size,
                    "depth": len(current_path),
                    "runtime_s": runtime
                }
            
            # Skip if max depth reached
            if len(current_path) >= max_depth:
                continue
            
            # Get all possible actions
            actions = self.env.get_available_actions(current_state)
            
            for action in actions:
                # Check if action violates rules
                if self.verifier.violates_rules(current_state, action):
                    continue
                
                # Apply action to get new state
                try:
                    new_state = self.env.apply_action(current_state, action)
                except ValueError:
                    continue
                
                # Check if we've seen this state before
                state_hash = hash(new_state)
                if state_hash in self.visited_states:
                    continue
                
                # Calculate scores
                new_g_score = g_score + 1
                h_score = self._heuristic(new_state, goal)
                new_f_score = new_g_score + h_score
                
                # Add to queue
                new_path = current_path + [action]
                heapq.heappush(queue, (new_f_score, new_g_score, new_state, new_path))
                self.visited_states.add(state_hash)
            
            max_queue_size = max(max_queue_size, len(queue))
        
        # No plan found
        runtime = time.time() - start_time
        return [], {
            "success": False,
            "nodes_expanded": nodes_expanded,
            "max_queue_size": max_queue_size,
            "depth": len(current_path),
            "runtime_s": runtime
        }
