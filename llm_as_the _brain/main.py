#!/usr/bin/env python3
"""
Main entry point for LLMs-Only Verifiable Search Project

Mathematical Verification Framework for Safe Action Invention
============================================================

This project implements the mathematical foundations for safe LLM action invention
as described in "Mathematical Verification Enables Safe LLM Action Invention: 
Beyond Predefined Schemas".

The framework enables LLMs to invent novel actions while providing mathematical
guarantees of safety through rigorous verification using:

- Bisimulation Metrics (Action Similarity)
- Maximum Causal Entropy (Schema Learning) 
- Formal Verification (Safety Proofs)
- Optimal Transport (State Distribution Matching)
- Reinforcement Learning Theory (Quality Functions)
- Information Theory (Feature Selection)

Usage:
    python main.py --mode safe_invention    # Test safe action invention
    python main.py --mode verification      # Test verification frameworks
    python main.py --mode comparison        # Compare with baselines
    python main.py --mode full_evaluation   # Run full evaluation
"""

import argparse
import sys
import os
import time
from typing import List, Dict, Any
from env.backend import Environment
from env.state import State, Object
from planner.plan_verifier import PlanVerifier, JSONPlanProcessor
from verification.bisimulation import BisimulationMetrics
from verification.entropy import MaximumCausalEntropy
from verification.formal import FormalVerifier
from verification.optimal_transport import OptimalTransport


def create_test_scenarios() -> Dict[str, State]:
    """Create test scenarios for safe action invention."""
    env = Environment(5, 5)
    
    scenarios = {}
    
    # Safe Microwave Scenario
    objects = {
        "bowl": Object(
            position=(0, 0),
            properties={"is_microwave_safe": True, "is_heated": False}
        ),
        "microwave": Object(
            position=(2, 2),
            properties={"is_microwave": True, "is_container": False}
        )
    }
    scenarios["safe_microwave"] = env.create_state(objects, robot_pos=(0, 0))
    
    # Tidy Desk Scenario
    objects = {
        "book": Object(
            position=(0, 0),
            properties={"is_fragile": True}
        ),
        "shelf": Object(
            position=(4, 4),
            properties={"is_container": False, "is_surface": True}
        )
    }
    scenarios["tidy_desk"] = env.create_state(objects, robot_pos=(0, 0))
    
    # Make Tea Scenario
    objects = {
        "cup": Object(
            position=(0, 0),
            properties={"is_breakable": True}
        ),
        "kettle": Object(
            position=(1, 1),
            properties={"is_heavy": True}
        ),
        "stove": Object(
            position=(2, 2),
            properties={"is_heating_element": True}
        )
    }
    scenarios["make_tea"] = env.create_state(objects, robot_pos=(0, 0))
    
    return scenarios


def test_json_plan_verification():
    """Test JSON plan verification system."""
    print("🔍 Testing JSON Plan Verification System")
    print("=" * 50)
    
    env = Environment(5, 5)
    processor = JSONPlanProcessor(env)
    scenarios = create_test_scenarios()
    
    # Test JSON plans
    test_plans = {
        "safe_microwave": '''
        {
            "goal": "safe_microwave",
            "actions": [
                {"type": "pick", "object": "bowl"},
                {"type": "move", "direction": [1, 0], "target": [1, 0]},
                {"type": "move", "direction": [1, 0], "target": [2, 0]},
                {"type": "move", "direction": [0, 1], "target": [2, 1]},
                {"type": "move", "direction": [0, 1], "target": [2, 2]},
                {"type": "heat", "object": "bowl"}
            ]
        }
        ''',
        "tidy_desk": '''
        {
            "goal": "tidy_desk", 
            "actions": [
                {"type": "pick", "object": "book"},
                {"type": "move", "direction": [1, 0], "target": [1, 0]},
                {"type": "move", "direction": [1, 0], "target": [2, 0]},
                {"type": "move", "direction": [1, 0], "target": [3, 0]},
                {"type": "move", "direction": [1, 0], "target": [4, 0]},
                {"type": "move", "direction": [0, 1], "target": [4, 1]},
                {"type": "move", "direction": [0, 1], "target": [4, 2]},
                {"type": "move", "direction": [0, 1], "target": [4, 3]},
                {"type": "move", "direction": [0, 1], "target": [4, 4]},
                {"type": "drop", "object": "book"}
            ]
        }
        ''',
        "make_tea": '''
        {
            "goal": "make_tea",
            "actions": [
                {"type": "pick", "object": "cup"},
                {"type": "move", "direction": [1, 0], "target": [1, 0]},
                {"type": "move", "direction": [1, 0], "target": [2, 0]},
                {"type": "move", "direction": [0, 1], "target": [2, 1]},
                {"type": "move", "direction": [0, 1], "target": [2, 2]},
                {"type": "drop", "object": "cup"},
                {"type": "pick", "object": "kettle"},
                {"type": "drop", "object": "kettle"}
            ]
        }
        '''
    }
    
    for scenario_name, initial_state in scenarios.items():
        print(f"\n📋 Testing scenario: {scenario_name}")
        print(f"Initial state: {initial_state}")
        
        if scenario_name in test_plans:
            json_plan = test_plans[scenario_name]
            
            # Validate JSON structure
            is_valid, message = processor.validate_json_structure(json_plan)
            print(f"JSON validation: {message}")
            
            if is_valid:
                # Verify plan
                result = processor.process_plan(json_plan, initial_state, scenario_name)
                
                print(f"📊 Verification results:")
                print(f"   Overall safe: {result.overall_safe}")
                print(f"   Goal achieved: {result.final_goal_achieved}")
                print(f"   Execution success: {result.execution_success}")
                print(f"   Actions verified: {len(result.action_results)}")
                
                # Show action details
                for i, action_result in enumerate(result.action_results):
                    status = "✅ SAFE" if action_result['is_safe'] else "❌ UNSAFE"
                    print(f"   Action {i+1}: {action_result['action']} - {status}")
                    if action_result['similarity_score'] is not None:
                        print(f"     Similarity: {action_result['similarity_score']:.3f}")
                    if action_result['violations']:
                        print(f"     Violations: {action_result['violations']}")
            else:
                print(f"❌ Invalid JSON plan: {message}")
        else:
            print(f"⚠️ No test plan available for {scenario_name}")


def test_verification_frameworks():
    """Test individual verification frameworks."""
    print("🔍 Testing Verification Frameworks")
    print("=" * 50)
    
    env = Environment(5, 5)
    scenarios = create_test_scenarios()
    state = scenarios["safe_microwave"]
    
    # Test Bisimulation Metrics
    print("\n1. Testing Bisimulation Metrics")
    bisimulation = BisimulationMetrics()
    action1 = {"type": "pick", "object": "bowl"}
    action2 = {"type": "pick", "object": "bowl"}
    
    distance = bisimulation.bisimulation_distance(state, action1, action2, env)
    print(f"   Bisimulation distance: {distance:.3f}")
    
    # Test Maximum Causal Entropy
    print("\n2. Testing Maximum Causal Entropy")
    entropy_learner = MaximumCausalEntropy()
    
    # Create sample demonstrations
    demonstrations = [
        (state, action1, state),  # Simplified demo
        (state, action2, state)
    ]
    features = ["robot_at_position", "gripper_contains", "action_type"]
    
    weights = entropy_learner.learn_schema(demonstrations, features)
    print(f"   Learned weights: {weights}")
    
    # Test Formal Verification
    print("\n3. Testing Formal Verification")
    formal_verifier = FormalVerifier()
    
    is_safe, violations = formal_verifier.verify_action_safety(state, action1, env)
    print(f"   Action safety: {is_safe}")
    if violations:
        print(f"   Violations: {violations}")
    
    # Test Optimal Transport
    print("\n4. Testing Optimal Transport")
    optimal_transport = OptimalTransport()
    
    distance = optimal_transport.compare_action_outcomes(state, action1, action2, env)
    print(f"   Transport distance: {distance:.3f}")


def test_mathematical_properties():
    """Test mathematical properties of the verification framework."""
    print("🧮 Testing Mathematical Properties")
    print("=" * 50)
    
    env = Environment(5, 5)
    scenarios = create_test_scenarios()
    state = scenarios["safe_microwave"]
    
    # Test Wasserstein distance properties
    print("\n1. Testing Wasserstein Distance Properties")
    optimal_transport = OptimalTransport()
    
    # Test symmetry
    action1 = {"type": "pick", "object": "bowl"}
    action2 = {"type": "drop", "object": "bowl"}
    
    dist1 = optimal_transport.compare_action_outcomes(state, action1, action2, env)
    dist2 = optimal_transport.compare_action_outcomes(state, action2, action1, env)
    print(f"   Symmetry test: {abs(dist1 - dist2) < 1e-6}")
    
    # Test triangle inequality
    action3 = {"type": "move", "direction": (1, 0), "target": (1, 0)}
    dist12 = optimal_transport.compare_action_outcomes(state, action1, action2, env)
    dist23 = optimal_transport.compare_action_outcomes(state, action2, action3, env)
    dist13 = optimal_transport.compare_action_outcomes(state, action1, action3, env)
    
    triangle_inequality = dist13 <= dist12 + dist23
    print(f"   Triangle inequality: {triangle_inequality}")
    
    # Test bisimulation properties
    print("\n2. Testing Bisimulation Properties")
    bisimulation = BisimulationMetrics()
    
    # Test reflexivity
    dist_self = bisimulation.bisimulation_distance(state, action1, action1, env)
    print(f"   Reflexivity (distance to self): {dist_self:.6f}")
    
    # Test symmetry
    dist_12 = bisimulation.bisimulation_distance(state, action1, action2, env)
    dist_21 = bisimulation.bisimulation_distance(state, action2, action1, env)
    print(f"   Symmetry: {abs(dist_12 - dist_21) < 1e-6}")


def run_full_evaluation():
    """Run full evaluation of the JSON plan verification framework."""
    print("🔬 Running Full Evaluation")
    print("=" * 50)
    
    env = Environment(5, 5)
    processor = JSONPlanProcessor(env)
    scenarios = create_test_scenarios()
    
    # Test JSON plans
    test_plans = {
        "safe_microwave": '''
        {
            "goal": "safe_microwave",
            "actions": [
                {"type": "pick", "object": "bowl"},
                {"type": "move", "direction": [1, 0], "target": [1, 0]},
                {"type": "move", "direction": [1, 0], "target": [2, 0]},
                {"type": "move", "direction": [0, 1], "target": [2, 1]},
                {"type": "move", "direction": [0, 1], "target": [2, 2]},
                {"type": "heat", "object": "bowl"}
            ]
        }
        ''',
        "tidy_desk": '''
        {
            "goal": "tidy_desk", 
            "actions": [
                {"type": "pick", "object": "book"},
                {"type": "move", "direction": [1, 0], "target": [1, 0]},
                {"type": "move", "direction": [1, 0], "target": [2, 0]},
                {"type": "move", "direction": [1, 0], "target": [3, 0]},
                {"type": "move", "direction": [1, 0], "target": [4, 0]},
                {"type": "move", "direction": [0, 1], "target": [4, 1]},
                {"type": "move", "direction": [0, 1], "target": [4, 2]},
                {"type": "move", "direction": [0, 1], "target": [4, 3]},
                {"type": "move", "direction": [0, 1], "target": [4, 4]},
                {"type": "drop", "object": "book"}
            ]
        }
        ''',
        "make_tea": '''
        {
            "goal": "make_tea",
            "actions": [
                {"type": "pick", "object": "cup"},
                {"type": "move", "direction": [1, 0], "target": [1, 0]},
                {"type": "move", "direction": [1, 0], "target": [2, 0]},
                {"type": "move", "direction": [0, 1], "target": [2, 1]},
                {"type": "move", "direction": [0, 1], "target": [2, 2]},
                {"type": "drop", "object": "cup"},
                {"type": "pick", "object": "kettle"},
                {"type": "drop", "object": "kettle"}
            ]
        }
        '''
    }
    
    results = {}
    
    for scenario_name, initial_state in scenarios.items():
        print(f"\n📊 Evaluating scenario: {scenario_name}")
        
        if scenario_name in test_plans:
            json_plan = test_plans[scenario_name]
            
            start_time = time.time()
            result = processor.process_plan(json_plan, initial_state, scenario_name)
            execution_time = time.time() - start_time
            
            results[scenario_name] = {
                "success": result.execution_success,
                "steps": len(result.plan),
                "execution_time": execution_time,
                "verification_methods": "mathematical_framework",
                "overall_safe": result.overall_safe,
                "goal_achieved": result.final_goal_achieved,
                "safe_actions": sum(1 for ar in result.action_results if ar['is_safe']),
                "total_actions": len(result.action_results)
            }
            
            print(f"   Success: {result.execution_success}")
            print(f"   Steps: {len(result.plan)}")
            print(f"   Time: {execution_time:.3f}s")
            print(f"   Overall safe: {result.overall_safe}")
            print(f"   Safe actions: {sum(1 for ar in result.action_results if ar['is_safe'])}/{len(result.action_results)}")
        else:
            print(f"   ⚠️ No test plan available")
            results[scenario_name] = {
                "success": False,
                "steps": 0,
                "execution_time": 0,
                "verification_methods": "none",
                "overall_safe": False,
                "goal_achieved": False,
                "safe_actions": 0,
                "total_actions": 0
            }
    
    # Print summary
    print(f"\n📈 Evaluation Summary")
    print("=" * 30)
    total_scenarios = len(results)
    successful_scenarios = sum(1 for r in results.values() if r["success"])
    avg_steps = sum(r["steps"] for r in results.values()) / total_scenarios
    avg_time = sum(r["execution_time"] for r in results.values()) / total_scenarios
    avg_safe_actions = sum(r["safe_actions"] for r in results.values()) / total_scenarios
    avg_total_actions = sum(r["total_actions"] for r in results.values()) / total_scenarios
    
    print(f"Success Rate: {successful_scenarios}/{total_scenarios} ({100*successful_scenarios/total_scenarios:.1f}%)")
    print(f"Average Steps: {avg_steps:.1f}")
    print(f"Average Time: {avg_time:.3f}s")
    print(f"Safety Rate: {avg_safe_actions/avg_total_actions*100:.1f}% ({avg_safe_actions:.1f}/{avg_total_actions:.1f} actions)")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LLMs-Only Verifiable Search Project - Mathematical Verification Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode safe_invention    # Test safe action invention
  python main.py --mode verification      # Test verification frameworks  
  python main.py --mode mathematical     # Test mathematical properties
  python main.py --mode full_evaluation  # Run full evaluation
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["json_verification", "verification", "mathematical", "full_evaluation"],
        default="json_verification",
        help="Test mode to run"
    )
    
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["safe_microwave", "tidy_desk", "make_tea"],
        help="Scenarios to test"
    )
    
    parser.add_argument(
        "--output_dir",
        default="results",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        if args.mode == "json_verification":
            test_json_plan_verification()
        elif args.mode == "verification":
            test_verification_frameworks()
        elif args.mode == "mathematical":
            test_mathematical_properties()
        elif args.mode == "full_evaluation":
            run_full_evaluation()
        else:
            print(f"Unknown mode: {args.mode}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    print("\n✅ Mathematical verification framework evaluation completed!")


if __name__ == "__main__":
    main()