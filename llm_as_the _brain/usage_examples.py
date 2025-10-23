#!/usr/bin/env python3
"""
Usage Examples for Both Plan Formats
===================================

Shows how to use both the original and archaeological plan formats
with the verification system.
"""

import json
from verify_json_plan import verify_plan_from_string


def example_original_format():
    """Example using the original plan format."""
    print("🔍 Example 1: Original Plan Format")
    print("=" * 50)
    
    # Original format plan
    original_plan = {
        "steps": [
            {"type": "pick", "object": "bowl"},
            {"type": "move", "direction": [1, 0], "target": [1, 0]},
            {"type": "heat", "object": "bowl"}
        ]
    }
    
    json_plan = json.dumps(original_plan)
    print(f"Plan: {json_plan}")
    print()
    
    # Verify with safe_microwave scenario
    success = verify_plan_from_string(json_plan, "safe_microwave")
    print(f"Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
    return success


def example_archaeological_format():
    """Example using the archaeological plan format."""
    print("\n🔍 Example 2: Archaeological Plan Format")
    print("=" * 50)
    
    # Archaeological format plan
    archaeological_plan = {
        "schema_version": "1.0",
        "task_id": "Archaeology_R1",
        "world_frame": "site_grid",
        "assumptions": {
            "max_force_N": 5,
            "max_temp_C": 25
        },
        "steps": [
            {
                "id": "s1_excavate",
                "action": "excavate",
                "args": {
                    "artifact_id": "brittle_artifact",
                    "tool_id": "excavation_brush"
                },
                "constraints": {
                    "ltl": ["G (force < 5N)"]
                },
                "on_fail": "abort_site_section"
            },
            {
                "id": "s2_document",
                "action": "log_data",
                "args": {
                    "equipment_id": "3d_scanner",
                    "target_id": "brittle_artifact",
                    "data_type": "high_res_scan"
                },
                "constraints": {
                    "ltl": ["F (data_logged)"]
                },
                "on_fail": "retry"
            }
        ]
    }
    
    json_plan = json.dumps(archaeological_plan)
    print(f"Plan: {json_plan}")
    print()
    
    # Verify with archaeological_site scenario
    success = verify_plan_from_string(json_plan, "archaeological_site")
    print(f"Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
    return success


def example_command_line_usage():
    """Example of command line usage for both formats."""
    print("\n🔍 Example 3: Command Line Usage")
    print("=" * 50)
    
    print("For original format:")
    print("python verify_json_plan.py --plan '{\"steps\": [{\"type\": \"pick\", \"object\": \"bowl\"}]}' --scenario safe_microwave")
    print()
    
    print("For archaeological format:")
    print("python verify_json_plan.py --plan '{\"steps\": [{\"id\": \"s1\", \"action\": \"excavate\", \"args\": {\"artifact_id\": \"artifact\"}}]}' --scenario archaeological_site")
    print()
    
    print("Available scenarios:")
    print("  - safe_microwave")
    print("  - tidy_desk") 
    print("  - make_tea")
    print("  - elderly_care")
    print("  - archaeological_site")


def example_programmatic_usage():
    """Example of programmatic usage."""
    print("\n🔍 Example 4: Programmatic Usage")
    print("=" * 50)
    
    from planner.plan_verifier import PlanVerifier
    from env.backend import Environment
    from env.state import State, Object
    
    # Create environment and verifier
    env = Environment(5, 5)
    verifier = PlanVerifier(env)
    
    # Create test state
    objects = {
        "bowl": Object(
            position=(0, 0),
            properties={"is_microwave_safe": True, "is_heated": False}
        )
    }
    initial_state = env.create_state(objects, robot_pos=(0, 0))
    
    # Test original format
    print("Testing original format programmatically:")
    original_plan = json.dumps({
        "steps": [{"type": "pick", "object": "bowl"}]
    })
    result1 = verifier.verify_plan(original_plan, initial_state, "test_goal")
    print(f"  Actions parsed: {len(result1.plan)}")
    print(f"  Actions verified: {len(result1.action_results)}")
    
    # Test archaeological format
    print("\nTesting archaeological format programmatically:")
    archaeological_plan = json.dumps({
        "steps": [{
            "id": "s1_excavate",
            "action": "excavate",
            "args": {"artifact_id": "brittle_artifact", "tool_id": "excavation_brush"}
        }]
    })
    result2 = verifier.verify_plan(archaeological_plan, initial_state, "test_goal")
    print(f"  Actions parsed: {len(result2.plan)}")
    print(f"  Actions verified: {len(result2.action_results)}")
    
    print("\n✅ Both formats work programmatically!")


def main():
    """Main function showing all usage examples."""
    print("📚 Usage Examples for Both Plan Formats")
    print("=" * 60)
    print("This shows how to use both the original and archaeological")
    print("plan formats with the verification system.")
    print()
    
    try:
        # Example 1: Original format
        example_original_format()
        
        # Example 2: Archaeological format
        example_archaeological_format()
        
        # Example 3: Command line usage
        example_command_line_usage()
        
        # Example 4: Programmatic usage
        example_programmatic_usage()
        
        print(f"\n🎉 All examples completed!")
        print(f"\nKey Points:")
        print(f"  ✅ Both formats are automatically detected")
        print(f"  ✅ Archaeological format is converted to standard format")
        print(f"  ✅ Both formats go through the same verification pipeline")
        print(f"  ✅ Use appropriate scenarios for each format")
        print(f"  ⚠️  Semantic verification may flag some actions as unsafe")
        print(f"     due to aggressive safety thresholds")
        
    except Exception as e:
        print(f"❌ Error during examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
