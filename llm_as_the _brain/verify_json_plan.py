#!/usr/bin/env python3
"""
JSON Plan Verifier - Simple Usage Example
=========================================

This script demonstrates how to verify a JSON plan from an LLM
using the mathematical verification framework.

Usage:
    python verify_json_plan.py --plan '{"actions": [{"type": "pick", "object": "bowl"}]}'
    python verify_json_plan.py --file plan.json
"""

import argparse
import json
import sys
from env.backend import Environment
from env.state import State, Object
from planner.plan_verifier import JSONPlanProcessor


def create_test_state(scenario: str = "safe_microwave") -> State:
    """Create a test state for verification."""
    env = Environment(5, 5)
    
    if scenario == "safe_microwave":
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
    elif scenario == "tidy_desk":
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
    else:
        # Default scenario
        objects = {
            "bowl": Object(
                position=(0, 0),
                properties={"is_microwave_safe": True, "is_heated": False}
            )
        }
    
    return env.create_state(objects, robot_pos=(0, 0))


def verify_plan_from_string(json_plan: str, scenario: str = "safe_microwave"):
    """Verify a JSON plan from string."""
    print(f"🔍 Verifying JSON plan for scenario: {scenario}")
    print(f"📋 Plan: {json_plan}")
    print("=" * 60)
    
    # Create environment and processor
    env = Environment(5, 5)
    processor = JSONPlanProcessor(env)
    
    # Create test state
    initial_state = create_test_state(scenario)
    print(f"🏠 Initial state: {initial_state}")
    print()
    
    # Validate JSON structure
    is_valid, message = processor.validate_json_structure(json_plan)
    print(f"✅ JSON validation: {message}")
    
    if not is_valid:
        print(f"❌ Invalid JSON plan: {message}")
        print("\n📝 Valid JSON plan formats:")
        print(processor.suggest_plan_format())
        return False
    
    # Verify plan
    result = processor.process_plan(json_plan, initial_state, scenario)
    
    # Print results
    print(f"\n📊 Verification Results:")
    print(f"   Overall safe: {'✅ YES' if result.overall_safe else '❌ NO'}")
    print(f"   Goal achieved: {'✅ YES' if result.final_goal_achieved else '❌ NO'}")
    print(f"   Execution success: {'✅ YES' if result.execution_success else '❌ NO'}")
    print(f"   Actions verified: {len(result.action_results)}")
    
    print(f"\n🔍 Action-by-Action Analysis:")
    for i, action_result in enumerate(result.action_results):
        status = "✅ SAFE" if action_result['is_safe'] else "❌ UNSAFE"
        print(f"   Action {i+1}: {action_result['action']} - {status}")
        
        if action_result['similarity_score'] is not None:
            print(f"     Similarity score: {action_result['similarity_score']:.3f}")
        if action_result['transport_distance'] is not None:
            print(f"     Transport distance: {action_result['transport_distance']:.3f}")
        if action_result['violations']:
            print(f"     Violations: {', '.join(action_result['violations'])}")
    
    if result.violation_reasons:
        print(f"\n⚠️ Safety Violations:")
        for violation in result.violation_reasons:
            print(f"   - {violation}")
    
    # Export detailed report
    report_filename = f"verification_report_{scenario}.json"
    processor.verifier.export_verification_report(result, report_filename)
    print(f"\n📁 Detailed report saved to: {report_filename}")
    
    return result.execution_success


def verify_plan_from_file(filename: str, scenario: str = "safe_microwave"):
    """Verify a JSON plan from file."""
    try:
        with open(filename, 'r') as f:
            json_plan = f.read()
        return verify_plan_from_string(json_plan, scenario)
    except FileNotFoundError:
        print(f"❌ File not found: {filename}")
        return False
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Verify JSON plans using mathematical verification framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python verify_json_plan.py --plan '{"actions": [{"type": "pick", "object": "bowl"}]}'
  python verify_json_plan.py --file plan.json --scenario tidy_desk
  python verify_json_plan.py --plan '{"type": "pick", "object": "bowl"}' --scenario safe_microwave
        """
    )
    
    parser.add_argument(
        "--plan",
        help="JSON plan as string"
    )
    
    parser.add_argument(
        "--file",
        help="JSON plan file"
    )
    
    parser.add_argument(
        "--scenario",
        choices=["safe_microwave", "tidy_desk", "make_tea"],
        default="safe_microwave",
        help="Test scenario"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    if not args.plan and not args.file:
        print("❌ Error: Must provide either --plan or --file")
        parser.print_help()
        sys.exit(1)
    
    if args.plan and args.file:
        print("❌ Error: Cannot provide both --plan and --file")
        sys.exit(1)
    
    try:
        if args.plan:
            success = verify_plan_from_string(args.plan, args.scenario)
        else:
            success = verify_plan_from_file(args.file, args.scenario)
        
        if success:
            print(f"\n🎉 Plan verification completed successfully!")
            sys.exit(0)
        else:
            print(f"\n⚠️ Plan verification completed with issues.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Verification interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during verification: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
