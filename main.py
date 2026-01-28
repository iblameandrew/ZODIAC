import asyncio
import argparse
from simulation import zodiac_simulation, SimParams

async def run_cli_simulation():
    parser = argparse.ArgumentParser(description="ZODIAC Agent Simulation CLI")
    parser.add_argument("--steps", type=int, default=50, help="Max steps per episode")
    parser.add_argument("--agents", type=int, default=4, help="Number of agents")
    parser.add_argument("--keywords", type=str, default="ethereal,cybernetic,obsidian", help="Comma-sep keywords")
    parser.add_argument("--test", action="store_true", help="Enable test mode debug logs")
    
    args = parser.parse_args()
    
    params = SimParams(
        keywords=args.keywords.split(","),
        max_steps=args.steps,
        K=args.agents,
        test_mode=args.test
    )
    
    print(f"Starting ZODIAC Simulation CLI with {params.K} agents...")
    print(f"Keywords: {params.keywords}")
    
    generator = zodiac_simulation(params)
    
    try:
        async for frame in generator:
            if "event" in frame and frame["event"] == "complete":
                print("\n>> Simulation Completed.")
                print(f"Stats: {frame['stats']}")
            else:
                # Print frame summary
                step = frame["step"]
                print(f"\nStep {step}")
                for agent in frame["agents"]:
                    # print clean text
                    print(f"Agent {agent['id']}: {agent['objective']} -> \"{agent['token']}\"")
                    if args.test:
                        print(f"   Pos: {agent['pos']}")

    except KeyboardInterrupt:
        print("\nStopping simulation...")

if __name__ == "__main__":
    asyncio.run(run_cli_simulation())
