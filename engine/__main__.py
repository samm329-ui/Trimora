import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Trimora Engine Pipeline")
    parser.add_argument("-i", "--input", required=True, help="Input video file path")
    parser.add_argument("-o", "--output", default=None, help="Output directory for results")
    parser.add_argument("--category", default="general", help="Video category (business, tech, education, entertainment, general)")
    parser.add_argument("--groq-key", default=None, help="Groq API key (overrides GROQ_API_KEY env)")
    parser.add_argument("--gemini-key", default=None, help="Gemini API key (overrides GEMINI_API_KEY env)")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    if args.groq_key:
        os.environ["GROQ_API_KEY"] = args.groq_key
    if args.gemini_key:
        os.environ["GEMINI_API_KEY"] = args.gemini_key

    if not os.environ.get("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY must be set via --groq-key or GROQ_API_KEY env var")
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    from engine.pipeline import Pipeline

    pipeline = Pipeline()
    result = pipeline.run(args.input, category=args.category)

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        output_path = os.path.join(args.output, f"{result.video_id}_result.json")
        with open(output_path, "w") as f:
            json.dump({
                "video_id": result.video_id,
                "candidates": result.candidates,
                "best_clip": result.best_clip,
                "stats": result.stats,
                "error": result.error,
            }, f, indent=2)
        print(f"Results written to {output_path}")

    if result.error:
        print(f"Error: {result.error}")
        sys.exit(1)

    print(f"\nPipeline Results:")
    print(f"  Video ID:      {result.video_id}")
    print(f"  Candidates:    {len(result.candidates)}")
    if result.best_clip:
        print(f"  Best Clip:     {result.best_clip.get('hook_text', '')[:60]}...")
        print(f"  Best Score:    {result.best_clip.get('total_score', 0):.3f}")
        print(f"  Duration:      {result.best_clip.get('total_duration', 0):.1f}s")
    print(f"  Processing:    {result.stats.get('processing_time_seconds', 0):.1f}s")
    print(f"  Pipeline v{result.stats.get('pipeline_version', '?')}")

    if args.json:
        print(json.dumps({
            "video_id": result.video_id,
            "candidates": result.candidates,
            "best_clip": result.best_clip,
            "stats": result.stats,
            "error": result.error,
        }, indent=2))


if __name__ == "__main__":
    main()
