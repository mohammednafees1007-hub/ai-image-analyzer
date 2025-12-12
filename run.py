import json
import sys
from pipeline.aggregator import analyze_image

def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py <image_path>")
        raise SystemExit(1)

    image_path = sys.argv[1]
    result = analyze_image(image_path)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
