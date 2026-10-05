import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import argparse
from streamgate.inference import StreamingEngine


def main():
    parser = argparse.ArgumentParser(description="Generate bytes with explicit streaming state")
    parser.add_argument("--model", choices=["linear", "softmax"], default="linear")
    parser.add_argument("--prompt", default="Holmes said, ")
    parser.add_argument("--bytes", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    session = StreamingEngine.load(kind=args.model).session(args.prompt)
    output = session.generate(args.bytes, args.temperature, args.seed)
    print(args.prompt + output.decode("utf-8", errors="replace"))
    print(f"Attention state: {session.attention_state_bytes:,} bytes; position: {session.position}")


if __name__ == "__main__":
    main()
