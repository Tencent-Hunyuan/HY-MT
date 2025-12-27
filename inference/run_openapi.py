import base64

from openai import OpenAI

import argparse

def main(args):

    client = OpenAI(api_key="EMPTY", base_url="http://localhost:8021/v1")

    stream = client.chat.completions.create(
        model=args.model_path,
        messages=[{"role": "user", "content": "请按面积大小对四大洋进行排序，并给出面积最小的洋是哪一个？"}],
        stream=True,
        temperature=0.6,
        top_p=0.9,
        extra_body={"top_k": 20, "repetition_penalty": 1.05, "stop_token_ids": [127960]}
    )
    for chunk in stream:
        if content := chunk.choices[0].delta.content:
            print(content, end="", flush=True)

if __name__ == "__main__":
    global_parser = argparse.ArgumentParser(description="inference", add_help=True)
    global_parser.add_argument(
        "--model_path",
        type=str,
        default="tencent/Hunyuan-1.8B-Instruct",
        help="model path or model name, e.g., 'Hunyuan-1.8b'",
    )
    main(global_parser.parse_args())

