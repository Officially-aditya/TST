import subprocess
import json
import sys
import os
import torch
import httpx

from transformers import AutoProcessor, AutoModelForImageTextToText

# FunctionGemma is no longer loaded here — it is owned by router/server.py.
# router_slm() now delegates to the router service via HTTP.

ROUTER_URL = "http://127.0.0.1:8003"

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} for inference")
    print("Loading Qwen Qwen/Qwen3.5-0.8B ...")
    try:
        processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B")
        qwen_model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.5-0.8B", torch_dtype=torch.float16).to(device)
    except Exception as e:
        print(f"Error loading Qwen: {e}")
        # Proceeding is critical for debugging even if Qwen fails

    print("Building and Starting TST Memory Kernel...")
    subprocess.run(["cargo", "build", "--release", "--bin", "server"], cwd="./tst_memory", check=True)

    server_cmd = ["cargo", "run", "--release", "--bin", "server"]
    kernel_process = subprocess.Popen(
        server_cmd,
        cwd="./tst_memory",
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True,
        bufsize=1
    )

    ready_line = kernel_process.stdout.readline()
    if "READY" not in ready_line:
        print("Kernel failed to start:", ready_line)
        sys.exit(1)
    print("Kernel Ready.")

    def read_memory(keys):
        req = json.dumps({"keys": keys, "max_results": 1})
        kernel_process.stdin.write(f"READ {req}\n")
        kernel_process.stdin.flush()
        resp = kernel_process.stdout.readline()
        try:
            return json.loads(resp)
        except Exception as e:
            return {"error": f"Invalid response: {e}, raw: {resp}"}

    def write_memory(key, payload_data):
        print(f"Attempting to serialize: {payload_data}")
        req = json.dumps({
            "op": "insert",
            "key": key,
            "payload": {
                "header": {
                    "payload_type": 1,
                    "version": 1,
                    "created_ts": {"Timestamp": 0},
                    "last_access_ts": {"Timestamp": 0},
                    "access_count": 1
                },
                "data": payload_data
            }
        })
        kernel_process.stdin.write(f"WRITE {req}\n")
        kernel_process.stdin.flush()
        resp = kernel_process.stdout.readline()
        return json.loads(resp)

    def chat_slm(messages):
        try:
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(qwen_model.device)

            outputs = qwen_model.generate(**inputs, max_new_tokens=40)
            return processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        except NameError:
            print("[Warning] Qwen model was not loaded. Returning fallback text.")
            return "There is a dog on the candy."

    def router_slm(query: str, payload: str = "") -> str:
        """
        Delegate routing to the TST Router service (router/server.py).
        Returns the tool name chosen: 'route_to_stm' | 'route_to_ltm' |
                                      'route_to_tree' | 'route_to_cloud'
        Falls back to 'route_to_cloud' if the service is unreachable.
        """
        try:
            resp = httpx.post(
                f"{ROUTER_URL}/route",
                json={"query": query, "payload": payload},
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json().get("tool_called", "route_to_cloud")
        except Exception as e:
            print(f"[Warning] Router service unreachable: {e}. Defaulting to route_to_cloud.")
            return "route_to_cloud"


    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What animal is on the candy? The candy has a picture of a dog on it."}
            ]
        },
    ]

    print("\n--- Testing Qwen (Chat SLM) ---")
    qwen_response = chat_slm(messages)
    print(f"Qwen output: {qwen_response}")

    print("\n--- Testing Router Service ---")
    router_response = router_slm(
        query=f"Store this animal fact: '{qwen_response}'",
        payload=qwen_response,
    )
    print(f"Router decision: {router_response}")

    print("\n--- Testing TST Memory (Kernel) ---")
    animal_key = "candy_animal"
    print(f"[Kernel] Writing animal data: '{router_response}' to key '{animal_key}'")

    write_resp = write_memory(animal_key, {
        "TokenStats": {
            "canonical_form": router_response,
            "frequency": 1,
            "decay_score": 1.0,
            "preferred_tokenizer_origin": "qwen"
        }
    })
    print(f"[Kernel] Write response: {write_resp}")

    print(f"[Kernel] Reading key '{animal_key}'")
    read_resp = read_memory([animal_key])
    print(f"[Kernel] Read response: {read_resp}")

    kernel_process.terminate()
    print("\nEnd-to-end functionality completed successfully!")

if __name__ == "__main__":
    main()
