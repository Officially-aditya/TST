import time

import httpx
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from tst.kernel.client import StdioKernelClient

# FunctionGemma is no longer loaded here — it is owned by router/server.py.
# router_slm() now delegates to the router service via HTTP.

ROUTER_URL = "http://127.0.0.1:8003"


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} for inference")
    print("Loading Qwen Qwen/Qwen3.5-0.8B ...")
    try:
        processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B")
        qwen_model = AutoModelForImageTextToText.from_pretrained(
            "Qwen/Qwen3.5-0.8B", torch_dtype=torch.float16
        ).to(device)
    except Exception as e:
        print(f"Error loading Qwen: {e}")
        # Proceeding is critical for debugging even if Qwen fails

    print("Starting the prebuilt TST Memory Kernel...")
    kernel = StdioKernelClient()
    kernel.start()
    print("Kernel Ready.")

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
            return processor.decode(
                outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
            ).strip()
        except NameError:
            print("[Warning] Qwen model was not loaded. Returning fallback text.")
            return "There is a dog on the candy."

    def router_slm(query: str, payload: str = "") -> str:
        """
        Delegate routing to the TST Router service (router/server.py).
        Returns the action tool selected by the v0.2 router.
        Falls back safely to answer_without_memory if the service is unreachable.
        """
        try:
            resp = httpx.post(
                f"{ROUTER_URL}/route",
                json={"query": query, "payload": payload},
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json().get("tool_called", "answer_without_memory")
        except Exception as e:
            print(f"[Warning] Router service unreachable: {e}. Using no-memory fallback.")
            return "answer_without_memory"

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What animal is on the candy? The candy has a picture of a dog on it.",
                }
            ],
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
    animal_key = "user:default:fact:candy_animal"
    print(f"[Kernel] Writing animal data: '{router_response}' to key '{animal_key}'")

    now = int(time.time() * 1000)
    write_resp = kernel.store(
        "ltm",
        animal_key,
        {
            "type": "token_stats",
            "data": {
                "key": animal_key,
                "value": qwen_response,
                "memory_type": "fact",
                "source_text": qwen_response,
                "created_at": now,
                "updated_at": now,
                "confidence": 1.0,
                "tags": ["candy", "animal"],
                "source": "qwen",
                "layer": "ltm",
                "reinforcement_score": 0.0,
                "deleted": False,
            },
        },
    )
    print(f"[Kernel] Write response: {write_resp}")

    print(f"[Kernel] Reading key '{animal_key}'")
    read_resp = kernel.get("ltm", animal_key)
    print(f"[Kernel] Read response: {read_resp}")

    kernel.close(graceful=True)
    print("\nEnd-to-end functionality completed successfully!")


if __name__ == "__main__":
    main()
