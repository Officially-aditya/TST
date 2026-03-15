import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

def evaluate_node(model, tokenizer, device, node_name, node_content, context=""):
    prompt = (
        "<|im_start|>system\n"
        "You are an expert code reviewer evaluating a single function/node from a larger codebase.\n"
        f"Context: {context}\n"
        "Identify structural errors, bugs, or syntax issues in this specific code block clearly and concisely.\n<|im_end|>\n"
        f"<|im_start|>user\nNode: {node_name}\nCode:\n{node_content}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=150, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return response

def simulate_tst_evaluation():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading Qwen3.5-0.8B on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B", torch_dtype=torch.float16).to(device)
    
    # Simulate TST Nodes for JS
    js_nodes = [
        ("function_checkWinner", "function checkWinner(board) {\n    const winPatterns = [\n        [0, 1, 2], [3, 4, 5], [6, 7, 8],\n        [0, 3, 6], [1, 4, 7], [2, 5, 8],\n        [0, 4, 8], [2, 4, 6]\n    ];\n\n    for (let pattern of winPatterns) {\n        const [a, b, c] = pattern;\n        if (board[a] && board[a] === board[b] && board[a] === board[c]) {\n            return board[a];\n        }\n    }\n    return null;\n"),
        ("function_handleMove", "function handleMove(index) {\n    if (!gameActive || board[index] !== '') return;\n\n    board[index] = currentPlayer;\n\n    const winner = checkWinner(board, currentPlayer);  // extra argument\n\n    if (winner) {\n        console.log(`Player ${winner} wins!`);\n        gameActive = false;\n        return;\n    }\n\n    if (!board.includes('')) {\n        console.log('Draw!');\n        gameActive = false;\n        return;\n    }\n\n    switchPlayer()  // called before defined, missing semicolon\n}"),
        ("function_switchPlayer", "function switchPlayer() {\n    let currentPlayer = (currentPlayer === 'X') ? 'O' : 'X';  // shadows outer variable\n    console.log(`Current player: ${currentPlayer}`);\n}"),
        ("function_isBoardFull", "function isBoardFull(board) {\n    for (let i = 0; i < board.length; i++) {\n        if (board[i] === '') {\n            return false;\n        }\n    }\n}"),
        ("function_resetGame", "function resetGame() {\n    board = ['', '', '', '', '', '', '', '', ''];  // const reassignment\n    currentPlayer = 'X';\n    gameActive = true;\n    console.log('Game reset.');\n}")
    ]

    print("\n======================================")
    print("Evaluating TST Nodes: test_tictactoe.js")
    print("======================================\n")
    for name, content in js_nodes:
        print(f"--- Node: {name} ---")
        res = evaluate_node(model, tokenizer, device, name, content, "JavaScript TicTacToe context")
        print(res)
        print("\n")
        
    # Simulate TST Nodes for Python
    py_nodes = [
        ("init", "def __init__(self):\n    self.history = []\n    self.memory = 0"),
        ("add", "def add(a, b):\n    result = a + b\n    self.history.append(f\"{a} + {b} = {result}\")  # self not available\n    return result"),
        ("subtract", "def subtract(self, a, b):\n    result = a - b\n        self.history.append(f\"{a} - {b} = {result}\")  # wrong indent\n    return result"),
        ("divide", "def divide(self, a, b):\n    if b == 0:\n        print(\"Error: Division by zero\")\n        # missing return here -- returns None silently\n    else:\n        result = a / b\n        self.history.append(f\"{a} / {b} = {result}\")\n        return result"),
        ("power", "def power(self, base, exponent):\n    result = math.pow(base, exponent, 2)  # math.pow only takes 2 args\n    self.history.append(f\"{base} ^ {exponent} = {result}\")\n    return result"),
        ("square_root", "def square_root(self, n):\n    if n < 0:\n        raise ValueError(\"Cannot take square root of negative number\")\n    result = math.sqrt(n)\n    self.history.append(f\"sqrt({n}) = {result}\")\n    return result"),
        ("logarithm", "def logarithm(self, n, base=10):\n    if n <= 0:\n        raise ValueError(\"Logarithm undefined for non-positive values\")\n    result = math.log(n, base)\n    self.history.append(f\"log({n}) = {result}\")\n    self.save_to_memory(result)  # method never defined\n    return result"),
        ("factorial", "def factorial(self, n):\n    if n < 0:\n        raise ValueError(\"Factorial undefined for negative numbers\")\n    if n == 0:\n        return 1\n    return n * self.factorial(n)  # should be n-1, causes infinite recursion")
    ]

    print("\n======================================")
    print("Evaluating TST Nodes: test_calculator.py")
    print("======================================\n")
    for name, content in py_nodes:
        print(f"--- Node: {name} ---")
        res = evaluate_node(model, tokenizer, device, name, content, "Python ScientificCalculator context")
        print(res)
        print("\n")

if __name__ == "__main__":
    simulate_tst_evaluation()
