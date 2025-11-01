# agent_step1.py
def calculator(expression: str) -> str:
    """
    Simple safe calculator: evaluate arithmetic expressions only.
    Returns result as string or error message.
    """
    # allow only digits, spaces, and arithmetic operators
    import re
    if not re.fullmatch(r"[0-9\.\+\-\*\/\(\) \t]+", expression):
        return "Error: invalid characters in expression."
    try:
        # safe eval pattern
        result = eval(expression, {"__builtins__": None}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    print("Tool test — calculator")
    print("2 + 2 ->", calculator("2 + 2"))
    print("10*(3+2) ->", calculator("10*(3+2)"))
