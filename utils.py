def input_yes_or_no(msg: str, default_answer: bool = False) -> bool:
    ret = default_answer
    print(msg, end="")
    while True:
        answer = input().lower()
        if answer in ("n", "no"):
            ret = False
            break
        if answer in ("y", "yes"):
            ret = True
            break
        print("Answer y[es] or n[o]: ", end="")
    return ret


def input_int(prompt: str, min_val: int, max_val: int, default: int) -> int:
    while True:
        user_input = input(
            f"{prompt} [{min_val}-{max_val}] (default={default}): "
        ).strip()

        # If user just hits Enter → return default
        if user_input == "":
            return default

        # Try converting to integer
        try:
            value = int(user_input)
        except ValueError:
            print("Please enter a valid integer.")
            continue

        # Check bounds
        if value < min_val or value > max_val:
            print(f"Please enter a number between {min_val} and {max_val}.")
            continue

        return value
