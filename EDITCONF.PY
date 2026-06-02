"""
edit_config.py  —  Interactive Gesture Config Editor
Run this to rename custom gesture labels and change spoken phrases.
Changes are written directly to gesture_config.py.
"""

import re, os, sys

CONFIG_FILE = "gesture_config.py"

def load_config_text():
    with open(CONFIG_FILE, "r") as f:
        return f.read()

def save_config_text(text):
    with open(CONFIG_FILE, "w") as f:
        f.write(text)

def parse_dict_block(text, dict_name):
    """Extract key→value pairs from a named dict block in source."""
    pattern = rf'{dict_name}\s*=\s*\{{([^}}]+)\}}'
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None, None
    block = match.group(1)
    pairs = re.findall(r'"([^"]+)"\s*:\s*"([^"]+)"', block)
    return pairs, match

def update_dict_value(text, dict_name, old_key, new_key, new_value):
    """Replace a single key:value pair inside a named dict in source text."""
    pattern = rf'("{re.escape(old_key)}"\s*:\s*)"([^"]*)"'
    replacement = f'"{new_key}": "{new_value}"'
    new_text, count = re.subn(pattern, replacement, text, count=1)
    return new_text, count > 0

def print_section(title, pairs, color_code="36"):
    print(f"\n\033[{color_code}m{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}\033[0m")
    for i, (label, phrase) in enumerate(pairs, 1):
        print(f"  [{i:>2}]  \033[1m{label:<14}\033[0m → \"{phrase}\"")

def main():
    print("\n╔══════════════════════════════════════════════╗")
    print("║     Gesture Config Editor                    ║")
    print("╚══════════════════════════════════════════════╝")

    text = load_config_text()

    predefined_pairs, _ = parse_dict_block(text, "PREDEFINED_GESTURES")
    custom_pairs,     _ = parse_dict_block(text, "CUSTOM_GESTURES")

    if not predefined_pairs or not custom_pairs:
        print("[ERROR] Could not parse gesture_config.py. Make sure it's unmodified.")
        sys.exit(1)

    while True:
        print_section("PREDEFINED GESTURES  (phrase editable, label fixed)", predefined_pairs, "36")
        print_section("CUSTOM GESTURES  (label + phrase both editable)", custom_pairs, "33")

        print("\n\033[90mOptions:\033[0m")
        print("  p <n>  — Edit predefined gesture #n phrase")
        print("  c <n>  — Edit custom gesture #n (label + phrase)")
        print("  q      — Quit / save\n")

        cmd = input(">>> ").strip().lower()

        if cmd == "q":
            break

        parts = cmd.split()
        if len(parts) != 2 or parts[0] not in ("p", "c"):
            print("Invalid command.")
            continue

        mode, idx_str = parts
        try:
            idx = int(idx_str) - 1
        except ValueError:
            print("Invalid number.")
            continue

        if mode == "p":
            if idx < 0 or idx >= len(predefined_pairs):
                print("Number out of range.")
                continue
            old_label, old_phrase = predefined_pairs[idx]
            print(f"\n  Current: \"{old_label}\" → \"{old_phrase}\"")
            new_phrase = input(f"  New phrase (Enter to keep): ").strip()
            if not new_phrase:
                continue
            text, ok = update_dict_value(text, "PREDEFINED_GESTURES",
                                          old_label, old_label, new_phrase)
            if ok:
                predefined_pairs[idx] = (old_label, new_phrase)
                print(f"  \033[32m✔ Updated \"{old_label}\" → \"{new_phrase}\"\033[0m")
            else:
                print("  \033[31m✘ Update failed.\033[0m")

        elif mode == "c":
            if idx < 0 or idx >= len(custom_pairs):
                print("Number out of range.")
                continue
            old_label, old_phrase = custom_pairs[idx]
            print(f"\n  Current: \"{old_label}\" → \"{old_phrase}\"")
            new_label  = input(f"  New label  (Enter to keep '{old_label}'): ").strip().upper()
            new_phrase = input(f"  New phrase (Enter to keep): ").strip()
            if not new_label:  new_label  = old_label
            if not new_phrase: new_phrase = old_phrase

            text, ok = update_dict_value(text, "CUSTOM_GESTURES",
                                          old_label, new_label, new_phrase)
            if ok:
                custom_pairs[idx] = (new_label, new_phrase)
                print(f"  \033[32m✔ Updated → \"{new_label}\": \"{new_phrase}\"\033[0m")
            else:
                print("  \033[31m✘ Update failed.\033[0m")

    save_config_text(text)
    print("\n\033[32m[SAVED] gesture_config.py updated.\033[0m")
    print("Re-run collect_data.py with updated labels if you changed custom gesture names.\n")


if __name__ == "__main__":
    main()