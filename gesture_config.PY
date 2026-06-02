"""
gesture_config.py  —  Central Gesture Configuration
────────────────────────────────────────────────────
• PREDEFINED_GESTURES  : 10 fixed gestures (label → phrase)
• CUSTOM_GESTURES      : 10 user-nameable gestures (edit name + phrase freely)
• VOICE_PROFILES       : Available TTS voice options
• get_phrase(label)    : Returns the spoken/displayed phrase for any label
"""

# ──────────────────────────────────────────────────────────────────────────────
# 10 PREDEFINED GESTURES  (do NOT rename labels — model is trained on these)
# You CAN change the phrase/text that gets spoken.
# ──────────────────────────────────────────────────────────────────────────────
PREDEFINED_GESTURES = {
    "HELLO":    "Hello! Nice to meet you.",
    "HELP":     "I need help please.",
    "YES":      "YES PLEASE",
    "NO":       "NO SORRY",
    "THANKS":   "Thank you very much.",
    "SORRY":    "I am sorry.",
    "STOP":     "Please stop.",
    "COME":     "Please come here.",
    "PEACE":    "SELF INTRO",
    "LOVE":     "I love you.",
}

# ──────────────────────────────────────────────────────────────────────────────
# 10 CUSTOM GESTURES  (freely rename label AND phrase)
# ──────────────────────────────────────────────────────────────────────────────
CUSTOM_GESTURES = {
    "CUSTOM_1":  "Custom Phrase 1",
    "CUSTOM_2":  "Custom Phrase 2",
    "CUSTOM_3":  "Custom Phrase 3",
    "CUSTOM_4":  "Custom Phrase 4",
    "CUSTOM_5":  "Custom Phrase 5",
    "CUSTOM_6":  "Custom Phrase 6",
    "CUSTOM_7":  "Custom Phrase 7",
    "CUSTOM_8":  "Custom Phrase 8",
    "CUSTOM_9":  "Custom Phrase 9",
    "CUSTOM_10": "Custom Phrase 10",
}

# ──────────────────────────────────────────────────────────────────────────────
# VOICE PROFILES  (pyttsx3 property bundles)
# rate  : words per minute  (default ~200)
# volume: 0.0 – 1.0
# voice : "male" | "female" | index int (resolved at runtime)
# ──────────────────────────────────────────────────────────────────────────────
VOICE_PROFILES = {
    "Female": "en-US-AriaNeural",
    "Male": "en-US-GuyNeural",
}

DEFAULT_VOICE_PROFILE = "Female"

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
ALL_GESTURES = {**PREDEFINED_GESTURES, **CUSTOM_GESTURES}

def get_phrase(label: str) -> str:
    """Return the spoken phrase for a gesture label. Falls back to label itself."""
    return ALL_GESTURES.get(label, label)

def list_all_gestures():
    print("\n── PREDEFINED GESTURES ──────────────────────")
    for label, phrase in PREDEFINED_GESTURES.items():
        print(f"  {label:<12} → \"{phrase}\"")
    print("\n── CUSTOM GESTURES ──────────────────────────")
    for label, phrase in CUSTOM_GESTURES.items():
        print(f"  {label:<12} → \"{phrase}\"")
    print()

if __name__ == "__main__":
    list_all_gestures()