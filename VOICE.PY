"""
voice_engine.py  —  Text-to-Speech Engine (Fixed)
──────────────────────────────────────────────────
Fix: pyttsx3 engine gets stuck after first speech when reused.
Solution: reinitialize a fresh engine for every single speak() call.
This is the most reliable cross-platform fix for this bug.
"""

import pyttsx3
import threading
from gesture_config import VOICE_PROFILES, DEFAULT_VOICE_PROFILE


class VoiceEngine:
    def __init__(self):
        self.current_profile = DEFAULT_VOICE_PROFILE
        self._speaking       = False
        self._lock           = threading.Lock()

        # Get voice list once at startup for profile matching
        _e = pyttsx3.init()
        self._voices = _e.getProperty("voices")
        _e.stop()
        del _e

        print(f"[VoiceEngine] Ready — {len(self._voices)} system voices found")

    # ── Pick voice id from profile ─────────────────────────────────────────
    def _pick_voice_id(self, profile_name):
        profile = VOICE_PROFILES.get(profile_name, VOICE_PROFILES[DEFAULT_VOICE_PROFILE])
        gender  = profile.get("voice", "female").lower()
        selected = None

        for v in self._voices:
            vn = v.name.lower()
            if gender == "female" and any(w in vn for w in ["zira","hazel","female","woman","samantha"]):
                selected = v.id; break
            elif gender == "male" and any(w in vn for w in ["david","mark","male","man","alex"]):
                selected = v.id; break

        if selected is None and self._voices:
            selected = self._voices[1].id if (gender == "female" and len(self._voices) > 1) else self._voices[0].id

        return selected, profile

    # ── Core speak (always fresh engine) ──────────────────────────────────
    def _do_speak(self, text, profile_name):
        """
        Creates a brand-new pyttsx3 engine, speaks, then destroys it.
        This is the ONLY reliable fix for pyttsx3 freezing after first use.
        """
        with self._lock:
            self._speaking = True
            try:
                engine = pyttsx3.init()
                voice_id, profile = self._pick_voice_id(profile_name)

                engine.setProperty("rate",   profile["rate"])
                engine.setProperty("volume", profile["volume"])
                if voice_id:
                    engine.setProperty("voice", voice_id)

                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as e:
                print(f"[VoiceEngine] Error: {e}")
            finally:
                self._speaking = False

    # ── Public API ─────────────────────────────────────────────────────────
    def speak(self, text: str):
        """Speak text in background thread. Skips if already speaking."""
        if self._speaking:
            return
        t = threading.Thread(
            target=self._do_speak,
            args=(text, self.current_profile),
            daemon=True
        )
        t.start()

    def set_profile(self, profile_name: str):
        """Switch voice profile."""
        if profile_name in VOICE_PROFILES:
            self.current_profile = profile_name
            print(f"[VoiceEngine] Profile → {profile_name}")
        else:
            print(f"[VoiceEngine] Unknown profile: {profile_name}")

    def list_profiles(self):
        return list(VOICE_PROFILES.keys())

    def list_system_voices(self):
        print("\n── System Voices ──────────────────────")
        for i, v in enumerate(self._voices):
            print(f"  [{i}] {v.name}")
        print()

    @property
    def is_speaking(self):
        return self._speaking


# Quick test
if __name__ == "__main__":
    ve = VoiceEngine()
    ve.list_system_voices()
    print("Testing Default...")
    ve.speak("Hello, this is the default voice.")
    import time; time.sleep(4)
    print("Testing Deep Male...")
    ve.set_profile("Deep Male")
    ve.speak("This is the deep male voice.")
    time.sleep(4)
    print("Testing Fast Female...")
    ve.set_profile("Fast Female")
    ve.speak("This is the fast female voice.")
    time.sleep(4)
    print("All done!")