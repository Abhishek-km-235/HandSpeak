from flask import Flask, Response, jsonify, request, send_from_directory
import cv2, mediapipe as mp, numpy as np
import pickle, os, time
from collections import deque
import tensorflow as tf
import logging
import threading
import asyncio
import edge_tts
import pygame
import importlib
import gesture_config
from gesture_config import get_phrase

# Hide pygame welcome message
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
try:
    pygame.mixer.init()
except Exception as e:
    print("Audio init error:", e)

voice_lock = threading.Lock()
last_spoken_gesture = "—"
voice_news = []
CURRENT_VOICE = gesture_config.VOICE_PROFILES[gesture_config.DEFAULT_VOICE_PROFILE]
VOICE_ENABLED = True
IS_SPEAKING = False

def speak_async(text):
    global IS_SPEAKING
    IS_SPEAKING = True
    # Lock only for the duration of synthesis and loading to prevent file conflicts
    with voice_lock:
        try:
            async def _speak():
                try:
                    # Use a very specific filename
                    ts_ms = int(time.time() * 1000)
                    temp_file = os.path.join(os.getcwd(), f"voice_{ts_ms}.mp3")
                    
                    print(f"[Voice] Generating: {text} (Voice: {CURRENT_VOICE})")
                    communicate = edge_tts.Communicate(text, CURRENT_VOICE)
                    await communicate.save(temp_file)
                    
                    if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
                        # Stop any currently playing music before loading new one
                        pygame.mixer.music.stop()
                        pygame.mixer.music.load(temp_file)
                        pygame.mixer.music.play()
                        
                        while pygame.mixer.music.get_busy():
                            pygame.time.Clock().tick(10)
                        
                        pygame.mixer.music.unload()
                    
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                except Exception as e:
                    print(f"[Voice] Error: {e}")
                    
            asyncio.run(_speak())
        except Exception as e:
            print(f"[Voice] Threading Error: {e}")
        finally:
            IS_SPEAKING = False

def trigger_voice(gesture, phrase):
    global last_spoken_gesture, IS_SPEAKING
    
    # Don't speak if it's the same gesture we just spoke
    if gesture == last_spoken_gesture:
        return
        
    if not gesture or gesture in ["", "—", "?"]:
        return
        
    last_spoken_gesture = gesture
    voice_news.append({"gesture": gesture, "phrase": phrase})
    print(f"[System] Triggering output for: {gesture} -> {phrase}")
    
    # Stop current playback immediately in the main thread for responsiveness
    pygame.mixer.music.stop()
    
    # Start new synthesis thread only if voice output is enabled
    if VOICE_ENABLED:
        threading.Thread(target=speak_async, args=(phrase,), daemon=True).start()
    else:
        # If voice is muted/disabled, block detection for a brief 1.5s to let text display finish
        def fake_speaking_delay():
            global IS_SPEAKING
            IS_SPEAKING = True
            time.sleep(1.5)
            IS_SPEAKING = False
        threading.Thread(target=fake_speaking_delay, daemon=True).start()

def reset_voice_state():
    global last_spoken_gesture
    last_spoken_gesture = "—"
    print("[System] Voice state reset (ready for next gesture)")

from gesture_config import get_phrase

# Configuration
MODEL_PATH            = "hybrid_model.h5"
ENCODER_PATH          = "label_encoder.pkl"
IMG_SIZE              = 128
OFFSET, WRIST_EXTRA   = 60, 60
CONFIDENCE_THRESHOLD  = 0.55
SMOOTHING_WINDOW      = 5
INTRO_GESTURE         = "PEACE"

# Suppress flask logging to keep terminal clean
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__, static_folder='.', template_folder='.')

# In-memory store for custom gestures (resets to defaults in gesture_config.py on every script restart)
LIVE_CUSTOM_GESTURES = gesture_config.CUSTOM_GESTURES.copy()

class DetectionEngine:
    def __init__(self, user_name):
        self.user_name       = user_name
        self.running         = False
        self.pred_buffer     = deque(maxlen=SMOOTHING_WINDOW)
        self.current_gesture = "—"
        self.current_phrase  = ""
        self.current_conf    = 0.0
        self.frame_rgb       = None
        self.model_loaded    = False
        self.gesture_start_time = None
        self.last_predicted_raw_gesture = None
        self.gesture_triggered = False
        self.lost_frames_counter = 0

        if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
            self.model = tf.keras.models.load_model(MODEL_PATH)
            with open(ENCODER_PATH,"rb") as f: self.le=pickle.load(f)
            self.model_loaded = True
        else:
            print("[Warning] Model or Encoder file not found. Predictions will not work.")

        self.mpH    = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils
        self.hands  = self.mpH.Hands(max_num_hands=1,min_detection_confidence=0.7)
        self.cap    = None

    def phrase(self, label):
        if label == INTRO_GESTURE:
            return f"I am {self.user_name}."
        # Use live custom gestures first, then fall back to predefined
        return LIVE_CUSTOM_GESTURES.get(label) or gesture_config.PREDEFINED_GESTURES.get(label, label)

    def _prep(self, crop):
        w = np.ones((IMG_SIZE,IMG_SIZE,3),np.uint8)*255
        hC,wC = crop.shape[:2]
        if hC/wC>1:
            k=IMG_SIZE/hC; wCal=max(1,int(k*wC))
            r=cv2.resize(crop,(wCal,IMG_SIZE),cv2.INTER_AREA)
            g=(IMG_SIZE-wCal)//2; w[:,g:g+wCal]=r
        else:
            k=IMG_SIZE/wC; hCal=max(1,int(k*hC))
            r=cv2.resize(crop,(IMG_SIZE,hCal),cv2.INTER_AREA)
            g=(IMG_SIZE-hCal)//2; w[g:g+hCal,:]=r
        return w

    def start(self):
        if not self.running:
            self.cap=cv2.VideoCapture(0,cv2.CAP_DSHOW if os.name=="nt" else cv2.CAP_ANY)
            self.running=True

    def stop(self):
        self.running=False
        if self.cap: self.cap.release(); self.cap=None
        self.current_gesture="—"; self.current_phrase=""; self.current_conf=0.0
        self.frame_rgb=None

    def process_frame(self):
        global IS_SPEAKING
        if not self.running or not self.cap: return None
        ret,frame=self.cap.read()
        if not ret: return None
        frame=cv2.flip(frame,1)
        
        # If currently speaking/outputting, bypass landmarks and model predictions
        if IS_SPEAKING:
            self.current_gesture = "—"
            self.current_phrase = ""
            self.current_conf = 0.0
            self.last_predicted_raw_gesture = None
            self.gesture_triggered = False
            self.gesture_start_time = None
            self.frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            return self.frame_rgb
            
        fh,fw=frame.shape[:2]
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
        # Face Mesh drawing for visual effect removed as per request
        res=self.hands.process(rgb)
        if res.multi_hand_landmarks and self.model_loaded:
            for lms in res.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(frame,lms,self.mpH.HAND_CONNECTIONS,
                    self.mpDraw.DrawingSpec((0,230,255),2,4),
                    self.mpDraw.DrawingSpec((180,40,250),2))
                xl,yl,xs,ys=[],[],[],[]
                for lm in lms.landmark:
                    px,py=int(lm.x*fw),int(lm.y*fh)
                    xl.append(px);yl.append(py);xs.append(lm.x);ys.append(lm.y)
                
                x1=max(0,min(xl)-OFFSET); y1=max(0,min(yl)-OFFSET)
                x2=min(fw,max(xl)+OFFSET); y2=min(fh,max(yl)+OFFSET+WRIST_EXTRA)
                crop=frame[y1:y2,x1:x2]
                
                if crop.size==0: continue
                
                white=self._prep(crop)
                ii=np.expand_dims(white/255.0,0).astype(np.float32)
                li=np.array(xs+ys,np.float32).reshape(1,-1)
                
                preds=self.model.predict({"image":ii,"landmarks":li},verbose=0)[0]
                ci=int(np.argmax(preds)); self.current_conf=float(preds[ci])
                
                if self.current_conf>=CONFIDENCE_THRESHOLD:
                    raw=self.le.inverse_transform([ci])[0]
                    self.pred_buffer.append(raw)
                    s = max(set(self.pred_buffer),key=self.pred_buffer.count)
                    
                    self.current_gesture = s
                    self.current_phrase = self.phrase(s)
                    
                    # 1-second hold logic to trigger output
                    if s != self.last_predicted_raw_gesture:
                        self.gesture_start_time = time.time()
                        self.last_predicted_raw_gesture = s
                        self.gesture_triggered = False
                    elif not self.gesture_triggered:
                        elapsed = time.time() - self.gesture_start_time
                        if elapsed >= 1.0:
                            self.gesture_triggered = True
                            trigger_voice(self.current_gesture, self.current_phrase)
                else:
                    self.current_gesture="?"; self.current_phrase=""
                    self.last_predicted_raw_gesture = None
                    self.gesture_triggered = False
                    self.gesture_start_time = None
        else:
            self.current_gesture="—"; self.current_phrase=""
            self.last_predicted_raw_gesture = None
            self.gesture_triggered = False
            self.gesture_start_time = None

        self.frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        return self.frame_rgb

engine = None

@app.route('/')
def index():
    return send_from_directory('.', 'register.html')

@app.route('/main.html')
def main_html():
    return send_from_directory('.', 'main.html')

@app.route('/register.html')
def register_html():
    return send_from_directory('.', 'register.html')

@app.route('/info.html')
def info_html():
    return send_from_directory('.', 'info.html')

@app.route('/report.pdf')
def serve_report():
    return send_from_directory('.', 'report.pdf')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global engine
    data = request.json or {}
    user_name = data.get('name', 'User')
    if not engine:
        engine = DetectionEngine(user_name)
    elif engine.user_name != user_name:
        engine.user_name = user_name
    engine.start()
    return jsonify({"status": "started"})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global engine
    if engine:
        engine.stop()
    return jsonify({"status": "stopped"})

def gen_frames():
    global engine
    while True:
        if not engine or not engine.running:
            break
            
        frame_rgb = engine.process_frame()
        if frame_rgb is None:
            time.sleep(0.03)
            continue
            
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        ret, buffer = cv2.imencode('.jpg', frame_bgr)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_stats')
def get_stats():
    global engine, voice_news
    if not engine or not engine.running:
        return jsonify({"gesture": "—", "confidence": 0.0, "news": [], "triggered": False, "held_seconds": 0.0})
    
    # Take a copy and clear the news list
    current_news = voice_news[:]
    voice_news.clear()
    
    held_seconds = 0.0
    if engine.last_predicted_raw_gesture and engine.gesture_start_time:
        held_seconds = time.time() - engine.gesture_start_time
    
    return jsonify({
        "gesture": engine.current_gesture,
        "confidence": engine.current_conf,
        "news": current_news,
        "triggered": engine.gesture_triggered,
        "held_seconds": min(1.0, held_seconds) if engine.last_predicted_raw_gesture else 0.0,
        "is_speaking": IS_SPEAKING
    })

@app.route('/get_config')
def get_config():
    # Return predefined from file but CUSTOM from our live in-memory store
    return jsonify({
        "predefined": gesture_config.PREDEFINED_GESTURES,
        "custom": LIVE_CUSTOM_GESTURES
    })

@app.route('/trigger_output', methods=['POST'])
def trigger_output():
    global engine
    data = request.json or {}
    label = data.get('label')
    if not label:
        return jsonify({"status": "error", "message": "No label provided"}), 400
    
    # Get the phrase using the engine logic (which now uses LIVE_CUSTOM_GESTURES)
    phrase = engine.phrase(label) if engine else (LIVE_CUSTOM_GESTURES.get(label) or gesture_config.PREDEFINED_GESTURES.get(label, label))
    
    trigger_voice(label, phrase)
    return jsonify({"status": "success", "phrase": phrase})

@app.route('/reset_output', methods=['POST'])
def reset_output():
    reset_voice_state()
    return jsonify({"status": "reset"})

@app.route('/change_voice', methods=['POST'])
def change_voice():
    global CURRENT_VOICE
    data = request.json or {}
    voice_key = data.get('voice', 'Female')
    if voice_key in gesture_config.VOICE_PROFILES:
        CURRENT_VOICE = gesture_config.VOICE_PROFILES[voice_key]
        print(f"[System] Voice changed to: {voice_key} ({CURRENT_VOICE})")
        return jsonify({"status": "success", "voice": voice_key})
    return jsonify({"status": "error", "message": "Invalid voice profile"}), 400

@app.route('/toggle_voice', methods=['POST'])
def toggle_voice_route():
    global VOICE_ENABLED
    data = request.json or {}
    enabled = data.get('enabled', True)
    VOICE_ENABLED = enabled
    print(f"[System] Voice enabled set to: {VOICE_ENABLED}")
    return jsonify({"status": "success", "voice_enabled": VOICE_ENABLED})

@app.route('/save_gestures', methods=['POST'])
def save_gestures():
    global LIVE_CUSTOM_GESTURES
    try:
        new_custom = request.json
        if not new_custom:
            return jsonify({"status": "error", "message": "No data received"}), 400

        print(f"[System] Updating session gestures: {new_custom}")
        
        # We update the LIVE version in memory ONLY.
        # We NO LONGER write to gesture_config.py, so it resets on restart.
        LIVE_CUSTOM_GESTURES = new_custom
        
        return jsonify({"status": "success"})
    except Exception as e:
        print(f"[System] Error updating session gestures: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask web server...")
    print("Open http://127.0.0.1:8000/ in your browser.")
    app.run(host='0.0.0.0', port=8000, threaded=True, use_reloader=False)
