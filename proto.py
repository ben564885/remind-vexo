"""Viability prototype: the band remembers what you DO.

Twist detector (gyro, local rules) -> cut audio around the twist -> Gemini labels the sound
-> rules log the action with its direction -> answer "Did I lock the door?".

    python proto.py --sim          scripted pretend band (lock, unlock, twist with no sound)
    python proto.py 9              real band 9 (double-tap it first)
"""
import argparse, base64, collections, datetime, json, os, sys, threading, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vexo-open-kit" / "python"))
from vexo_open.sync import Band
from vexo_open.audio import PcmDecoder, level_db, ogg_packets, write_wav

# --- settings (tune on the real band) ---
AXIS = os.environ.get("TWIST_AXIS", "gy")      # forearm axis on the real band: measure it
LOCK_SIGN = int(os.environ.get("LOCK_SIGN", "-1"))  # clockwise key turn reads negative on gy  # +1: positive rotation = locking
MIN_ANGLE = float(os.environ.get("MIN_ANGLE", "55"))   # one-way degrees inside WINDOW (real cap turns: 20-33 each)
MIN_RATE = 120.0       # peak degrees/second
WINDOW = float(os.environ.get("WINDOW", "3.0"))   # seconds: a cap opens in several short same-way turns
COOLDOWN = 2.5         # seconds between twists
CLIP_BEFORE, CLIP_AFTER = 4.0, 1.0
MODEL = os.environ.get("GEMINI_MODEL", "gemini-4-hydrogen")
LISTENERS = []   # the web UI subscribes here; the terminal just prints


def emit(kind, text, **data):
    print(text, flush=True)
    for fn in list(LISTENERS):
        try:
            fn(kind, text, data)
        except Exception:  # noqa: BLE001
            pass
MIN_ONSET_DB = 15.0
MIN_ONSET_TWIST_DB = 5.0   # a real twist already proves something happened: let Gemini judge quiet keys    # loudest 100 ms must be this far above the clip's background
SOUND_TRIGGER_DB = float(os.environ.get("SOUND_TRIGGER_DB", "18"))  # burst above room level starts a check
ASK_SECONDS = 3.5      # how long to listen after a double-tap
LED_COLOURS = {"yes": "#00ff40", "no": "#ff0000", "unsure": "#ffb000"}
LED_SECONDS = 5.0
TAP_THRESHOLD_G = float(os.environ.get("TAP_G", "0.65"))   # default 0.8 missed the second tap
PILL_MIN_TURN = 30.0  # degrees one way: a real cap twist
STILL_DPS = 150.0   # a real 'ask' double-tap happens with a still wrist
KEY_RETURN_SECONDS = 4.0
KEY_TURN_DEG = 65.0   # a key turn and its turn back are each at least this big   # an opposite turn this soon after a lock turn = key coming back out


def sound_onset_db(pcm):
    """Loudest 100 ms block minus the median block, in dB. Silence or steady noise ~ 0."""
    import numpy as np
    a = np.frombuffer(pcm, np.int16).astype(float)
    if len(a) < 3200:
        return 0.0
    db = [20 * np.log10(np.sqrt(np.mean(b ** 2)) + 1) for b in np.array_split(a, len(a) // 1600)]
    return float(max(db) - np.median(db))


def load_env():
    env = ROOT / ".env"
    for line in env.read_text().splitlines() if env.exists() else []:
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


PROMPT = ("Audio from a microphone on a person's wrist, 16 kHz, cut around a moment when the wrist "
          "twisted. Classify the everyday action. lock = keys jingling on a key ring, a key going "
          "into a lock or turning, or a deadbolt or latch clicking. pills = pills, vitamins or "
          "tablets rattling in a bottle, or a pill bottle cap clicking or opening. A dull, hollow, plastic rattle of many small objects is pills; a bright metallic jingle or click is lock. drink = sipping or "
          "swallowing. none = anything else (speech, room noise, other sounds). Judge ONLY the physical sounds: ignore what people say, even if they talk about locks or pills.")


def deepgram_text(wav_path):
    req = urllib.request.Request(
        "https://api.deepgram.com/v1/listen?smart_format=true&model=nova-3",
        data=Path(wav_path).read_bytes(),
        headers={"Authorization": "Token " + os.environ["DEEPGRAM_API_KEY"], "Content-Type": "audio/wav"})
    r = json.load(urllib.request.urlopen(req, timeout=30))
    return r["results"]["channels"][0]["alternatives"][0]["transcript"]


def speak(text):
    """ElevenLabs voice through the laptop speaker. Falls back to macOS say."""
    out = ROOT / "recordings" / "answer.mp3"
    try:
        req = urllib.request.Request(
            "https://api.elevenlabs.io/v1/text-to-speech/EXAVITQu4vr4xnSDxMaL?output_format=mp3_44100_128",
            data=json.dumps({"text": text, "model_id": "eleven_flash_v2_5"}).encode(),
            headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"], "Content-Type": "application/json"})
        out.write_bytes(urllib.request.urlopen(req, timeout=20).read())
        os.system(f"afplay '{out}' &")
    except Exception as ex:  # noqa: BLE001
        print("  ElevenLabs failed, using say:", ex)
        os.system(f"say {json.dumps(text)} &")


def gemini_label(wav_path):
    body = {"contents": [{"parts": [
        {"inline_data": {"mime_type": "audio/wav", "data": base64.b64encode(Path(wav_path).read_bytes()).decode()}},
        {"text": PROMPT}]}],
        "generationConfig": {"responseMimeType": "application/json", "responseSchema": {"type": "OBJECT", "properties": {
            "action": {"type": "STRING", "enum": ["lock", "pills", "drink", "none"]},
            "confidence": {"type": "NUMBER"}, "heard": {"type": "STRING"}},
            "required": ["action", "confidence", "heard"]}}}
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "x-goog-api-key": os.environ["GEMINI_API_KEY"]})
    r = json.load(urllib.request.urlopen(req, timeout=30))
    return json.loads(r["candidates"][0]["content"]["parts"][0]["text"])


class Memory:
    """Rules own the facts. The LLM only labels sounds."""
    def __init__(self):
        self.log, self.lock = [], threading.Lock()

    def add(self, entry):
        with self.lock:
            self.log.append(entry)
        hhmm = datetime.datetime.fromtimestamp(entry["time"]).strftime("%H:%M:%S")
        emit("log", f"  LOG {hhmm}  {entry['label']:<9} (sound={entry['action']} {entry['confidence']:.2f}, "
                    f"turn={entry['angle']:+.0f} deg, heard: {entry['heard'][:60]})", entry=entry)

    def pills_answer(self):
        with self.lock:
            today = datetime.date.today()
            took = [e for e in self.log if e["label"] == "pills"
                    and datetime.date.fromtimestamp(e["time"]) == today]
        if not took:
            return "No. I have not seen you take your pills today."
        when = datetime.datetime.fromtimestamp(took[-1]["time"]).strftime("%-I:%M %p")
        return f"Yes. You took your pills at {when}. I heard the bottle and felt the cap twist."

    def answer(self, question):
        """Rules route the question. No LLM decides the answer."""
        q = question.lower()
        if any(w in q for w in ("lock", "door", "key")):
            return self.door_answer()
        if any(w in q for w in ("pill", "med", "vitamin", "tablet", "advil")):
            return self.pills_answer()
        return "I can tell you about your pills or your door. Try: did I take my pills?"

    def door_answer(self):
        with self.lock:
            locks = [e for e in self.log if e["label"] in ("locked", "unlocked")]
        if not locks:
            return "No. I have not seen you lock the door today."
        last = locks[-1]
        when = datetime.datetime.fromtimestamp(last["time"]).strftime("%-I:%M %p")
        if last["label"] == "locked":
            return f"Yes. You locked the door at {when}. I heard the lock and felt your wrist turn."
        return f"No. The last thing I saw was you unlocking the door at {when}."


class Watcher:
    def __init__(self, band, memory, clips):
        self.band, self.memory, self.clips = band, memory, clips
        self.motion = collections.deque(maxlen=750)      # (time, rate on AXIS)
        self.audio = collections.deque(maxlen=50 * 8)    # (time, 20 ms pcm)
        self.loud = collections.deque(maxlen=50 * 5)     # (time, dB of 20 ms)
        self.decoder = PcmDecoder()
        self.last_twist = 0.0
        self.pending = []
        self.ask_at = None          # set by a double-tap: listen for a question
        self.led_request = None     # answer colour waiting to be shown
        self.last_ask_tap = 0.0
        self.twists = []            # (time, angle) of every twist trigger, for the key turn-and-back pattern

    def on_tap(self, e):
        now = time.time()
        emit("tap", "  (tap)", )
        self._last_tap = now          # single taps are only logged; the band's double-tap asks

    def on_double_tap(self, e):
        recent = [abs(p[1]) for p in list(self.motion) if p[0] >= time.time() - 1.0]
        if e is not None and recent and max(recent) > STILL_DPS:
            emit("diag", f"  (double-tap ignored: wrist moving {max(recent):.0f} deg/s)")
            return
        self.last_ask_tap = time.time()
        if self.ask_at is None and self.led_request is None:
            self.ask_at = time.time()
            emit("listening", "DOUBLE-TAP: listening for a question...")

    def check_question(self):
        if self.ask_at is None or time.time() < self.ask_at + ASK_SECONDS:
            return
        t, self.ask_at = self.ask_at, None
        pcm = b"".join(p for ts, p in list(self.audio) if t <= ts <= t + ASK_SECONDS)
        path = self.clips / f"question-{int(t)}.wav"
        write_wav(path, pcm)
        threading.Thread(target=self.answer_question, args=(path,), daemon=True).start()

    def answer_question(self, path):
        try:
            question = deepgram_text(path)
        except Exception as ex:  # noqa: BLE001
            emit("error", f"  Deepgram failed: {ex}")
            return
        if not question.strip():
            emit("answer", "ASK (nothing heard) -> ignored", question="", reply="", verdict="none")
            return
        if not any(w in question.lower() for w in ("pill", "med", "vitamin", "tablet", "advil", "lock", "door", "key")):
            emit("answer", f'ASK "{question}" -> not a question for me, ignored', question=question, reply="", verdict="none")
            return
        reply = self.memory.answer(question)
        verdict = "yes" if reply.startswith("Yes") else "no" if reply.startswith("No") else "unsure"
        emit("answer", f'ASK "{question}" -> {reply}', question=question, reply=reply, verdict=verdict)
        self.led_request = LED_COLOURS[verdict]     # light first: shown by the main loop (one band thread)
        speak(reply)

    def show_answer_light(self):
        """The mic locks the LED, so: mic off, show green/red, mic back on."""
        colour, self.led_request = self.led_request, None
        if colour is None:
            return
        try:
            self.band.configure(audio=False)
            self.band.set_led(colour, brightness=1.0, pattern="blink", period=500)
            emit("light", f"  LIGHT {colour} blinking for {LED_SECONDS:.0f} s", colour=colour)
            self.band.wait(LED_SECONDS)              # motion keeps arriving during this time
            self.band.led_off()
        except Exception as ex:  # noqa: BLE001
            emit("error", f"  light failed: {ex}")
        finally:
            self.decoder = PcmDecoder()              # new decoder after the audio stream restarts
            self.band.configure(audio=True)
            self.band.configure(events=False)        # force the band to restart tap events:
            self.band.configure(events=True)         # after the light cycle they stop otherwise
            self.band.set_events(tap_threshold=TAP_THRESHOLD_G)

    def on_motion(self, e):
        others = [abs(getattr(e, k) or 0.0) for k in ("gx", "gy", "gz") if k != AXIS]
        self.motion.append((e.time, getattr(e, AXIS) or 0.0, max(others)))

    def on_audio(self, e):
        pcm = self.decoder.feed(e.opus, e.seq)
        if pcm:
            self.audio.append((e.time, pcm))
            self.loud.append((e.time, level_db(pcm)))

    def sound_trigger(self, now):
        """A loud burst (a pill shake, keys) well above the room's recent level."""
        lv = [x for x in list(self.loud) if x[0] >= now - 4.0]
        if len(lv) < 50:
            return False
        recent = [d for ts, d in lv if ts >= now - 0.3]
        floor = sorted(d for _, d in lv)[len(lv) // 2]
        return bool(recent) and max(recent) - floor >= SOUND_TRIGGER_DB

    def check(self):
        """Called from the main loop: look for a twist, then process clips that are due."""
        now = time.time()
        pts = [p for p in list(self.motion) if p[0] >= max(now - WINDOW, self.last_twist)]
        if len(pts) > 5 and now - self.last_twist > COOLDOWN:
            # One-way sums: re-grips go the other way and must not cancel the real turns.
            pos = neg = 0.0
            for i in range(1, len(pts)):
                r, dt = pts[i][1], pts[i][0] - pts[i - 1][0]
                if r > 40:
                    pos += r * dt
                elif r < -40:
                    neg += r * dt
            angle = pos if pos >= -neg else neg
            peak = max(abs(p[1]) for p in pts)
            off_axis = sum(p[2] * (pts[i][0] - pts[i - 1][0]) for i, p in enumerate(pts) if i)
            if os.environ.get("DEBUG") and abs(angle) >= 20 and now - getattr(self, "_dbg", 0) > 0.6:
                self._dbg = now
                print(f"  (motion: {AXIS} {angle:+.0f} deg, peak {peak:.0f} deg/s, off-axis {off_axis:.0f} deg)")
            if abs(angle) >= MIN_ANGLE and peak >= MIN_RATE and abs(angle) > 0.6 * off_axis:
                self.last_twist = now
                self.twists.append((now, angle))
                emit("trigger", f"TWIST {angle:+.0f} deg (peak {peak:.0f} deg/s)", by="twist", angle=angle)
                self.pending.append((now, angle))
            elif self.ask_at is None and self.sound_trigger(now):
                self.last_twist = now
                emit("trigger", f"SOUND burst (turn so far {angle:+.0f} deg)", by="sound", angle=angle)
                self.pending.append((now + 1.0, angle))   # cut the clip a little later: the shake goes on
        for item in list(self.pending):
            t, angle = item
            if now >= t + CLIP_AFTER:
                self.pending.remove(item)
                pcm = b"".join(p for ts, p in list(self.audio) if t - CLIP_BEFORE <= ts <= t + CLIP_AFTER)
                path = self.clips / f"twist-{int(t)}.wav"
                write_wav(path, pcm)
                onset = sound_onset_db(pcm)
                if abs(angle) < KEY_TURN_DEG and onset < (MIN_ONSET_TWIST_DB if abs(angle) >= MIN_ANGLE else MIN_ONSET_DB):
                    self.memory.add(dict(time=t, angle=angle, label="ignored", action="silent",
                                         confidence=1.0, heard=f"no sound (onset {onset:.0f} dB)", latency=0.0))
                    continue
                threading.Thread(target=self.label, args=(t, angle, path, onset), daemon=True).start()

    def twist_between(self, t0, t1):
        """Largest one-way turn on AXIS between t0 and t1 (re-grips do not cancel)."""
        pts = [p for p in list(self.motion) if t0 <= p[0] <= t1]
        pos = neg = 0.0
        for i in range(1, len(pts)):
            r, dt = pts[i][1], pts[i][0] - pts[i - 1][0]
            if r > 40:
                pos += r * dt
            elif r < -40:
                neg += r * dt
        return pos if pos >= -neg else neg

    def key_pattern(self, t, angle):
        """Key in a lock = one big turn, then a big turn back within a few seconds to pull the key out.
        Returns 'turn' (this is the lock action), 'back' (this is the return), or None."""
        if abs(angle) < KEY_TURN_DEG:
            return None
        for ts, a in self.twists:
            if a * angle < 0 and abs(a) >= KEY_TURN_DEG:
                if 0 < ts - t <= KEY_RETURN_SECONDS:
                    return "turn"
                if 0 < t - ts <= KEY_RETURN_SECONDS:
                    return "back"
        return None

    def near_key_pattern(self, t):
        return any(self.key_pattern(ts, a) for ts, a in self.twists if abs(ts - t) <= KEY_RETURN_SECONDS + 1)

    def label(self, t, angle, path, onset=99.0):
        start = time.time()
        if abs(angle) >= KEY_TURN_DEG:                      # wait to see if a turn back follows
            time.sleep(max(0.0, t + KEY_RETURN_SECONDS + 0.3 - time.time()))
            pattern = self.key_pattern(t, angle)
            if pattern:
                heard = "key turn, then turn back to pull the key out (motion pattern)"
                label = "key-back" if pattern == "back" else "locked"   # demo: every key turn is a lock
                self.memory.add(dict(time=t, angle=angle, label=label, action="lock", confidence=1.0,
                                     heard=heard, latency=time.time() - start))
                return
            if onset < MIN_ONSET_TWIST_DB:
                self.memory.add(dict(time=t, angle=angle, label="ignored", action="silent", confidence=1.0,
                                     heard=f"big turn, no turn back, no sound (onset {onset:.0f} dB)", latency=0.0))
                return
        if abs(t - self.last_ask_tap) < CLIP_BEFORE + 1.0:     # the ask double-tap itself makes noise
            self.memory.add(dict(time=t, angle=angle, label="ignored", action="tap-noise", confidence=1.0,
                                 heard="sound next to an ask double-tap", latency=0.0))
            return
        try:
            out = gemini_label(path)
        except Exception as ex:  # noqa: BLE001
            emit("error", f"  Gemini failed: {ex}")
            return
        action, conf = out["action"], float(out["confidence"])
        prev = [e for e in self.memory.log if e["label"] in ("locked", "unlocked")]
        is_return = bool(prev) and t - prev[-1]["time"] < KEY_RETURN_SECONDS and prev[-1]["angle"] * angle < 0
        if action == "lock":
            label = "ignored"      # the lock is decided by the key turn-and-back motion, not by sound alone
            out["heard"] = "lock sound without a key turn-and-back: " + out["heard"]
        elif action == "pills" and conf >= 0.6 and self.near_key_pattern(t):
            label = "ignored"      # keys rattling in the cup during a key turn are not pills
            out["heard"] = "rattle during a key turn: " + out["heard"]
        elif action == "pills" and conf >= 0.6:
            turn = self.twist_between(t - CLIP_BEFORE - 2.0, time.time())   # sound AND motion, both required
            label = "pills" if abs(turn) >= PILL_MIN_TURN else "ignored"
            if label == "ignored":
                out["heard"] = f"rattle but no cap twist ({turn:+.0f} deg): " + out["heard"]
            angle = turn
        elif action == "drink" and conf >= 0.6:
            label = action
        else:
            label = "ignored"
        self.memory.add(dict(time=t, angle=angle, label=label, action=action, confidence=conf,
                             heard=out["heard"], latency=time.time() - start))


def twist(sim, degrees, seconds=0.45):
    rate = degrees / seconds
    sim.play_motion(lambda s: {AXIS: rate}, seconds=seconds)


def connect_patiently(args, seconds=120):
    """Keep looking for the band until someone double-taps it (or the time runs out)."""
    deadline = time.time() + seconds
    print("Looking for the band: double-tap it now (it blinks blue).")
    while True:
        try:
            return Band.connect(args.band, transport=("serial" if args.usb else "ble"), auto_reconnect=True)
        except Exception as ex:  # noqa: BLE001
            if time.time() > deadline:
                raise
            emit("status", "  not visible yet, still looking...", state="looking")


def calibrate(args):
    """Turn the key (or the cap) a few times. Each burst of rotation prints its angle on gx, gy, gz.
    The axis with the big number is TWIST_AXIS; its sign when you LOCK is LOCK_SIGN."""
    if args.sim:
        from vexo_open.transports.sim import SimTransport
        band = Band.connect(args.band, transport=SimTransport(args.band, auto_tap=False))
        band.transport.after(1.0, lambda: twist(band.transport, +90))
    else:
        band = Band.connect(args.band, transport=("serial" if args.usb else "ble"), auto_reconnect=True)
    print("Connected to", band.name, "- turn the key now, one turn at a time. Ctrl+C to stop.")
    acc, last, moving = {"gx": 0.0, "gy": 0.0, "gz": 0.0}, [None], [False]

    def on_motion(e):
        dt = 0.0 if last[0] is None else e.time - last[0]
        last[0] = e.time
        rate = max(abs(e.gx or 0), abs(e.gy or 0), abs(e.gz or 0))
        if rate > 60:
            moving[0] = True
            for k in acc:
                acc[k] += (getattr(e, k) or 0.0) * dt
        elif moving[0]:
            moving[0] = False
            if max(abs(v) for v in acc.values()) > 30:
                print("  turn: " + "  ".join(f"{k} {v:+5.0f} deg" for k, v in acc.items()))
            for k in acc:
                acc[k] = 0.0
    band.on("motion", on_motion)
    band.configure(motion=50)
    try:
        band.wait(5 if args.sim else args.seconds)
    except KeyboardInterrupt:
        pass
    band.disconnect()


def main():
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("band", nargs="?", type=int, default=9)
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--demo", action="store_true", help="with --sim: pills + lock + spoken questions")
    ap.add_argument("--usb", action="store_true", help="use the USB-C cable instead of Bluetooth")
    ap.add_argument("--calibrate", action="store_true", help="print the turn angle on each gyro axis")
    args = ap.parse_args()
    if args.calibrate:
        return calibrate(args)

    clips = ROOT / "recordings"
    clips.mkdir(exist_ok=True)
    memory = Memory()
    print("Loading the sound decoder...")
    if args.sim:
        from vexo_open.transports.sim import SimTransport
        band = Band.connect(args.band, transport=SimTransport(args.band, auto_tap=False))
    else:
        band = connect_patiently(args)
    print("Connected to", band.name)
    w = Watcher(band, memory, clips)
    band.on("motion", w.on_motion)
    band.on("audio", w.on_audio)
    band.on("double-tap", w.on_double_tap)
    band.configure(motion=50, audio=True, events=True)

    if args.sim and args.demo:
        sim = band.transport
        pills = ogg_packets(ROOT / "fixtures" / "pills1.ogg")
        ask_pills = ogg_packets(ROOT / "fixtures" / "ask_pills.ogg")
        ask_lock = ogg_packets(ROOT / "fixtures" / "ask_lock.ogg")
        lock = ogg_packets(ROOT / "fixtures" / "lock1.ogg")

        def ask(packets):
            sim.tap("double-tap")
            sim.after(0.3, lambda: sim.play_loudness(0.7, seconds=2, packets=packets))
        sim.after(2.0, lambda: ask(ask_pills))                                  # before: "No"
        sim.after(8.0, lambda: (twist(sim, +90), sim.play_loudness(0.7, seconds=2, packets=pills)))
        sim.after(14.0, lambda: ask(ask_pills))                                 # after: "Yes"
        sim.after(20.0, lambda: (twist(sim, +90), sim.play_loudness(0.7, seconds=2, packets=lock)))
        sim.after(23.5, lambda: (twist(sim, -90), sim.play_loudness(0.7, seconds=2, packets=lock)))  # key back, with jingle
        sim.after(27.0, lambda: ask(ask_lock))                                  # "Yes, locked"
        checkpoints, end = {}, 35.0
    elif args.sim:
        sim = band.transport
        lock = ogg_packets(ROOT / "fixtures" / "lock1.ogg")
        # 1) lock the door: clockwise turn + lock sound
        sim.after(2.0, lambda: (twist(sim, +90), sim.play_loudness(0.7, seconds=2, packets=lock)))
        # 2) unlock it: opposite turn + the same lock sound
        sim.after(7.0, lambda: (twist(sim, -90), sim.play_loudness(0.7, seconds=2, packets=lock)))
        # 3) twist with silence (opening a jar, turning a page): must be ignored
        sim.after(12.0, lambda: twist(sim, +90))
        # 4) lock again
        sim.after(17.0, lambda: (twist(sim, +90), sim.play_loudness(0.7, seconds=2, packets=lock)))
        checkpoints = {6.9: "after lock", 11.9: "after unlock", 16.9: "after silent twist",
                       24.0: "after lock again"}
        end = 24.5
    else:
        checkpoints, end = {}, args.seconds

    t0 = time.time()
    asked = set()
    while time.time() - t0 < end:
        band.wait(0.05)
        w.check()
        w.check_question()
        w.show_answer_light()
        for at, name in checkpoints.items():
            if time.time() - t0 >= at and at not in asked:
                asked.add(at)
                print(f'ASK ({name}) "Did I lock the door?" -> {memory.door_answer()}')
    band.disconnect()
    lat = [e["latency"] for e in memory.log]
    if lat:
        print(f"Gemini latency: avg {sum(lat)/len(lat):.1f}s, max {max(lat):.1f}s")


if __name__ == "__main__":
    main()
