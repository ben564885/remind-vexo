# Remind

**Vexo remembers what you say. Remind remembers what you do.**

"Did I lock the door?" "Did I take my pills?" Everyone asks these every day, and an
always-listening wearable can't answer them, because nobody *says* "I'm locking the door."

Remind runs on the [Vexo](https://github.com/VexoAI-Org/vexo-open-kit) band and fuses its two
sensors. The **microphone** hears *what* happened (pills rattling in a bottle). The **gyroscope**
feels *how your wrist moved* (a cap twist, a key turn and the turn back to pull it out). Double-tap
the band, ask out loud, and it answers by voice and with a green or red light.

Built at VexoHacks, Nov 14, 2026.

## How it works

```
band 9 ──Bluetooth──> laptop
  gyroscope 50 Hz ──> twist detector (one-way turn sums, so re-grips don't cancel)
  microphone (Opus) ──> 8 s ring buffer ──> loud-burst detector
        on a twist or a burst: cut the clip around it
        rules first: no real sound → no model call
        Gemini 4 (gemini-4-hydrogen) labels the sound: pills / lock / drink / none
        rules decide the fact:
          pills  = pill rattle  AND a real cap twist (≥30°), never during a key turn
          locked = a key turn (≥65°) followed by a turn back within 4 s (motion pattern)
double-tap ──> Deepgram (speech → text) ──> rules look up the log ──> answer
          ──> band light: green = yes, red = no  +  ElevenLabs voice
```

**Rules decide, the model only labels.** Gemini never decides that you took a pill: it says what
it heard, and deterministic rules combine that with the motion. Two lessons from real testing
shaped this:

- Gemini once "heard" a lock in a silent clip, so a clip with no real sound never reaches the model.
- A double-tap on the band's plastic case sounds like a pill rattle, so pills need the rattle *and*
  a cap twist, and sounds next to an "ask" double-tap are ignored.
- In a loud room the keys are too quiet to hear reliably, but a key turn has an unmistakable
  motion signature (a big turn, then a turn back to pull the key out). So the lock is decided by
  motion, and pills by sound.

## Run it

You need band 9 (or any band: change `BAND`), Python 3.9+, Chrome, and three keys.

```bash
git clone https://github.com/VexoAI-Org/vexo-open-kit.git   # the band library, next to this code
python3 -m venv .venv && . .venv/bin/activate
pip install bleak av numpy
cp env.example .env        # then fill in the keys; .env is gitignored
python server.py           # open http://localhost:8790, click Connect band, double-tap the band
```

Other modes:

| Command | What it does |
|---|---|
| `python server.py --sim` | Test console with a pretend band (no hardware) |
| `python proto.py --sim --demo` | Scripted demo in the terminal: questions, pills, lock |
| `python proto.py 9 --calibrate` | Prints the turn angle on each gyro axis, to find the twist axis |
| `python proto.py 9 --usb` | Use the USB-C cable instead of Bluetooth |

## Files

| File | What it is |
|---|---|
| `proto.py` | Detection: twist and sound triggers, Gemini labelling, rules, memory, Deepgram, ElevenLabs, band light |
| `server.py` | Test console server: holds the band, streams events to the page (Server-Sent Events) |
| `ui/index.html` | Test console page |
| `fixtures/` | Test sounds in band format (Opus): lock and pill-bottle recordings from Wikimedia Commons, spoken test questions |
| `ledtest.py` | Checks the band's light |

## Built with

Vexo Open band · Gemini 4 (sound understanding) · Deepgram (speech to text) · ElevenLabs (voice)

Team: Ben, Philip, Daniel, Sebastian.
