import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "vexo-open-kit" / "python"))
from vexo_open.sync import Band
for i in range(12):
    try:
        band = Band.connect(9); break
    except Exception as ex:
        print("looking... double-tap the band", flush=True)
else:
    raise SystemExit("band not found")
print("connected; streams:", band.streams, flush=True)
band.configure(audio=True, events=True)
print("mic ON (light should be solid blue) - 3 s", flush=True); band.wait(3)
band.configure(audio=False)
print("mic OFF; streams:", band.streams, flush=True)
for name, colour in (("GREEN", "#00ff40"), ("RED", "#ff0000")):
    try:
        band.set_led(colour, brightness=1.0)
        print(name, "set; band reports:", band.get_led(), flush=True)
    except Exception as ex:
        print(name, "FAILED:", type(ex).__name__, ex, flush=True)
    band.wait(4)
band.led_off(); band.disconnect(); print("done", flush=True)
