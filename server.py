"""Remind test console: one process holds band 9, the page controls it.

    .venv/bin/python server.py          then open http://localhost:8790
    .venv/bin/python server.py --sim    pretend band
"""
import argparse
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import proto
from proto import Band, Memory, Watcher, connect_patiently, load_env

ROOT = Path(__file__).resolve().parent
PORT = 8790


class Hub:
    """Fan-out of events to every open page (Server-Sent Events)."""
    def __init__(self):
        self.subs, self.lock = [], threading.Lock()
        self.state = {"link": "idle", "band": "Vexo-HACK-09"}

    def push(self, kind, data):
        msg = json.dumps({"kind": kind, **data}, default=str)
        with self.lock:
            for q in list(self.subs):
                try:
                    q.put_nowait(msg)
                except queue.Full:
                    pass

    def status(self, **changes):
        self.state.update(changes)
        self.push("status", self.state)


hub = Hub()
memory = Memory()
commands = queue.Queue()
proto.LISTENERS.append(lambda kind, text, data: hub.push(kind, {"text": text, **data}))


def band_thread(args):
    """All band calls happen on this thread."""
    while True:
        cmd = commands.get()
        if cmd != "connect":
            if cmd == "reset":
                memory.log.clear()
                hub.push("reset", {})
            continue
        hub.status(link="looking")
        try:
            if args.sim:
                from vexo_open.transports.sim import SimTransport
                band = Band.connect(9, transport=SimTransport(9, auto_tap=False))
            else:
                band = connect_patiently(args, seconds=180)
        except Exception as ex:  # noqa: BLE001
            hub.status(link="idle")
            hub.push("error", {"text": f"Could not find the band: {ex}"})
            continue
        w = Watcher(band, memory, ROOT / "recordings")
        (ROOT / "recordings").mkdir(exist_ok=True)
        band.on("motion", w.on_motion)
        band.on("audio", w.on_audio)
        band.on("tap", w.on_tap)
        band.on("double-tap", lambda e: (proto.emit("tap", "  (double-tap event)"), w.on_double_tap(e)))
        lost = threading.Event()
        band.on("disconnect", lambda e: (lost.set(), hub.status(link="idle"),
                                         hub.push("error", {"text": "The band disconnected. Click Connect band, then double-tap the band."})))
        band.configure(motion=50, audio=True, events=True)
        band.set_events(tap_threshold=proto.TAP_THRESHOLD_G)
        try:
            proto.emit("diag", f"  (event settings: {band.get_events()})")
        except Exception as ex:  # noqa: BLE001
            proto.emit("diag", f"  (event settings failed: {ex})")
        band.on("event", lambda e: proto.emit("diag", f"  (band event: {e.kind} {getattr(e, 'arg', '')})"))
        hub.status(link="connected")
        try:
            b = band.get_battery()
            proto.emit("error", f"Battery: {b.percent}% ({b.state})")
            print(f"BATTERY {b}")
        except Exception as ex:  # noqa: BLE001
            print(f"BATTERY read failed: {ex}")
        last_meter = 0.0
        running = True
        while running and not lost.is_set():
            band.wait(0.05)
            w.check()
            w.check_question()
            w.show_answer_light()
            now = time.time()
            if now - last_meter > 0.15:          # live meters for the page
                last_meter = now
                gy = [abs(p[1]) for p in list(w.motion)[-8:]] or [0]
                lv = [d for _, d in list(w.loud)[-8:]] or [-96]
                hub.push("meter", {"turn": max(gy), "level": max(lv)})
            try:
                cmd = commands.get_nowait()
            except queue.Empty:
                continue
            if cmd == "ask":
                w.on_double_tap(None)
            elif cmd == "reset":
                memory.log.clear()
                hub.push("reset", {})
            elif cmd in ("disconnect", "connect"):
                running = False
                if cmd == "connect":
                    commands.put("connect")
        try:
            band.disconnect()
        except Exception:  # noqa: BLE001
            pass
        hub.status(link="idle")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body=b"", kind="text/plain; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, (ROOT / "ui" / "index.html").read_bytes(), "text/html; charset=utf-8")
        if self.path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            q = queue.Queue(maxsize=500)
            with hub.lock:
                hub.subs.append(q)
            try:
                snap = {"kind": "snapshot", "state": hub.state, "log": memory.log}
                self.wfile.write(f"data: {json.dumps(snap, default=str)}\n\n".encode())
                self.wfile.flush()
                while True:
                    try:
                        msg = q.get(timeout=15)
                    except queue.Empty:
                        msg = None
                    self.wfile.write((f"data: {msg}\n\n" if msg else ": ping\n\n").encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with hub.lock:
                    hub.subs.remove(q)
            return
        self._send(404, b"not found")

    def do_POST(self):
        origin = self.headers.get("Origin")
        if origin and origin not in (f"http://localhost:{PORT}", f"http://127.0.0.1:{PORT}"):
            return self._send(403, b"forbidden")
        cmd = self.path.strip("/")
        if cmd not in ("connect", "disconnect", "ask", "reset"):
            return self._send(404, b"unknown command")
        commands.put(cmd)
        self._send(204)


def main():
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--usb", action="store_true")
    args = ap.parse_args()
    args.band = 9
    threading.Thread(target=band_thread, args=(args,), daemon=True).start()
    print(f"Remind test console: http://localhost:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
