import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

import { DuplexSession } from "./session.js";

const appSource = readFileSync(new URL("./app.js", import.meta.url), "utf8").replace(/^import .+;$/gm, "");

// The page with a settings panel that answers the given session options; locks records what the page asked the panel to lock.
function createApp(sessionOptions) {
  const sockets = [];
  const locks = [];
  const elements = new Map();
  const context = vm.createContext({
    DuplexSession,
    window: { DEMO_WS_URL: "ws://localhost/v1/realtime", addEventListener() {} },
    location: { search: "" },
    URLSearchParams,
    performance: { now: () => 12345 },
    navigator: { mediaDevices: { getUserMedia: async () => { throw new Error("No microphone in playback test"); } } },
    document: {
      documentElement: { dataset: {} },
      addEventListener() {},
      getElementById: (id) => {
        if (!elements.has(id)) elements.set(id, { dataset: {}, addEventListener() {}, setAttribute() {} });
        return elements.get(id);
      },
    },
    matchMedia: () => ({ matches: false }),
    localStorage: { getItem() { return null; } },
    setTimeout() {},
    clearTimeout() {},
    setInterval() {},
    requestAnimationFrame() {},
    Orb: class {},
    Settings: class {
      async init() {}
      lock(locked) { locks.push(locked); }
      sessionOptions() { return sessionOptions; }
      micConstraint() { return {}; }
    },
    AudioContext: class {
      constructor({ sampleRate }) {
        this.sampleRate = sampleRate;
        this.audioWorklet = { addModule: async () => {} };
        this.destination = {};
      }
      async resume() {}
      async close() {}
      createAnalyser() { return {}; }
    },
    AudioWorkletNode: class {
      constructor() {
        this.port = { onmessage: null, postMessage() {} };
      }
      connect() { return this; }
    },
    WebSocket: class {
      static OPEN = 1;
      static CLOSING = 2;
      constructor() {
        this.readyState = 0;
        this.listeners = {};
        this.sent = [];
        sockets.push(this);
      }
      addEventListener(name, listener) { this.listeners[name] = listener; }
      send(text) { this.sent.push(JSON.parse(text)); }
      open() {
        this.readyState = 1;
        this.listeners.open();
        this.receive({ type: "session.created", session: { id: "test-session" } });
      }
      receive(event) { this.listeners.message({ data: JSON.stringify(event) }); }
      async close() { this.readyState = 3; await this.listeners.close({ code: 1000 }); }
    },
  });
  vm.runInContext(appSource, context);
  return {
    sockets,
    locks,
    connect: () => vm.runInContext("connect()", context),
    settled: () => vm.runInContext("eventChain", context),
  };
}

async function firstSessionUpdate(app) {
  await app.connect();
  const socket = app.sockets.at(-1);
  socket.open();
  await app.settled();
  return { socket, session: socket.sent.find((event) => event.type === "session.update").session };
}

test("the page sends only output_modalities while the panel is empty and the chosen settings otherwise, locked during the call", async () => {
  const empty = createApp({ instructions: "", outputModalities: ["audio"], extension: {}, turnDetection: null });
  const { socket, session } = await firstSessionUpdate(empty);
  assert.deepEqual(session, { output_modalities: ["audio"] });
  assert.equal(empty.locks.at(-1), true);
  await socket.close();
  assert.equal(empty.locks.at(-1), false);

  const filled = createApp({
    instructions: "Answer in one short English sentence.",
    outputModalities: ["audio"],
    extension: { sampling: { temperature: 0.2, max_new_tokens_per_unit: 128 } },
    turnDetection: { type: "semantic_vad", eagerness: "high" },
  });
  assert.deepEqual((await firstSessionUpdate(filled)).session, {
    output_modalities: ["audio"],
    instructions: "Answer in one short English sentence.",
    audio: { input: { turn_detection: { type: "semantic_vad", eagerness: "high" } } },
    sglang: { sampling: { temperature: 0.2, max_new_tokens_per_unit: 128 } },
  });
});
