import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

import { DuplexSession } from "./session.js";

const appSource = readFileSync(new URL("./app.js", import.meta.url), "utf8").replace(/^import .+;$/gm, "");

// elements receives the page's elements by id, so a test can read a control's state or label; microphone grants a fake
// microphone whose capture node a test feeds frames; the clock is performance.now, advanced by advance(ms).
function createApp({ actualSampleRate, loadModule = async () => {}, sessionOptions = {}, elements = new Map(), microphone = false } = {}) {
  const nodes = [];
  const sockets = [];
  const clock = { now: 12345 };
  let microphoneRequests = 0;
  const context = vm.createContext({
    DuplexSession,
    window: { DEMO_WS_URL: "ws://localhost/v1/realtime", addEventListener() {} },
    location: { search: "" },
    URLSearchParams,
    performance: { now: () => clock.now },
    navigator: { mediaDevices: { getUserMedia: async () => {
      microphoneRequests += 1;
      if (!microphone) throw new Error("No microphone in playback test");
      return { getTracks: () => [] };
    } } },
    document: {
      documentElement: { dataset: {} },
      addEventListener() {},
      getElementById: (id) => {
        if (!elements.has(id)) elements.set(id, { dataset: {}, attributes: {}, listeners: {}, addEventListener(name, listener) { this.listeners[name] = listener; }, setAttribute(name, value) { this.attributes[name] = value; } });
        return elements.get(id);
      },
    },
    matchMedia: () => ({ matches: false }),
    localStorage: { getItem() { return null; } },
    setTimeout() {},
    clearTimeout() {},
    setInterval() {},
    requestAnimationFrame() {},
    Camera: class { stop() {} },
    Orb: class {},
    Settings: class {
      async init() {}
      lock() {}
      sessionOptions() { return sessionOptions; }
      micConstraint() { return {}; }
      async refreshDevices() {}
    },
    AudioContext: class {
      constructor({ sampleRate }) {
        this.sampleRate = actualSampleRate || sampleRate;
        this.audioWorklet = { addModule: () => loadModule(sampleRate) };
        this.destination = {};
      }
      async resume() {}
      async close() {}
      createAnalyser() { return {}; }
      createMediaStreamSource() { return { connect: (node) => node }; }
      createGain() { return { gain: {} }; }
    },
    AudioWorkletNode: class {
      constructor(audioContext, name) {
        this.context = audioContext;
        this.name = name;
        this.messages = [];
        this.port = { onmessage: null, postMessage: (message) => this.messages.push(message) };
        nodes.push(this);
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
    run: (code) => vm.runInContext(code, context),
    advance: (milliseconds) => { clock.now += milliseconds; },
    nodes,
    sockets,
    get microphoneRequests() { return microphoneRequests; },
    get session() { return vm.runInContext("session", context); },
    connect: () => vm.runInContext("connect()", context),
    settled: () => vm.runInContext("eventChain", context),
  };
}

function grant(socket, unitMilliseconds, outputSampleRate = 24000) {
  socket.receive({ type: "session.updated", session: { sglang: { granted: {
    native_unit_ms: unitMilliseconds,
    input_audio_format: { rate: 16000 },
    output_audio_format: { rate: outputSampleRate },
  } } } });
}

function status(generation) {
  return { type: "status", generation, droppedMs: 10, rate: 1, queueMs: 500, targetMs: 300, playedMs: 1000, buffering: false, underruns: 2, outputRms: 0.2, responseOpen: true, responseId: "unattributed" };
}

test("reused nodes restore each granted profile and attribute status only to its session", async () => {
  const app = createApp({ actualSampleRate: 48000 });
  let previousSession;
  let previousGeneration;
  let previousHandler;
  for (const unitMilliseconds of [200, 1000, 200]) {
    await app.connect();
    const socket = app.sockets.at(-1);
    socket.open();
    grant(socket, unitMilliseconds);
    await app.settled();
    assert.equal(app.nodes.length, 1);
    const node = app.nodes[0];
    const configuration = node.messages.findLast((message) => message.type === "config");
    const streaming = unitMilliseconds === 200;
    assert.equal(configuration.initialSamples, streaming ? 19200 : 14400);
    assert.equal(configuration.maxJitterSamples, streaming ? 43200 : 14400);
    assert.equal(configuration.maxSamples, streaming ? 168000 : 120000);
    assert.equal(configuration.keepSamples, streaming ? 105600 : 72000);
    assert.deepEqual({ ...configuration.catchUp }, streaming
      ? { slow: 0.5, fast: 1.0, skip: 2.0, slowRate: 1.06, fastRate: 1.12 }
      : { slow: 0.2, fast: 0.7, skip: 1.5, slowRate: 1.12, fastRate: 1.25 });
    assert.equal(configuration.nativeUnitSamples, streaming ? 9600 : 48000);
    const generation = node.messages.findLast((message) => message.type === "reset").generation;
    if (previousSession) {
      const previousRows = previousSession.trace.length;
      node.port.onmessage({ data: status(previousGeneration) });
      previousHandler({ data: status(previousGeneration) });
      assert.equal(previousSession.trace.length, previousRows);
      assert.equal(app.session.trace.filter((row) => row.direction === "local").length, 0);
    }
    node.port.onmessage({ data: status(generation) });
    const row = JSON.parse(app.session.exportTrace()).trace.at(-1);
    assert.equal(row.direction, "local");
    assert.equal(row.time_s, 12.345);
    assert.equal(row.event.type, "playback.status");
    assert.equal(row.event.generation, generation);
    assert.equal(row.event.underruns, 2);
    assert.equal(row.event.responseOpen, true);
    assert.equal("responseId" in row.event, false);
    socket.receive({ type: "response.output_audio.delta", response_id: "reply", delta: Buffer.alloc(48000, 32).toString("base64") });
    await app.settled();
    assert.equal(node.messages.at(-1).samples.length, 48000);
    const resets = node.messages.filter((message) => message.type === "reset").length;
    grant(socket, unitMilliseconds);
    await app.settled();
    assert.equal(node.messages.filter((message) => message.type === "reset").length, resets);
    previousSession = app.session;
    previousGeneration = generation;
    previousHandler = node.port.onmessage;
    await socket.close();
  }
});

test("audio waits for a replacement node to be configured and old node status is ignored", async () => {
  const moduleReady = Promise.withResolvers();
  const moduleLoading = Promise.withResolvers();
  const app = createApp({ loadModule: (sampleRate) => {
    if (sampleRate !== 48000) return Promise.resolve();
    moduleLoading.resolve();
    return moduleReady.promise;
  } });
  await app.connect();
  const socket = app.sockets[0];
  socket.open();
  grant(socket, 1000);
  await app.settled();
  const oldNode = app.nodes[0];
  const oldHandler = oldNode.port.onmessage;
  const oldGeneration = oldNode.messages.findLast((message) => message.type === "reset").generation;
  grant(socket, 1000, 48000);
  socket.receive({ type: "response.created", response: { id: "reply" } });
  socket.receive({ type: "response.output_audio.delta", response_id: "reply", delta: Buffer.alloc(96000, 32).toString("base64") });
  await moduleLoading.promise;
  assert.equal(oldNode.port.onmessage, null);
  assert.ok(oldNode.messages.every((message) => message.type !== "push"));
  moduleReady.resolve();
  await app.settled();
  const node = app.nodes.at(-1);
  assert.notEqual(node, oldNode);
  const messages = node.messages.map((message) => message.type);
  assert.ok(messages.indexOf("config") < messages.indexOf("push"));
  assert.ok(messages.indexOf("reset") < messages.indexOf("push"));
  assert.equal(node.messages.at(-1).samples.length, 48000);
  const rows = app.session.trace.length;
  oldHandler({ data: status(oldGeneration) });
  assert.equal(app.session.trace.length, rows);
});

test("a grant finishing during reconnect cannot configure or start the next session", async () => {
  const moduleReady = Promise.withResolvers();
  const moduleLoading = Promise.withResolvers();
  const app = createApp({ loadModule: (sampleRate) => {
    if (sampleRate !== 48000) return Promise.resolve();
    moduleLoading.resolve();
    return moduleReady.promise;
  } });
  await app.connect();
  const oldSocket = app.sockets[0];
  oldSocket.open();
  grant(oldSocket, 200, 48000);
  oldSocket.receive({ type: "response.output_audio.delta", response_id: "old", delta: "" });
  await moduleLoading.promise;
  await oldSocket.close();
  const reconnect = app.connect();
  moduleReady.resolve();
  await reconnect;
  assert.equal(app.microphoneRequests, 0);
  assert.ok(app.nodes.every((node) => node.messages.every((message) => !["config", "push"].includes(message.type))));
  const socket = app.sockets.at(-1);
  socket.open();
  grant(socket, 1000);
  await app.settled();
  assert.equal(app.microphoneRequests, 1);
  const node = app.nodes.at(-1);
  assert.equal(node.context.sampleRate, 24000);
  assert.equal(node.messages.find((message) => message.type === "config").nativeUnitSamples, 24000);
});

test("the first session.update carries what the settings panel chose, turn detection included", async () => {
  const app = createApp({ sessionOptions: {
    instructions: "Answer briefly.",
    outputModalities: ["audio"],
    extension: { voice: "NATM1", sampling: { temperature: 0.5 } },
    turnDetection: { type: "server_vad", silence_duration_ms: 800 },
  } });
  await app.connect();
  const socket = app.sockets[0];
  socket.open();
  await app.settled();
  assert.deepEqual(socket.sent.find((event) => event.type === "session.update").session, {
    output_modalities: ["audio"],
    instructions: "Answer briefly.",
    audio: { input: { turn_detection: { type: "server_vad", silence_duration_ms: 800 } } },
    sglang: { voice: "NATM1", sampling: { temperature: 0.5 } },
  });
});

// The stop button follows the grant: it stops the reply on the server where supports_cancel is true and only mutes playback
// elsewhere. Either way the mute outlives the response id, since a stopped model may continue its answer under a new one, and
// lifts only when the user speaks again: the server's speech_started on a turn-based deployment, the microphone on a native one.
test("the stop button cancels only where granted, mutes every response id until the user speaks again, and reports Stopped only after the cancelled response.done", async () => {
  const speech = Buffer.alloc(48000, 32).toString("base64");
  const quiet = Buffer.alloc(48000, 0).toString("base64");
  for (const [interaction, supportsCancel] of [["turn_based", true], ["native", false]]) {
    const elements = new Map();
    const app = createApp({ elements, microphone: true });
    await app.connect();
    const socket = app.sockets.at(-1);
    socket.open();
    socket.receive({ type: "session.updated", session: { sglang: { granted: { interaction, native_unit_ms: 1000, input_audio_format: { rate: 16000 }, output_audio_format: { rate: 24000 }, supports_cancel: supportsCancel } } } });
    await app.settled();
    // The microphone starts after the grant, outside the event chain.
    await new Promise((resolve) => setTimeout(resolve, 0));
    const capture = app.nodes.find((node) => node.name === "capture-processor");
    const playback = app.nodes[0];
    const button = elements.get("interruptBtn");
    const cancels = () => socket.sent.filter((event) => event.type === "response.cancel").length;
    const pushes = () => playback.messages.filter((message) => message.type === "push").length;
    const line = () => app.run("mood(0, 0, performance.now()).line");
    const microphone = (rms) => capture.port.onmessage({ data: { type: "frame", frame: new Int16Array(1280).buffer, rms } });
    assert.equal(button.title, supportsCancel ? "Stop the current answer" : "Mute the current answer here (the model keeps talking)");
    assert.equal(button.attributes["aria-label"], button.title);

    socket.receive({ type: "response.created", response: { id: "first" } });
    await app.settled();
    app.run("tick()");
    // A reply the server can stop is offered before it is audible; a local mute has nothing to mute yet.
    assert.equal(button.disabled, !supportsCancel);
    socket.receive({ type: "response.output_audio.delta", response_id: "first", delta: speech });
    await app.settled();
    assert.equal(pushes(), 1);
    app.advance(100);
    button.listeners.click();
    button.listeners.click();
    assert.equal(cancels(), supportsCancel ? 1 : 0);
    assert.equal(line(), supportsCancel ? "Stopping…" : "Muted until you speak");
    socket.receive({ type: "response.done", response: { id: "first", status: supportsCancel ? "cancelled" : "completed" } });
    await app.settled();
    assert.equal(line(), supportsCancel ? "Stopped" : "Muted until you speak");

    // The model goes on under a new response id while the microphone carries only silence: nothing is played.
    app.advance(1200);
    microphone(0);
    socket.receive({ type: "response.created", response: { id: "continued" } });
    socket.receive({ type: "response.output_audio.delta", response_id: "continued", delta: speech });
    socket.receive({ type: "response.output_audio.delta", response_id: "continued", delta: quiet });
    socket.receive({ type: "response.output_audio.delta", response_id: "continued", delta: speech });
    await app.settled();
    assert.equal(pushes(), 1);
    app.advance(2000);
    assert.equal(line(), "Muted until you speak");
    socket.receive({ type: "response.done", response: { id: "continued", status: "completed" } });
    await app.settled();

    // A genuinely new utterance lifts the mute and its reply plays.
    app.advance(5000);
    if (supportsCancel) {
      socket.receive({ type: "input_audio_buffer.speech_started", item_id: "item_user_1", audio_start_ms: 9000 });
      socket.receive({ type: "input_audio_buffer.speech_stopped", item_id: "item_user_1", audio_end_ms: 10000 });
    } else {
      microphone(0.3);
    }
    socket.receive({ type: "response.created", response: { id: "answer" } });
    socket.receive({ type: "response.output_audio.delta", response_id: "answer", delta: speech });
    await app.settled();
    assert.equal(pushes(), 2);
    // The reply is heard again: the orb follows the open reply on a turn-based server and the microphone on a native one.
    assert.equal(app.run("mood(0, 0, performance.now()).mood"), supportsCancel ? "speaking" : "listening");
    if (supportsCancel) {
      // A reply that ends on its own while its stop is pending was not stopped by the server.
      app.advance(100);
      button.listeners.click();
      assert.equal(cancels(), 2);
      assert.equal(line(), "Stopping…");
      socket.receive({ type: "response.done", response: { id: "answer", status: "completed" } });
      await app.settled();
      assert.equal(line(), "Muted until you speak");
    }
    await socket.close();
  }
});
