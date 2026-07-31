/**
 * Behavioural guard for the SSE error contract.
 *
 * EventSource fires "error" for transport failures as well as for a
 * server-sent `event: error`, and both dispatch to the same listener. The
 * shipped bug: a dropped connection had no data, so `d.message || "analysis
 * failed"` put a bare "analysis failed" on screen, closed the stream, and
 * pre-empted the reconnect/poll recovery — killing a run that was still fine
 * on the server. Only Render dropped connections, so it never showed locally.
 *
 * This drives a real EventSource against a real server that drops mid-stream,
 * with the handler logic mirrored from the page.
 *
 * Run:  node --experimental-eventsource --test backend/tests/sse_contract.test.mjs
 *
 * (Node exposes a global EventSource only behind that flag as of v24 — the
 * point of this suite is to use the real thing rather than a hand-rolled stub
 * that could not reproduce the bug.)
 */
import { createServer } from "node:http";
import test from "node:test";
import assert from "node:assert/strict";

const HAVE_ES = typeof EventSource !== "undefined";
const needsFlag = {
  skip: HAVE_ES ? false : "needs node --experimental-eventsource",
};

function startServer() {
  const server = createServer((req, res) => {
    if (req.url !== "/drop" && req.url !== "/err") {
      res.writeHead(404).end();
      return;
    }
    res.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
    });
    res.write('id: 1\nevent: step_start\ndata: {"step":0}\n\n');
    setTimeout(() => {
      if (req.url === "/err") {
        const payload = JSON.stringify({ message: "a real backend failure" });
        res.write(`id: 2\nevent: error\ndata: ${payload}\n\n`);
      }
      // Abrupt destroy == what an OOM-killed worker or proxy timeout looks
      // like to the browser: an error event carrying no data.
      setTimeout(() => res.destroy(), 60);
    }, 60);
  });
  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () =>
      resolve({ server, port: server.address().port }),
    );
  });
}

/** Mirrors the page's _listen error handling. */
function listen(url, { guard }) {
  return new Promise((resolve) => {
    const st = { phase: "running", error: null, esErrs: 0 };
    const es = new EventSource(url);
    const J = (e) => {
      try {
        return JSON.parse(e.data);
      } catch {
        return {};
      }
    };

    es.onerror = () => {
      if (st.phase !== "running") return;
      st.esErrs += 1; // reconnect, then fall back to polling
    };

    es.addEventListener("error", (e) => {
      if (guard && typeof e.data !== "string") return; // THE FIX
      const d = J(e);
      es.close();
      st.phase = "upload";
      st.error = d.message || "analysis failed";
    });

    setTimeout(() => {
      es.close();
      resolve(st);
    }, 700);
  });
}

test("a dropped stream must not be reported as a failed analysis", needsFlag, async () => {
  const { server, port } = await startServer();
  try {
    const fixed = await listen(`http://127.0.0.1:${port}/drop`, { guard: true });
    assert.equal(fixed.phase, "running", "a transport drop ended the run");
    assert.equal(fixed.error, null, "a transport drop surfaced a user-facing error");
    assert.ok(fixed.esErrs > 0, "onerror never saw the drop; test is not exercising it");
  } finally {
    server.close();
  }
});

test("without the guard the old bug reproduces (test is meaningful)", needsFlag, async () => {
  const { server, port } = await startServer();
  try {
    const old = await listen(`http://127.0.0.1:${port}/drop`, { guard: false });
    assert.equal(old.phase, "upload");
    assert.equal(old.error, "analysis failed");
  } finally {
    server.close();
  }
});

test("a genuine server-sent error still surfaces its own message", needsFlag, async () => {
  const { server, port } = await startServer();
  try {
    const fixed = await listen(`http://127.0.0.1:${port}/err`, { guard: true });
    assert.equal(fixed.phase, "upload", "a real backend error was swallowed");
    assert.equal(fixed.error, "a real backend failure");
  } finally {
    server.close();
  }
});
