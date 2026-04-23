// Headless smoke test that verifies:
//   1. Vite serves the demo
//   2. jax-js + the vendored @jax-js/onnx load successfully
//   3. Both ONNX branches compile
//   4. The detection / tracking presets render in the DOM
//   5. Clicking a preset wires through to `Running template branch...`
//
// Full end-to-end inference requires a browser backend that supports fp16
// (native WebGPU with `shader-f16`, typically Chrome on Windows / macOS with
// a recent GPU). On headless-Linux CI the available backends (cpu, wasm,
// webgl, SwiftShader-WebGPU) do not support fp16 yet; that limitation is
// upstream in jax-js and isn't anything UMatcher-jax-js itself controls.
//
// Run with:
//   node test-umatcher.mjs              # default headless check
//   UMATCHER_URL=http://host:port/ node test-umatcher.mjs
import { chromium } from "playwright";

const URL = process.env.UMATCHER_URL || "http://127.0.0.1:5173/";

const browser = await chromium.launch({
  headless: true,
  args: [
    "--enable-unsafe-webgpu",
    "--enable-features=Vulkan,UseSkiaRenderer,WebGPU",
    "--use-gl=swiftshader",
    "--use-angle=swiftshader",
    "--ignore-gpu-blocklist",
    "--enable-webgl",
    "--no-sandbox",
    "--disable-dev-shm-usage",
  ],
});
const context = await browser.newContext({ viewport: { width: 1400, height: 900 } });
const page = await context.newPage();
page.setDefaultTimeout(120000);

const consoleLogs = [];
const failedReqs = [];
page.on("console", (msg) => consoleLogs.push(`[${msg.type()}] ${msg.text()}`));
page.on("pageerror", (err) => consoleLogs.push(`[pageerror] ${err.message}`));
page.on("requestfailed", (req) => failedReqs.push(`${req.method()} ${req.url()} -> ${req.failure()?.errorText}`));

let exitCode = 0;
try {
  console.log("Navigating to", URL);
  await page.goto(URL, { waitUntil: "domcontentloaded", timeout: 60000 });

  console.log("Waiting for backend readiness...");
  await page.waitForFunction(
    () => {
      const txt = document.getElementById("backend-line")?.textContent || "";
      return txt.includes("ready") || txt.includes("Could not");
    },
    { timeout: 120000, polling: 500 },
  );
  const backend = await page.locator("#backend-line").innerText();
  console.log("backend-line =", backend);
  if (backend.includes("Could not")) {
    throw new Error("Model / jax-js load failed: " + backend);
  }

  const detectionPresets = await page.locator("#detectionPresets .preset-card").count();
  const trackerPresets = await page.locator("#trackerPresets .preset-card").count();
  console.log(`Detection presets: ${detectionPresets}`);
  console.log(`Tracker presets  : ${trackerPresets}`);
  if (detectionPresets < 1) throw new Error("no detection presets rendered");
  if (trackerPresets < 1) throw new Error("no tracker presets rendered");

  // Capture status updates while we click the first detection preset.
  await page.exposeFunction("onStatus", (t) => {
    const line = `[status] ${t}`;
    console.log(line);
    consoleLogs.push(line);
  });
  await page.evaluate(() => {
    const s = document.getElementById("status");
    if (!s) return;
    new MutationObserver(() => window.onStatus?.(s.textContent || ""))
      .observe(s, { childList: true, characterData: true, subtree: true });
  });

  console.log("Clicking first detection preset...");
  await page.locator("#detectionPresets .preset-card").first().click();

  // We only wait for the template branch to kick in - any further work needs
  // fp16-capable GPU which isn't available in this headless environment.
  // Terminal states: inference succeeded, inference failed, or the handler
  // bailed out early. Any of these prove our UI wiring went all the way to
  // the model-inference code path.
  await page.waitForFunction(
    () => {
      const t = document.getElementById("status")?.textContent || "";
      return (
        t.includes("Found ") ||
        t.includes("failed") ||
        t.includes("Set a template first")
      );
    },
    { timeout: 60000, polling: 250 },
  );
  const status = await page.locator("#status").innerText();
  console.log("status =", status);

  await page.screenshot({ path: "/tmp/umatcher-ui.png", fullPage: true });
  console.log("screenshot saved to /tmp/umatcher-ui.png");

  const sawTemplate = consoleLogs.some((l) => l.includes("[status] Running template branch"));
  if (status.includes("Found ")) {
    console.log("SUCCESS: full detection completed in headless run");
  } else if (sawTemplate) {
    console.log("UI OK: template branch executed; full inference needs");
    console.log("a browser with native WebGPU + shader-f16 (Chrome on Win/macOS).");
  } else {
    console.log("UI OK but inference path did not start. status =", status);
    exitCode = 2;
  }
} catch (err) {
  console.error("FAIL:", err.message);
  console.log("\n--- Recent console logs ---");
  for (const l of consoleLogs.slice(-30)) console.log(l);
  console.log("\n--- Failed requests ---");
  for (const r of failedReqs) console.log(r);
  await page.screenshot({ path: "/tmp/umatcher-fail.png", fullPage: true }).catch(() => {});
  exitCode = 1;
}

await browser.close();
process.exit(exitCode);
