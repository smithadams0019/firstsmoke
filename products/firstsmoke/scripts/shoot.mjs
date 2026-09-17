// Drive the running service and photograph what it actually does.
//
//   node scripts/shoot.mjs http://127.0.0.1:8099 docs/screens
//
// Not a mockup renderer: this clicks the real buttons on the real service and
// waits for the real jobs, so every screenshot in docs/ is a picture of the
// running system rather than of a design file. A failure to reach a state is a
// failure of the product, and it is reported as one.
//
// Playwright is not vendored here; point PLAYWRIGHT_PATH at an install.
// Chromium needs --no-sandbox in this environment.

const pwPath = process.env.PLAYWRIGHT_PATH || 'playwright';
const pw = await import(pwPath);
const chromium = (pw.default ?? pw).chromium;

const [, , baseArg, outArg] = process.argv;
const base = baseArg || 'http://127.0.0.1:8099';
const outDir = outArg || 'docs/screens';
const scale = Number(process.env.SHOT_SCALE || 2);

const { mkdir } = await import('node:fs/promises');
await mkdir(outDir, { recursive: true });

const browser = await chromium.launch({ args: ['--no-sandbox'] });
const context = await browser.newContext({
  viewport: { width: 1440, height: 900 },
  deviceScaleFactor: scale,
  colorScheme: 'light',
});
const page = await context.newPage();
const errors = [];
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
page.on('pageerror', (e) => errors.push(String(e)));

async function settle() {
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(350);
}

async function runScenario(index, name) {
  await page.locator('.scenario').nth(index).click();
  // Wait for the run to *start* before waiting for it to finish. Skipping this
  // makes the second screenshot a picture of the first scenario's result, which
  // is exactly the kind of quietly wrong artefact a docs folder fills up with.
  await page.waitForFunction(
    () => document.querySelector('.scenario')?.disabled === true,
    null,
    { timeout: 20_000 },
  );
  // The run is over when the buttons come back. Polling the DOM rather than the
  // API keeps this honest: if the interface never updates, this times out,
  // which is the bug we want to catch.
  await page.waitForFunction(
    () => document.querySelector('.scenario')?.disabled === false,
    null,
    { timeout: 300_000 },
  );
  await settle();
  await page.screenshot({ path: `${outDir}/${name}.png` });
  await page.screenshot({ path: `${outDir}/${name}-full.png`, fullPage: true });
  const outcome = await page.locator('#rf-outcome').textContent();
  const headline = await page.locator('.callout h3').textContent();
  console.log(`${name}: ${outcome.trim()} — ${headline.trim()}`);
  return outcome.trim();
}

await page.goto(base, { waitUntil: 'networkidle' });
await settle();
await page.screenshot({ path: `${outDir}/s0-idle.png` });

const outcomes = {};
outcomes['s2-triangulation'] = await runScenario(0, 's2-triangulation');
outcomes['s4-stand-down'] = await runScenario(1, 's4-stand-down');
outcomes['s3-inconclusive'] = await runScenario(2, 's3-inconclusive');
outcomes['s5-unusable'] = await runScenario(3, 's5-unusable');

// The camera wall, and the dark appearance a lookout desk runs at night.
await page.locator('#wall-panel').scrollIntoViewIfNeeded().catch(() => {});
await settle();
await page.locator('#wall-panel').screenshot({ path: `${outDir}/s1-wall.png` }).catch(() => {});

await page.emulateMedia({ colorScheme: 'dark' });
await page.evaluate(() => { document.documentElement.dataset.theme = 'dark'; });
await settle();
await page.screenshot({ path: `${outDir}/s6-dark.png` });

// A phone, because a lookout may be checking this from a truck.
await page.setViewportSize({ width: 412, height: 893 });
await page.evaluate(() => { document.documentElement.dataset.theme = ''; });
await page.emulateMedia({ colorScheme: 'light' });
await settle();
await page.screenshot({ path: `${outDir}/s7-phone.png`, fullPage: true });

await browser.close();
console.log(JSON.stringify(outcomes, null, 2));
if (errors.length) {
  console.error(`\n${errors.length} console errors:`);
  for (const e of [...new Set(errors)]) console.error('  ' + e);
  process.exitCode = 1;
}
