// Firstsmoke's interface. No framework, no build step, to match the shell.
//
// The HTTP client and the job event stream are servicekit's, imported from
// /shell/js/api.js, so there is one client and one job lifecycle across all five
// entries. Everything below is the part the shell cannot know about: the map,
// the bearing fan, the consultation trail and the camera wall.

import { api, ApiError } from '/shell/js/api.js';

const $ = (id) => document.getElementById(id);
const el = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};
const NS = 'http://www.w3.org/2000/svg';
const svg = (tag, attrs = {}, text) => {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  if (text !== undefined) node.textContent = text;
  return node;
};
const clock = (iso) => (iso ? iso.slice(11, 19) : '—');
const km = (m) => `${(m / 1000).toFixed(1)} km`;

const S = {
  network: null,
  scenario: null,
  running: false,
  transitions: [],
  consultations: [],
  evidence: [],
  alert: null,
  summary: null,
  version: null,
  startedAt: null,
  endedAt: null,
};

// ══════════════════════════════════════════════════════════════════ map ══

const MAP_W = 1000;
const MAP_H = 660;

function projector(points) {
  const lats = points.map((p) => p.lat);
  const lons = points.map((p) => p.lon);
  let minLat = Math.min(...lats), maxLat = Math.max(...lats);
  let minLon = Math.min(...lons), maxLon = Math.max(...lons);
  const padLat = Math.max((maxLat - minLat) * 0.42, 0.02);
  const padLon = Math.max((maxLon - minLon) * 0.42, 0.02);
  minLat -= padLat; maxLat += padLat; minLon -= padLon; maxLon += padLon;

  // One degree of longitude is shorter than one of latitude. A map that ignores
  // that turns a square crossing into a skewed one, and the whole point of this
  // picture is that the crossing angle is honest.
  const kx = Math.cos(((minLat + maxLat) / 2 * Math.PI) / 180);
  const spanLon = (maxLon - minLon) * kx;
  const spanLat = maxLat - minLat;
  const scale = Math.min(MAP_W / spanLon, MAP_H / spanLat);
  const ox = (MAP_W - spanLon * scale) / 2;
  const oy = (MAP_H - spanLat * scale) / 2;

  const project = (lat, lon) => [
    ox + (lon - minLon) * kx * scale,
    MAP_H - oy - (lat - minLat) * scale,
  ];
  project.metres = (m) => (m / 111_320) * scale;
  return project;
}

function contours(group, seed = 11) {
  // Not terrain. A quadrangle sheet's ruling, so the map reads as a drawing
  // rather than a blank rectangle, drawn faintly enough that nothing on it can
  // be mistaken for data. The contour interval label says 20 m and means it as
  // an illustration, which is why the label sits outside the drawing.
  let v = seed;
  const rand = () => (v = (v * 1103515245 + 12345) % 2147483648) / 2147483648;
  for (let i = 0; i < 13; i += 1) {
    const y0 = (i / 13) * MAP_H + rand() * 14;
    let d = `M 0 ${y0.toFixed(1)}`;
    for (let x = 60; x <= MAP_W; x += 60) {
      const w = Math.sin((x / MAP_W) * Math.PI * (1.3 + rand())) * (11 + rand() * 18);
      d += ` Q ${(x - 30).toFixed(1)} ${(y0 + w).toFixed(1)} ${x} ${(y0 + w * 0.5).toFixed(1)}`;
    }
    group.append(svg('path', { d, class: i % 5 === 0 ? 'contour contour--index' : 'contour' }));
  }
}

function cameras() {
  // The incident's own network once a run has started, the service's published
  // one before that. An uploaded bundle brings its own cameras, and drawing the
  // global list instead put the markers on the wrong hills.
  return (S.summary?.network?.cameras ?? S.network?.cameras ?? []);
}

function cameraFor(id) {
  return cameras().find((c) => c.camera_id === id) ?? null;
}

function plate(canvas, x, y, id, subtitle, anchorLeft) {
  const w = Math.max(id.length, subtitle.length) * 7.6 + 22;
  const h = 40;
  const px = anchorLeft ? x + 14 : x - w - 14;
  const py = y - h - 10;
  canvas.append(svg('rect', { x: px, y: py, width: w, height: h, rx: 4, class: 'plate' }));
  canvas.append(svg('text', { x: px + 11, y: py + 17, class: 'plate__id' }, id));
  canvas.append(svg('text', { x: px + 11, y: py + 32, class: 'plate__bearing' }, subtitle));
}

function drawMap() {
  const host = $('map');
  if (!host || !S.network) return;
  for (const child of [...host.children]) if (child.tagName !== 'SPAN') child.remove();

  const alert = S.alert;
  const rays = alert ? alert.rays : [];
  const fix = alert && alert.fix ? alert.fix : null;
  const approach = alert && alert.fix_refusal ? alert.fix_refusal.nearest_approach : null;

  const involved = new Set([
    ...rays.map((r) => r.camera_id),
    ...S.consultations.map((c) => c.camera_id),
  ]);
  const all = cameras();
  let shown = all.filter((c) => involved.has(c.camera_id));
  if (shown.length < 2) shown = all.slice(0, 8);

  const points = shown.map((c) => ({ lat: c.lat, lon: c.lon }));
  if (fix) points.push({ lat: fix.lat, lon: fix.lon });
  if (approach) points.push({ lat: approach.lat, lon: approach.lon });
  const project = projector(points);

  const canvas = svg('svg', {
    viewBox: `0 0 ${MAP_W} ${MAP_H}`,
    preserveAspectRatio: 'xMidYMid meet',
    role: 'img',
    'aria-label': describeMap(),
  });
  const sheet = svg('g', { 'aria-hidden': 'true' });
  contours(sheet);
  canvas.append(sheet);

  const supported = new Set(S.consultations.filter((c) => c.outcome === 'supported').map((c) => c.camera_id));
  const blind = new Set(S.consultations.filter((c) => c.outcome === 'blind').map((c) => c.camera_id));
  const onRay = new Set(rays.map((r) => r.camera_id));

  // Field-of-view wedges first, so everything else sits over them.
  for (const camera of shown) {
    const [x, y] = project(camera.lat, camera.lon);
    // Short. A wedge is a note about which way the camera faces, not a claim
    // about how far it sees; drawn at full range, three 90-degree wedges tinted
    // the whole survey sheet blue and the paper stopped reading as paper.
    const reach = Math.min(project.metres(camera.range_m ?? 40000), MAP_H * 0.22);
    const half = ((camera.hfov_deg ?? 90) / 2) * (Math.PI / 180);
    const mid = ((camera.azimuth_deg ?? 0) - 90) * (Math.PI / 180);
    canvas.append(svg('path', {
      class: 'wedge',
      d: `M ${x} ${y} L ${x + Math.cos(mid - half) * reach} ${y + Math.sin(mid - half) * reach}`
        + ` A ${reach} ${reach} 0 0 1 ${x + Math.cos(mid + half) * reach} ${y + Math.sin(mid + half) * reach} Z`,
    }));
  }

  // Bearings the world answered: ember, with their own uncertainty cone.
  const meetsAt = fix ? project(fix.lat, fix.lon)
    : (approach ? project(approach.lat, approach.lon) : null);
  for (const ray of rays) {
    const [x, y] = project(ray.lat, ray.lon);
    // Stop a little past where the bearings meet. A ray drawn to the camera's
    // full range leaves the sheet and implies the system claims something all
    // the way out there, which it does not.
    const reach = meetsAt
      ? Math.hypot(meetsAt[0] - x, meetsAt[1] - y) * 1.12
      : project.metres(Math.min(ray.length_m, 26000));
    const theta = ((ray.bearing_deg - 90) * Math.PI) / 180;
    const sigma = (ray.sigma_deg * Math.PI) / 180;
    for (const s of [-sigma, sigma]) {
      canvas.append(svg('line', {
        x1: x, y1: y,
        x2: x + Math.cos(theta + s) * reach, y2: y + Math.sin(theta + s) * reach,
        class: 'ray--soft',
      }));
    }
    const line = svg('line', {
      x1: x, y1: y, x2: x + Math.cos(theta) * reach, y2: y + Math.sin(theta) * reach,
      class: `ray${fix ? '' : ' ray--unconfirmed'}`,
    });
    line.append(svg('title', {}, `${ray.camera_id} bearing ${ray.bearing_deg.toFixed(1)} degrees, plus or minus ${ray.sigma_deg.toFixed(1)}`));
    canvas.append(line);
  }

  // Cameras that were asked and could not answer: a short dashed stub toward
  // where they were told to look, and a ring instead of a filled marker.
  for (const consultation of S.consultations) {
    if (supported.has(consultation.camera_id)) continue;
    const camera = cameraFor(consultation.camera_id);
    if (!camera) continue;
    const [x, y] = project(camera.lat, camera.lon);
    const theta = ((consultation.expected_bearing_deg - 90) * Math.PI) / 180;
    const reach = project.metres(9000);
    canvas.append(svg('line', {
      x1: x, y1: y, x2: x + Math.cos(theta) * reach, y2: y + Math.sin(theta) * reach,
      class: 'ray--discarded',
    }));
  }

  // The crossing, with its real uncertainty ellipse.
  if (fix) {
    const [fx, fy] = project(fix.lat, fix.lon);
    const rx = Math.max(project.metres(fix.semi_major_m), 7);
    const ry = Math.max(project.metres(fix.semi_minor_m), 5);
    canvas.append(svg('ellipse', { cx: fx, cy: fy, rx, ry, fill: 'url(#ell)', stroke: 'var(--danger)', 'stroke-width': 1.6 }));
    canvas.append(svg('circle', { cx: fx, cy: fy, r: 4, fill: 'var(--danger)' }));
    tagBox(canvas, fx + rx + 10, fy - 16,
      `${fix.lat.toFixed(4)} N, ${Math.abs(fix.lon).toFixed(4)} W`,
      `ellipse ${Math.round(fix.semi_major_m)} m by ${Math.round(fix.semi_minor_m)} m, bearings ${fix.crossing_angle_deg.toFixed(1)}° apart`);
  } else if (approach) {
    const [ax, ay] = project(approach.lat, approach.lon);
    canvas.append(svg('circle', { cx: ax, cy: ay, r: 5, class: 'approach__end' }));
    tagBox(canvas, ax + 16, ay + 14,
      `Nearest approach ${Math.round(approach.distance_m)} m`,
      `combined 1σ corridor here: ${Math.round(approach.corridor_m)} m`, true);
  }

  for (const camera of shown) {
    const [x, y] = project(camera.lat, camera.lon);
    const discarded = blind.has(camera.camera_id) || (!onRay.has(camera.camera_id) && S.consultations.length > 0);
    canvas.append(svg('circle', { cx: x, cy: y, r: 6, class: `marker${discarded ? ' marker--discarded' : ''}` }));
    if (onRay.has(camera.camera_id)) {
      const ray = rays.find((r) => r.camera_id === camera.camera_id);
      plate(canvas, x, y, camera.camera_id,
        `bearing ${ray.bearing_deg.toFixed(1)}° ± ${ray.sigma_deg.toFixed(1)}°`, x < MAP_W / 2);
    } else if (discarded) {
      const consultation = S.consultations.find((c) => c.camera_id === camera.camera_id);
      canvas.append(svg('text', { x: x + 12, y: y + 5, class: 'discard__label' },
        `${camera.camera_id}  ${consultation ? consultation.outcome.replace(/_/g, ' ') : 'not consulted'}`));
    } else {
      canvas.append(svg('text', { x: x + 12, y: y + 5, class: 'discard__label' }, camera.camera_id));
    }
  }

  host.append(canvas);
  host.append(bearingTable(rays, fix, approach));

  const legend = el('div', 'legend');
  legend.append(
    swatch('var(--danger)', fix ? 'bearing, with its ±' : 'bearing, unconfirmed'),
    swatch('var(--accent)', 'camera used'),
    swatch('var(--unusable)', 'camera discarded'),
  );
  if (approach && !fix) legend.append(swatch('var(--refuse)', 'nearest approach'));
  host.append(legend);
}

function bearingTable(rays, fix, approach) {
  // The map's text equivalent. A fan of coloured lines is unreadable to a
  // screen reader and to anyone printing this in black and white, so every
  // bearing is also a row of numbers.
  const wrap = el('div', 'sr-only');
  const table = el('table');
  table.append(el('caption', '', 'Bearings behind this decision'));
  const head = el('tr');
  for (const label of ['Camera', 'Bearing, degrees', 'Uncertainty, degrees']) {
    head.append(el('th', '', label));
  }
  const thead = el('thead');
  thead.append(head);
  table.append(thead);
  const body = el('tbody');
  for (const ray of rays) {
    const row = el('tr');
    row.append(el('td', '', ray.camera_id));
    row.append(el('td', '', ray.bearing_deg.toFixed(1)));
    row.append(el('td', '', ray.sigma_deg.toFixed(1)));
    body.append(row);
  }
  table.append(body);
  wrap.append(table);
  if (fix) {
    wrap.append(el('p', '', `They cross at ${fix.lat.toFixed(4)}, ${fix.lon.toFixed(4)}, at `
      + `${fix.crossing_angle_deg.toFixed(1)} degrees, inside an ellipse `
      + `${Math.round(fix.semi_major_m)} by ${Math.round(fix.semi_minor_m)} metres.`));
  } else if (approach) {
    wrap.append(el('p', '', 'They do not cross. Their nearest approach is '
      + `${Math.round(approach.distance_m)} metres, against a combined uncertainty of `
      + `${Math.round(approach.corridor_m)} metres at that range.`));
  } else {
    wrap.append(el('p', '', 'No bearing has been established.'));
  }
  return wrap;
}

function tagBox(canvas, x, y, title, sub, refuse = false) {
  const w = Math.max(title.length, sub.length) * 6.9 + 26;
  const h = 46;
  const px = Math.min(Math.max(x, 6), MAP_W - w - 6);
  const py = Math.min(Math.max(y, 6), MAP_H - h - 6);
  canvas.append(svg('rect', { x: px, y: py, width: w, height: h, rx: 4, class: `tag${refuse ? ' tag--refuse' : ''}` }));
  canvas.append(svg('text', { x: px + 13, y: py + 19, class: 'tag__title' }, title));
  canvas.append(svg('text', { x: px + 13, y: py + 35, class: 'tag__sub' }, sub));
}

function swatch(colour, label) {
  const node = el('span');
  const bar = el('i');
  bar.style.background = colour;
  node.append(bar, document.createTextNode(label));
  return node;
}

function describeMap() {
  if (S.alert && S.alert.fix) {
    return `Map: ${S.alert.rays.length} bearings crossing at ${S.alert.fix.lat.toFixed(4)}, `
      + `${S.alert.fix.lon.toFixed(4)}, inside an ellipse ${Math.round(S.alert.fix.semi_major_m)} metres across`;
  }
  if (S.consultations.length) return 'Map: the cameras consulted, and the bearings they were asked about';
  return 'Map of the camera network';
}

// ═══════════════════════════════════════════════════════════════ panels ══

function drawTop() {
  const state = S.summary?.state ?? (S.running ? 'watch' : 'idle');
  $('tb-dot').dataset.state = state;
  $('tb-title').textContent = S.scenario ? S.scenario.title : 'No incident loaded';
  const consulted = new Set(S.consultations.map((c) => c.camera_id)).size;
  const total = S.summary?.cameras ?? cameras().length;
  $('tb-consulted').innerHTML = `Consulted <b>${consulted}</b> of ${total} cameras`;

  if (S.startedAt) {
    const elapsed = S.endedAt ? Math.round((S.endedAt - S.startedAt) / 1000) : null;
    const last = S.transitions.at(-1);
    $('tb-times').innerHTML =
      `Flag <b>${clock(S.transitions[0]?.at)}</b> → ${last ? last.to.replace(/_/g, ' ') : 'watching'} `
      + `<b>${clock(last?.at)}</b>${elapsed !== null ? `, ${elapsed} s` : ''}`;
  }
  $('tb-raw').hidden = !S.evidence.length;
}

function drawRail() {
  $('rf-opencv').textContent = S.version ? S.version.opencv_version : '…';
  $('rf-sha').textContent = S.version ? S.version.git_sha : '…';
  $('rf-frames').textContent = S.summary ? String(S.summary.frames_read) : '—';
  $('rf-consulted').textContent = String(new Set(S.consultations.map((c) => c.camera_id)).size || '—');

  const outcome = $('rf-outcome');
  const state = S.summary?.state ?? (S.running ? 'watching' : 'idle');
  outcome.textContent = state.replace(/_/g, ' ');
  outcome.className = state === 'alerted' ? 'hot' : state === 'needs_human' ? 'refuse' : '';
  $('nav-map').textContent = String(cameras().length || '—');
  $('nav-wall').textContent = String(S.evidence.length || '—');
  $('nav-detections').textContent = String(S.alert ? 1 : 0);
  $('nav-timeline').textContent = String(S.transitions.length || '—');
  $('trail-panel').closest('.split')?.classList.toggle('is-alert', Boolean(S.alert));
}

function drawKpis() {
  const host = $('kpis');
  if (!S.summary) { host.hidden = true; return; }
  host.hidden = false;
  host.replaceChildren();
  const alert = S.alert;
  const fix = alert && alert.fix ? alert.fix : null;
  const approach = alert && alert.fix_refusal ? alert.fix_refusal.nearest_approach : null;

  const cards = [
    { k: 'Cameras watched', v: String(S.summary.cameras), n: 'every one, every tick' },
    { k: 'Frames read', v: String(S.summary.frames_read), n: 'including the re-reads' },
    {
      k: 'Confidence', v: alert ? alert.confidence.toFixed(2) : '—',
      n: alert ? 'of 1.00, after the rejectors' : 'nothing above 0.35',
      cls: alert ? 'hot' : '',
    },
  ];
  if (fix) {
    cards.push(
      { k: 'Position', v: `${fix.lat.toFixed(4)}, ${fix.lon.toFixed(4)}`, n: `± ${Math.round(fix.semi_major_m)} m along the long axis`, mono: true },
      { k: 'Crossing angle', v: fix.crossing_angle_deg.toFixed(1), unit: '°', n: 'below 8° the fix is refused', cls: 'good' },
    );
  } else if (approach) {
    cards.push(
      { k: 'Nearest approach', v: km(approach.distance_m), n: `corridor ${Math.round(approach.corridor_m)} m at that range`, cls: 'refuse' },
      { k: 'Position', v: 'not reported', n: 'the bearings do not meet', cls: 'refuse' },
    );
  } else {
    cards.push({ k: 'Unusable cameras', v: String(S.summary.unusable.length), n: 'named, with the fault' });
  }

  for (const card of cards) {
    const node = el('div', `kpi ${card.cls ?? ''}`.trim());
    node.append(el('div', 'k', card.k));
    const value = el('div', `v${card.mono ? ' mono' : ''}`);
    value.append(document.createTextNode(card.v));
    if (card.unit) value.append(el('u', '', card.unit));
    node.append(value);
    node.append(el('div', 'n', card.n));
    host.append(node);
  }
}

function drawCallout() {
  const box = $('callout');
  const text = $('callout-text');
  const actions = $('callout-actions');
  const heading = box.querySelector('h3');
  actions.replaceChildren();

  const alert = S.alert;
  if (!alert) {
    box.dataset.tone = 'plain';
    heading.textContent = S.running ? 'The watch is running' : 'Nothing is being watched yet';
    text.replaceChildren(document.createTextNode(S.running
      ? 'Every camera is read on every tick. Nothing is reported until a candidate has been '
        + 'tracked for long enough that its growth rate means something.'
      : 'Four incidents are bundled with the service, one for each way the watch can end: a '
        + 'confirmed crossing, a stand-down, a fire nobody can corroborate, and a pair of '
        + 'cameras that are up but not watching anything.'));
    return;
  }

  const fix = alert.fix;
  const approach = alert.fix_refusal ? alert.fix_refusal.nearest_approach : null;
  if (fix) {
    box.dataset.tone = 'alert';
    heading.textContent = alert.headline;
    actions.append(button('btn-hot', 'Escalate to dispatch'), button('btn-line', 'Keep watching, do not escalate'));
  } else {
    box.dataset.tone = 'refuse';
    heading.textContent = alert.state === 'needs_human'
      ? 'Firstsmoke will not close this on its own'
      : 'Firstsmoke does not report a location it cannot triangulate';
    actions.append(button('btn-line', 'Ask a wider ring of cameras'), button('btn-line', 'Escalate anyway, with a reason'));
  }
  // Each reasoning line is a whole sentence about a different camera. Joining
  // them with spaces produced one unreadable paragraph, so they get a list.
  text.replaceChildren();
  const list = el('ul', 'why');
  for (const line of alert.reasoning.slice(0, 5)) list.append(el('li', '', tidy(line)));
  text.append(list);
}

function tidy(line) {
  const trimmed = line.trim();
  return /[.!?]$/.test(trimmed) ? trimmed : `${trimmed}.`;
}

function button(cls, label) {
  const node = el('button', cls, label);
  node.type = 'button';
  node.disabled = true;
  node.title = 'This demonstration stops at the alert. Nothing is dispatched.';
  return node;
}

function drawTrail() {
  const host = $('trail');
  host.replaceChildren();
  const held = $('held');

  const steps = [];
  for (const transition of S.transitions) {
    if (transition.trigger === 'alert_raised') continue;
    let detail = transition.detail;
    if (transition.trigger === 'bearing_selected_cameras') {
      // The spec asks for the actual rule that selected each camera, not a
      // summary of it, so each candidate prints its own distance and crossing
      // angle exactly as the geometry computed them.
      const reasons = (transition.data?.candidates ?? [])
        .map((c) => `${c.camera_id}, ${c.why}`)
        .join('; ');
      if (reasons) detail += `. Chosen because ${reasons}.`;
    }
    steps.push({
      at: transition.at,
      head: headline(transition),
      detail,
      tone: ['crossed_bearings', 'weak_detection'].includes(transition.trigger) ? 'hot'
        : ['neighbours_blind', 'unresolved', 'no_second_view'].includes(transition.trigger) ? 'refuse'
        : transition.trigger === 'not_corroborated' ? 'wait' : '',
    });
  }

  if (!steps.length) {
    host.append(el('p', 'empty', 'The trail fills in as the watch runs.'));
    held.hidden = true;
    return;
  }
  steps.forEach((step, index) => {
    const row = el('div', `step ${step.tone}`.trim());
    row.append(el('div', 't', String(index + 1)));
    const body = el('div');
    const head = el('div', 'h');
    head.append(el('time', '', clock(step.at)), document.createTextNode(step.head));
    body.append(head, el('div', 'd', step.detail));
    row.append(body);
    host.append(row);
  });

  $('trail-sub').textContent = `${steps.length} steps`;
  if (S.alert) {
    held.hidden = false;
    held.innerHTML = S.alert.state === 'alerted'
      ? '<b>Next: a human decides.</b> The watch holds at the escalation card. It does not '
        + 'dispatch, it cannot move a camera, and it has no way to contact anyone outside this page.'
      : '<b>Held open.</b> Nothing was sent. The flag stays open so that a later frame, or a '
        + 'camera that comes back, can settle it.';
  } else {
    held.hidden = !S.summary;
    if (S.summary) {
      held.innerHTML = '<b>Nothing was raised.</b> Every candidate either failed to grow like a '
        + 'column or was contradicted by a camera that could see the same bearing.';
    }
  }
}

const HEADLINES = {
  weak_detection: 'Flagged a weak change',
  re_examined: 'Re-read its own recent frames',
  bearing_selected_cameras: 'Chose cameras from the bearing',
  crossed_bearings: 'Crossing found',
  not_corroborated: 'Stood down',
  faded: 'Candidate faded',
  neighbours_blind: 'Nobody could corroborate',
  unresolved: 'Could not settle it',
  no_second_view: 'No second view exists',
  single_camera_confident: 'One camera, confident enough',
  sequence_ended: 'The recording ended',
};
function headline(transition) {
  const base = HEADLINES[transition.trigger] ?? transition.to.replace(/_/g, ' ');
  return transition.cameras.length ? `${base} — ${transition.cameras[0]}` : base;
}

function drawWall() {
  const panel = $('wall-panel');
  const host = $('wall');
  if (!S.evidence.length) { panel.hidden = true; return; }
  panel.hidden = false;
  host.replaceChildren();

  const byLabel = new Map();
  for (const consultation of S.consultations) byLabel.set(consultation.label, consultation);
  const alertCamera = S.alert ? S.alert.origin_camera : null;

  for (const item of S.evidence) {
    const consultation = byLabel.get(item.label);
    const usable = item.metrics?.usability === 'usable' || item.metrics?.usability === 'degraded';
    const camera = cameras().find((c) => c.label === item.label);
    const flagged = camera && camera.camera_id === alertCamera;

    let cls = 'tile';
    if (flagged || consultation?.outcome === 'supported') cls += ' flag';
    else if (consultation) cls += ' asked';
    if (!usable) cls += ' dark';

    const tile = el('figure', cls);
    tile.style.margin = '0';
    if (item.uri) {
      const img = el('img');
      img.src = item.uri;
      img.alt = `${item.label}: ${item.caption || 'annotated frame'}`;
      img.loading = 'lazy';
      tile.append(img);
    }
    tile.append(el('span', 'st', usable ? `${(item.metrics?.confidence ?? 0).toFixed(2)}` : (item.metrics?.usability ?? 'unusable')));
    const lab = el('div', 'lab');
    lab.append(el('div', 'nm', item.label));
    lab.append(el('div', 'nums', `${camera ? camera.camera_id : ''} · frame ${item.frame_index ?? '—'}`));
    lab.append(el('div', 'why', consultation ? consultation.answer : (item.caption || '')));
    tile.append(lab);
    host.append(tile);
  }
  $('wall-sub').textContent = `${S.evidence.length} cameras`;
}

function drawTimeline() {
  const panel = $('timeline-panel');
  const host = $('timeline');
  if (!S.transitions.length) { panel.hidden = true; return; }
  panel.hidden = false;
  host.replaceChildren();

  const table = el('table');
  const thead = el('thead');
  const head = el('tr');
  for (const [label, cls] of [['Time', ''], ['State', ''], ['What changed', ''], ['Cameras', '']]) {
    head.append(el('th', cls, label));
  }
  thead.append(head);
  table.append(thead);
  const body = el('tbody');
  for (const transition of S.transitions) {
    const row = el('tr');
    row.append(el('td', 'mono', clock(transition.at)));
    const state = el('td');
    const tone = transition.to === 'alerted' || transition.to === 'confirmed' ? 'hot'
      : transition.to === 'needs_human' ? 'refuse'
      : transition.to === 'stood_down' ? 'ok' : '';
    state.append(el('span', `chip ${tone}`.trim(), transition.to.replace(/_/g, ' ')));
    row.append(state);
    row.append(el('td', '', transition.detail));
    row.append(el('td', 'mono', transition.cameras.join(', ') || '—'));
    body.append(row);
  }
  table.append(body);
  host.append(table);
  $('timeline-sub').textContent = `${S.transitions.length} transitions`;
}

function drawMethod() {
  const host = $('method');
  if (host.children.length) return;
  const items = [
    ['It watches how a shape behaves, not how it looks',
     'A cloud, a dust plume and a column of smoke look alike in one still. Over ten minutes '
     + 'they do not. Smoke keeps its base where the fire is and pushes its top upward; a cloud '
     + 'moves all of itself at the wind speed; dust runs along the ground without rising.'],
    ['Every score is a weighted sum you can read',
     'Eight named terms, each between 0 and 1, each carrying the sentence that explains what it '
     + 'measured. A rejector then multiplies the score down and says which impostor it thinks '
     + 'this is. Nothing is hidden in a model.'],
    ['A camera that cannot see is not a camera that sees nothing',
     'Night, fog, a wet lens, direct sun and a frozen feed are five separate verdicts, each with '
     + 'its own test. Silence from a blind camera is never counted as evidence of an empty hillside.'],
    ['It stops at the alert',
     'The system may read a camera it was not reading, re-read one it already read, and ask a '
     + 'human. It cannot pan a camera and it cannot dispatch anything. A false alarm costs someone '
     + 'a look at a picture; the alternative costs more.'],
  ];
  for (const [title, text] of items) {
    const card = el('div');
    card.append(el('h3', '', title), el('p', '', text));
    host.append(card);
  }
}

function redraw() {
  drawTop(); drawRail(); drawKpis(); drawCallout(); drawMap(); drawTrail(); drawWall(); drawTimeline();
}

// ══════════════════════════════════════════════════════════════════ run ══

function reset(scenario) {
  S.scenario = scenario ?? null;
  S.transitions = []; S.consultations = []; S.evidence = [];
  S.alert = null; S.summary = null;
  S.startedAt = Date.now(); S.endedAt = null;
  $('split').hidden = false;
  if (scenario) {
    $('page-title').textContent = scenario.title;
    $('page-sub').textContent = scenario.blurb + ' ' + scenario.expect;
  }
  redraw();
}

async function runScenario(scenario, button) {
  if (S.running) return;
  S.running = true;
  reset(scenario);
  for (const node of document.querySelectorAll('.scenario')) {
    node.disabled = true;
    node.setAttribute('aria-pressed', String(node === button));
  }
  $('run-default').disabled = true;
  $('rf-meter').style.width = '4%';

  try {
    const response = await fetch(`/api/scenarios/${encodeURIComponent(scenario.name)}`, { method: 'POST' });
    const payload = await response.json();
    if (!response.ok) throw new ApiError(payload.error ?? {}, response.status);
    follow(payload.job_id);
  } catch (error) {
    finished();
    $('callout-text').replaceChildren(document.createTextNode(error.message ?? String(error)));
  }
}

function finished() {
  S.running = false;
  S.endedAt = Date.now();
  for (const node of document.querySelectorAll('.scenario')) node.disabled = false;
  $('run-default').disabled = false;
}

function follow(jobId) {
  api.events(jobId, {
    progress: (event) => { $('rf-meter').style.width = `${event.percent ?? 0}%`; },
    note: (event) => {
      if (event.transition) S.transitions.push(event.transition);
      if (event.consultation) S.consultations.push(event.consultation);
      redraw();
    },
    status: async (event) => {
      if (event.status !== 'done' && event.status !== 'failed') return;
      finished();
      $('rf-meter').style.width = '100%';
      try {
        const job = await api.job(jobId);
        const record = job.result;
        if (job.error) $('callout-text').replaceChildren(document.createTextNode(job.error.message));
        if (record) {
          S.evidence = record.evidence ?? [];
          const summary = (record.results ?? []).find((r) => r && r.transitions);
          if (summary) {
            S.summary = summary;
            S.transitions = summary.transitions ?? S.transitions;
            S.consultations = summary.consultations ?? S.consultations;
            S.alert = (summary.alerts ?? [])[0] ?? null;
          }
        }
      } catch { /* the job endpoint already reports its own errors */ }
      redraw();
    },
  });
}

// ═════════════════════════════════════════════════════════════════ boot ══

function initTheme() {
  const toggle = $('theme-toggle');
  const stored = localStorage.getItem('firstsmoke-theme');
  if (stored) document.documentElement.dataset.theme = stored;
  const label = () => {
    const current = document.documentElement.dataset.theme || 'system';
    toggle.textContent = current === 'dark' ? 'Dark' : current === 'light' ? 'Light' : 'Theme';
  };
  label();
  toggle.addEventListener('click', () => {
    const order = ['', 'light', 'dark'];
    const next = order[(order.indexOf(document.documentElement.dataset.theme || '') + 1) % order.length];
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem('firstsmoke-theme', next); } catch { /* private mode */ }
    label();
  });
}

(async function boot() {
  initTheme();
  drawMethod();
  try {
    const [network, scenarios, version] = await Promise.all([
      fetch('/api/network').then((r) => r.json()),
      fetch('/api/scenarios').then((r) => r.json()),
      api.version(),
    ]);
    S.network = network;
    S.version = version;
    const attribution = $('attribution');
    if (attribution && network.attribution) attribution.textContent = network.attribution;

    const host = $('scenarios');
    host.replaceChildren();
    const list = scenarios.scenarios ?? [];
    for (const scenario of list) {
      const node = el('button', 'scenario');
      node.type = 'button';
      node.setAttribute('aria-pressed', 'false');
      node.append(el('span', 'st', scenario.title), el('span', 'sb', scenario.blurb), el('span', 'se', scenario.expect));
      node.addEventListener('click', () => runScenario(scenario, node));
      host.append(node);
    }
    $('run-default').addEventListener('click', () => {
      const first = document.querySelector('.scenario');
      if (first) first.click();
    });
    redraw();
  } catch (error) {
    console.error(error);
    $('page-sub').textContent = 'Could not load the camera network from this service.';
  }
})();
