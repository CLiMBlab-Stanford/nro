// Browser client for `nro viz`: draws cortical surfaces and composites map layers.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TrackballControls } from "three/addons/controls/TrackballControls.js";

const $ = (id) => document.getElementById(id);
const HEMISPHERES = ["L", "R"];
const KINDS = ["inflated", "midthickness", "pial", "white"];
const SPLIT_GAP_MM = 6;
const FIT_MARGIN = 0.06; // Fraction of the canvas left empty on each side.
const SEED_COLOR = 0x19d36b;

const COLORMAPS = {
  viridis: [[0.267, 0.005, 0.329], [0.229, 0.322, 0.546], [0.128, 0.567, 0.551], [0.369, 0.789, 0.383], [0.993, 0.906, 0.144]],
  hot: [[0.55, 0, 0], [1, 0, 0], [1, 0.55, 0], [1, 1, 0], [1, 1, 0.8]],
  cool: [[0, 0, 0.6], [0, 0.3, 1], [0, 0.8, 1], [0.6, 1, 1]],
  gray: [[0.1, 0.1, 0.1], [0.95, 0.95, 0.95]],
  // Signed maps use "hot" for positive values and this ramp for negative values.
  negative: [[0, 0, 0.6], [0, 0.3, 1], [0, 0.8, 1], [0.6, 1, 1]],
};
// Single-color schemes keep a fixed color and map the value to opacity instead.
const SOLID_COLORS = {
  red: [1, 0.1, 0.1],
  green: [0.1, 0.85, 0.2],
  blue: [0.15, 0.35, 1],
  yellow: [1, 0.9, 0.1],
  cyan: [0.1, 0.9, 0.95],
  magenta: [0.95, 0.2, 0.9],
};
const SOLID = Object.keys(SOLID_COLORS);
const CMAP_CHOICES = {
  metric: ["viridis", "hot", "signed", "cool", "gray", ...SOLID],
  connectivity: ["signed", "hot", "viridis", ...SOLID],
  label: ["labels"],
};

const state = {
  participants: [],
  view: null,
  family: null,
  kind: "inflated",
  meshes: {},
  geometries: new Map(),
  sulc: null,
  sulcScale: 1,
  layers: [],
  seed: null,
  only: null,
  busy: 0,
};

// ---------- status and requests

function setStatus(text, error = false) {
  const element = $("status");
  element.textContent = text || "";
  element.classList.toggle("error", error);
}

async function busy(message, task) {
  state.busy += 1;
  setStatus(message);
  try {
    return await task();
  } catch (error) {
    console.error(error);
    setStatus(error.message || String(error), true);
    throw error;
  } finally {
    state.busy -= 1;
    if (state.busy === 0 && !$("status").classList.contains("error")) setStatus("");
  }
}

function apiURL(path, params = {}) {
  const query = new URLSearchParams(params).toString();
  return query ? `${path}?${query}` : path;
}

async function failure(response) {
  try {
    return new Error((await response.json()).error || response.statusText);
  } catch {
    return new Error(response.statusText);
  }
}

async function getJSON(path, params) {
  const response = await fetch(apiURL(path, params), { credentials: "same-origin" });
  if (!response.ok) throw await failure(response);
  return response.json();
}

async function post(path) {
  const response = await fetch(path, { method: "POST", credentials: "same-origin" });
  if (!response.ok) throw await failure(response);
  return response.json();
}

async function getBinary(path, params) {
  const response = await fetch(apiURL(path, params), { credentials: "same-origin" });
  if (!response.ok) throw await failure(response);
  return { buffer: await response.arrayBuffer(), headers: response.headers };
}

async function getValues(path, params) {
  const { buffer, headers } = await getBinary(path, params);
  const counts = JSON.parse(headers.get("X-Counts"));
  const values = {};
  let offset = 0;
  for (const hemisphere of HEMISPHERES) {
    if (!(hemisphere in counts)) continue;
    values[hemisphere] = new Float32Array(buffer, offset * 4, counts[hemisphere]);
    offset += counts[hemisphere];
  }
  return values;
}

// ---------- rendering

const canvas = $("canvas");
let renderer;
try {
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
} catch (error) {
  setStatus("This browser could not start WebGL, which the viewer requires.", true);
  throw error;
}
renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
renderer.setPixelRatio(window.devicePixelRatio);
const scene = new THREE.Scene();
// Orthographic projection, as in Workbench: sizes do not change with depth.
const CAMERA_DISTANCE_MM = 1000;
const camera = new THREE.OrthographicCamera(-100, 100, 100, -100, 1, 2 * CAMERA_DISTANCE_MM);
let aspect = 1;
let frustumHeight = 200;

function updateFrustum() {
  camera.left = (-frustumHeight / 2) * aspect;
  camera.right = (frustumHeight / 2) * aspect;
  camera.top = frustumHeight / 2;
  camera.bottom = -frustumHeight / 2;
  camera.updateProjectionMatrix();
}
camera.up.set(0, 0, 1);
// Turntable rotation keeps the superior axis vertical on screen. OrbitControls
// reads camera.up once, so camera.up must stay +z while it is active.
const turntable = new OrbitControls(camera, canvas);
turntable.rotateSpeed = 0.8;
turntable.zoomSpeed = 1.2;
turntable.screenSpacePanning = true;
const trackball = new TrackballControls(camera, canvas);
trackball.rotateSpeed = 3;
trackball.zoomSpeed = 1.2;
trackball.panSpeed = 0.6;
trackball.staticMoving = true;
trackball.enabled = false;
let controls = turntable;
scene.add(new THREE.AmbientLight(0xffffff, 0.4 * Math.PI));
const light = new THREE.DirectionalLight(0xffffff, 0.75 * Math.PI);
scene.add(light, light.target);
const material = new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide });
const seedMarker = new THREE.Mesh(
  new THREE.SphereGeometry(1.6, 16, 12),
  new THREE.MeshBasicMaterial({ color: SEED_COLOR }),
);

function applyTheme() {
  const color = getComputedStyle(document.documentElement).getPropertyValue("--canvas").trim();
  scene.background = new THREE.Color(color || "#ffffff");
  invalidate();
}

function resize() {
  const { clientWidth: width, clientHeight: height } = canvas.parentElement;
  renderer.setSize(width, height, false);
  aspect = width / Math.max(height, 1);
  updateFrustum();
  trackball.handleResize();
  invalidate();
}

// Draw only after the camera, geometry, or colors change.
let stale = true;
function invalidate() {
  stale = true;
}
turntable.addEventListener("change", invalidate);
trackball.addEventListener("change", invalidate);

function useTrackball(enabled) {
  const target = controls.target.clone();
  controls.enabled = false;
  controls = enabled ? trackball : turntable;
  controls.target.copy(target);
  controls.enabled = true;
  resetView();
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  if (!stale) return;
  stale = false;
  light.position.copy(camera.position);
  light.target.position.copy(controls.target);
  renderer.render(scene, camera);
}

function visibleMeshes() {
  return HEMISPHERES.map((hemisphere) => state.meshes[hemisphere]).filter((mesh) => mesh && mesh.visible);
}

function frame() {
  const box = new THREE.Box3();
  for (const mesh of visibleMeshes()) box.expandByObject(mesh);
  const sphere = box.getBoundingSphere(new THREE.Sphere());
  return { center: sphere.center, radius: sphere.radius || 100 };
}

const VIEWS = {
  // Lateral and medial views show one hemisphere; with an orthographic camera
  // the far hemisphere would otherwise show around the near one's edges.
  left: { direction: [-1, 0, 0], up: [0, 0, 1], only: "L" },
  right: { direction: [1, 0, 0], up: [0, 0, 1], only: "R" },
  "left-medial": { direction: [1, 0, 0], up: [0, 0, 1], only: "L" },
  "right-medial": { direction: [-1, 0, 0], up: [0, 0, 1], only: "R" },
  // Polar views lean slightly posterior so turntable rotation keeps a defined up.
  dorsal: { direction: [0, -0.002, 1], up: [0, 1, 0] },
  ventral: { direction: [0, -0.002, -1], up: [0, 1, 0] },
  anterior: { direction: [0, 1, 0], up: [0, 0, 1] },
  posterior: { direction: [0, -1, 0], up: [0, 0, 1] },
};

function setView(name) {
  const view = VIEWS[name];
  state.only = view.only || null;
  for (const hemisphere of HEMISPHERES) {
    if (state.meshes[hemisphere]) state.meshes[hemisphere].visible = !state.only || state.only === hemisphere;
  }
  const { center } = frame();
  camera.position.copy(center).addScaledVector(new THREE.Vector3(...view.direction).normalize(), 100);
  if (controls === trackball) camera.up.set(...view.up);
  else camera.up.set(0, 0, 1);
  camera.lookAt(center);
  fitView();
}

// Keep the current viewing direction and size the orthographic view so the
// visible vertices fill the canvas width (or height, if that is the tighter
// limit) up to FIT_MARGIN, centered on their projected extent.
function fitView() {
  const meshes = visibleMeshes();
  if (!meshes.length) return;
  const box = new THREE.Box3();
  for (const mesh of meshes) box.expandByObject(mesh);
  const center = box.getCenter(new THREE.Vector3());
  const back = camera.position.clone().sub(controls.target);
  if (back.lengthSq() === 0) back.set(-1, 0, 0);
  back.normalize();
  camera.position.copy(center).add(back);
  camera.lookAt(center);
  camera.updateMatrixWorld();
  const right = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0);
  const up = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 1);
  const point = new THREE.Vector3();
  let [minRight, maxRight, minUp, maxUp] = [Infinity, -Infinity, Infinity, -Infinity];
  for (const mesh of meshes) {
    const positions = mesh.geometry.getAttribute("position");
    for (let index = 0; index < positions.count; index += 2) {
      point.fromBufferAttribute(positions, index).applyMatrix4(mesh.matrixWorld).sub(center);
      const [r, u] = [point.dot(right), point.dot(up)];
      minRight = Math.min(minRight, r);
      maxRight = Math.max(maxRight, r);
      minUp = Math.min(minUp, u);
      maxUp = Math.max(maxUp, u);
    }
  }
  center.addScaledVector(right, (minRight + maxRight) / 2).addScaledVector(up, (minUp + maxUp) / 2);
  const usable = 1 - 2 * FIT_MARGIN;
  frustumHeight = Math.max((maxRight - minRight) / usable / aspect, (maxUp - minUp) / usable, 1);
  camera.zoom = 1;
  updateFrustum();
  camera.position.copy(center).addScaledVector(back, CAMERA_DISTANCE_MM);
  controls.target.copy(center);
  controls.update();
  invalidate();
}

function resetView() {
  setView($("layout").value === "splayed" ? "posterior" : "left");
}

// "separated" moves the hemispheres apart along x. "splayed" rotates each one
// 90 degrees about the vertical axis so both lateral surfaces face posterior.
// From behind, the left hemisphere then appears in its usual lateral view on
// the left and the right hemisphere on the right, with the base still down.
function layoutHemispheres() {
  const layout = $("layout").value;
  for (const hemisphere of HEMISPHERES) {
    const mesh = state.meshes[hemisphere];
    if (!mesh) continue;
    mesh.geometry.computeBoundingBox();
    const box = mesh.geometry.boundingBox;
    const sign = hemisphere === "L" ? -1 : 1;
    if (layout === "splayed") {
      const rotation = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), (-sign * Math.PI) / 2);
      const center = box.getCenter(new THREE.Vector3());
      const halfWidth = (box.max.y - box.min.y) / 2;
      const destination = new THREE.Vector3(sign * (SPLIT_GAP_MM / 2 + halfWidth), 0, center.z);
      mesh.quaternion.copy(rotation);
      mesh.position.copy(destination.sub(center.applyQuaternion(rotation)));
      continue;
    }
    mesh.quaternion.identity();
    let offset = 0;
    if (layout === "separated") {
      offset = sign < 0 ? -SPLIT_GAP_MM / 2 - Math.max(box.max.x, 0) : SPLIT_GAP_MM / 2 - Math.min(box.min.x, 0);
    }
    mesh.position.set(offset, 0, 0);
  }
  invalidate();
}

async function surfaceGeometry(family, kind, hemisphere) {
  const key = `${family.id}/${kind}/${hemisphere}`;
  if (state.geometries.has(key)) return state.geometries.get(key);
  const { buffer, headers } = await getBinary("/api/mesh", { surface: family.id, hemi: hemisphere, kind });
  const vertices = Number(headers.get("X-Vertices"));
  const triangles = Number(headers.get("X-Triangles"));
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(buffer, 0, vertices * 3), 3));
  geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(buffer, vertices * 12, triangles * 3), 1));
  geometry.computeVertexNormals();
  geometry.setAttribute("color", new THREE.BufferAttribute(new Float32Array(vertices * 3), 3));
  state.geometries.set(key, geometry);
  return geometry;
}

async function showSurfaces({ reframe }) {
  const family = state.family;
  if (!family) return;
  await busy(`Loading ${state.kind} surfaces...`, async () => {
    const geometries = await Promise.all(HEMISPHERES.map((hemisphere) => surfaceGeometry(family, state.kind, hemisphere)));
    HEMISPHERES.forEach((hemisphere, index) => {
      let mesh = state.meshes[hemisphere];
      if (!mesh) {
        mesh = new THREE.Mesh(geometries[index], material);
        mesh.userData.hemisphere = hemisphere;
        state.meshes[hemisphere] = mesh;
        scene.add(mesh);
      }
      mesh.geometry = geometries[index];
      mesh.visible = !state.only || state.only === hemisphere;
    });
    layoutHemispheres();
    placeSeedMarker();
    recolor();
    if (reframe) resetView();
    else fitView();
  });
}

function clearSurfaces() {
  for (const hemisphere of HEMISPHERES) {
    if (state.meshes[hemisphere]) scene.remove(state.meshes[hemisphere]);
  }
  for (const geometry of state.geometries.values()) geometry.dispose();
  state.geometries.clear();
  state.meshes = {};
}

// ---------- color

function sample(stops, t) {
  const position = Math.min(Math.max(t, 0), 1) * (stops.length - 1);
  const index = Math.min(Math.floor(position), stops.length - 2);
  const fraction = position - index;
  const [a, b] = [stops[index], stops[index + 1]];
  return [a[0] + (b[0] - a[0]) * fraction, a[1] + (b[1] - a[1]) * fraction, a[2] + (b[2] - a[2]) * fraction];
}

function finite(value) {
  return value !== "" && value !== null && Number.isFinite(Number(value)) ? Number(value) : null;
}

// Return [r, g, b, alpha] for one value, or null when the vertex stays uncolored.
function colorFor(layer, value) {
  if (!Number.isFinite(value)) return null;
  const opacity = layer.opacity;
  if (layer.cmap === "labels") {
    const entry = layer.labels && layer.labels[String(Math.round(value))];
    if (!entry || entry.rgba[3] === 0 || Math.round(value) === 0) return null;
    return [entry.rgba[0], entry.rgba[1], entry.rgba[2], entry.rgba[3] * opacity];
  }
  const threshold = finite(layer.threshold);
  const max = finite(layer.max) ?? 1;
  if (layer.cmap === "signed") {
    // Max saturates positive values and min saturates negative ones; an empty
    // min mirrors max. The threshold hides magnitudes below it on both sides.
    const magnitude = Math.abs(value);
    const floor = threshold ?? 0;
    if (magnitude < floor || magnitude === 0) return null;
    const limit = value > 0 ? Math.abs(max) : Math.abs(finite(layer.min) ?? -max);
    const t = (magnitude - floor) / Math.max(limit - floor, 1e-12);
    const rgb = sample(value > 0 ? COLORMAPS.hot : COLORMAPS.negative, t);
    return [...rgb, opacity];
  }
  if (threshold !== null && value < threshold) return null;
  const min = finite(layer.min) ?? 0;
  const t = Math.min(Math.max((value - min) / Math.max(max - min, 1e-12), 0), 1);
  if (layer.cmap in SOLID_COLORS) return t > 0 ? [...SOLID_COLORS[layer.cmap], t * opacity] : null;
  return [...sample(COLORMAPS[layer.cmap], t), opacity];
}

// FreeSurfer sulc is positive in sulci, which are drawn darker.
function sulcGray(value, shading) {
  if (!Number.isFinite(value)) return 0.62;
  if (shading === "binary") return value > 0 ? 0.42 : 0.68;
  const t = Math.min(Math.max(value / state.sulcScale, -1), 1);
  return 0.56 - 0.24 * t;
}

function recolor() {
  const additive = $("additive").checked;
  const drawn = state.layers.filter((layer) => layer.visible && layer.values).slice().reverse();
  for (const hemisphere of HEMISPHERES) {
    const mesh = state.meshes[hemisphere];
    if (!mesh) continue;
    const colors = mesh.geometry.getAttribute("color");
    const array = colors.array;
    const count = colors.count;
    const shading = $("shading").value;
    const sulc = shading !== "none" && state.sulc?.[hemisphere]?.length === count ? state.sulc[hemisphere] : null;
    const sources = drawn
      .map((layer) => [layer, layer.values[hemisphere]])
      .filter(([layer, values]) => {
        const matches = values && values.length === count;
        if (values && !matches) layer.mismatch = true;
        return matches;
      });
    for (let vertex = 0; vertex < count; vertex += 1) {
      const gray = sulc ? sulcGray(sulc[vertex], shading) : 0.62;
      let r = gray;
      let g = gray;
      let b = gray;
      if (additive) {
        // Sum the layers' weighted colors among themselves, then lay the sum over
        // the shading with their combined opacity. The shading never enters the sum.
        let [sumR, sumG, sumB, coverage] = [0, 0, 0, 0];
        for (const [layer, values] of sources) {
          const color = colorFor(layer, values[vertex]);
          if (!color) continue;
          const alpha = color[3];
          sumR += color[0] * alpha;
          sumG += color[1] * alpha;
          sumB += color[2] * alpha;
          coverage += alpha;
        }
        const keep = 1 - Math.min(coverage, 1);
        r = gray * keep + sumR;
        g = gray * keep + sumG;
        b = gray * keep + sumB;
      } else {
        for (const [layer, values] of sources) {
          const color = colorFor(layer, values[vertex]);
          if (!color) continue;
          const alpha = color[3];
          r = r * (1 - alpha) + color[0] * alpha;
          g = g * (1 - alpha) + color[1] * alpha;
          b = b * (1 - alpha) + color[2] * alpha;
        }
      }
      array[vertex * 3] = Math.min(r, 1);
      array[vertex * 3 + 1] = Math.min(g, 1);
      array[vertex * 3 + 2] = Math.min(b, 1);
    }
    colors.needsUpdate = true;
  }
  invalidate();
  for (const layer of state.layers) {
    if (layer.mismatch) layer.hint.textContent = "Vertex counts differ from the displayed surfaces.";
  }
}

// Summarize a sample of finite values. Exact zeros are counted separately because
// surface metrics such as thickness use zero for the medial wall.
function summarize(values) {
  const sample = [];
  let zeros = 0;
  let total = 0;
  for (const hemisphere of HEMISPHERES) {
    const array = values[hemisphere];
    if (!array) continue;
    for (let index = 0; index < array.length; index += 7) {
      const value = array[index];
      if (!Number.isFinite(value)) continue;
      total += 1;
      if (value === 0) zeros += 1;
      else sample.push(value);
    }
  }
  sample.sort((a, b) => a - b);
  const quantile = (fraction) => (sample.length ? sample[Math.min(sample.length - 1, Math.floor(fraction * sample.length))] : 0);
  return { quantile, zeroFraction: total ? zeros / total : 0 };
}

// ---------- layers

function compatible(layer, family) {
  if (!family) return false;
  return Object.entries(layer.vertex_counts).every(([hemisphere, count]) => family.vertex_counts[hemisphere] === count);
}

function describe(layer) {
  const suffix = layer.kind === "connectivity" ? " (seed correlation)" : layer.maps.length > 1 ? ` (${layer.maps.length} maps)` : "";
  return `${layer.module}: ${layer.label}${suffix}`;
}

function populateAvailable() {
  const select = $("available");
  select.replaceChildren();
  const layers = (state.view?.layers || []).filter((layer) => compatible(layer, state.family));
  for (const layer of layers) select.append(new Option(describe(layer), layer.id));
  if (!layers.length) select.append(new Option("No layers match these surfaces", ""));
  $("add").disabled = !layers.length;
}

function syncInputs(layer) {
  const card = layer.card;
  card.querySelector(".min").value = layer.min ?? "";
  card.querySelector(".max").value = layer.max ?? "";
  card.querySelector(".threshold").value = layer.threshold ?? "";
  card.querySelector(".min").disabled = layer.cmap === "labels";
  card.querySelector(".max").disabled = layer.cmap === "labels";
  card.querySelector(".threshold").disabled = layer.cmap === "labels";
}

function defaultScale(layer) {
  if (layer.meta.kind === "label") return;
  if (layer.meta.kind === "connectivity") {
    Object.assign(layer, { cmap: "signed", max: 0.6, threshold: null, min: -0.6 });
    return;
  }
  const { quantile, zeroFraction } = summarize(layer.values);
  const low = quantile(0.02);
  const high = quantile(0.98);
  const round = (value) => Number(value.toPrecision(3));
  if (low < 0 && high > 0) {
    const magnitude = Math.max(Math.abs(low), Math.abs(high));
    Object.assign(layer, { cmap: "signed", min: -round(magnitude), max: round(magnitude), threshold: null });
  } else {
    // Hide a block of exact zeros, such as a medial wall, below the lowest real value.
    const threshold = zeroFraction > 0.01 && low > 0 ? round(quantile(0)) : null;
    Object.assign(layer, { min: round(low), max: round(high), threshold });
  }
}

async function loadLayer(layer, { rescale }) {
  layer.card.classList.add("loading");
  layer.mismatch = false;
  try {
    if (layer.meta.kind === "connectivity") {
      if (!state.seed) {
        layer.values = null;
        layer.hint.textContent = "Click the cortex to place a seed.";
        const size = (layer.meta.size_bytes / 1e9).toFixed(1);
        await busy(`Reading ${layer.meta.label} (${size} GB)...`, () => getJSON("/api/prepare", { layer: layer.meta.id }));
        return;
      }
      const { hemisphere, vertex } = state.seed;
      layer.values = await busy("Computing seed correlations...", () =>
        getValues("/api/seed", { layer: layer.meta.id, hemi: hemisphere, vertex }),
      );
      layer.hint.textContent = `Seed: ${hemisphere} vertex ${vertex}`;
    } else {
      const params = { layer: layer.meta.id, map: layer.map };
      layer.values = await busy(`Loading ${layer.meta.label}...`, () => getValues("/api/layer", params));
      if (layer.meta.kind === "label") layer.labels = await getJSON("/api/labels", params);
    }
    if (rescale) {
      defaultScale(layer);
      layer.card.querySelector(".cmap").value = layer.cmap;
      syncInputs(layer);
    }
  } finally {
    layer.card.classList.remove("loading");
    recolor();
  }
}

// Open any displayable file by its path relative to the project's BIDS directory.
async function openFileLayer() {
  const path = $("file-path").value.trim();
  if (!path) return;
  const meta = await busy(`Opening ${path}...`, () =>
    getJSON("/api/file", { project: $("project").value, path, branch: branch() }),
  );
  if (!compatible(meta, state.family)) {
    const counts = Object.entries(meta.vertex_counts).map(([h, n]) => `${h} ${n}`).join(", ");
    setStatus(`${path} has ${counts} vertices, which do not match the displayed surfaces.`, true);
    return;
  }
  addLayer(meta);
}

function addLayer(meta) {
  const card = $("layer-template").content.firstElementChild.cloneNode(true);
  const layer = {
    meta,
    card,
    hint: card.querySelector(".hint"),
    map: 0,
    values: null,
    labels: null,
    cmap: CMAP_CHOICES[meta.kind][0],
    min: null,
    max: null,
    threshold: null,
    opacity: meta.kind === "label" ? 0.85 : 1,
    visible: true,
  };
  card.querySelector(".name").textContent = describe(meta);
  const mapSelect = card.querySelector(".map");
  meta.maps.forEach((name, index) => mapSelect.append(new Option(name || `map ${index + 1}`, index)));
  card.querySelector(".map-row").hidden = meta.maps.length < 2;
  const cmapSelect = card.querySelector(".cmap");
  for (const name of CMAP_CHOICES[meta.kind]) cmapSelect.append(new Option(name, name));
  card.querySelector(".opacity").value = layer.opacity;

  mapSelect.addEventListener("change", () => {
    layer.map = Number(mapSelect.value);
    loadLayer(layer, { rescale: true }).catch(() => {});
  });
  cmapSelect.addEventListener("change", () => {
    layer.cmap = cmapSelect.value;
    syncInputs(layer);
    recolor();
  });
  for (const field of ["min", "max", "threshold"]) {
    card.querySelector(`.${field}`).addEventListener("change", (event) => {
      layer[field] = finite(event.target.value);
      recolor();
    });
  }
  card.querySelector(".opacity").addEventListener("input", (event) => {
    layer.opacity = Number(event.target.value);
    recolor();
  });
  card.querySelector(".visible").addEventListener("change", (event) => {
    layer.visible = event.target.checked;
    recolor();
  });
  card.querySelector(".remove").addEventListener("click", () => {
    state.layers = state.layers.filter((item) => item !== layer);
    card.remove();
    recolor();
  });

  if (meta.kind === "connectivity") {
    defaultScale(layer);
    cmapSelect.value = layer.cmap;
  }
  state.layers.unshift(layer);
  $("layers").prepend(card);
  syncInputs(layer);
  loadLayer(layer, { rescale: true }).catch(() => {});
}

// ---------- picking

function pick(event) {
  const bounds = canvas.getBoundingClientRect();
  const pointer = new THREE.Vector2(
    ((event.clientX - bounds.left) / bounds.width) * 2 - 1,
    -((event.clientY - bounds.top) / bounds.height) * 2 + 1,
  );
  const raycaster = new THREE.Raycaster();
  raycaster.setFromCamera(pointer, camera);
  const [hit] = raycaster.intersectObjects(visibleMeshes(), false);
  if (!hit) return;
  const mesh = hit.object;
  const local = mesh.worldToLocal(hit.point.clone());
  const positions = mesh.geometry.getAttribute("position");
  let best = hit.face.a;
  let bestDistance = Infinity;
  for (const candidate of [hit.face.a, hit.face.b, hit.face.c]) {
    const distance = local.distanceToSquared(new THREE.Vector3().fromBufferAttribute(positions, candidate));
    if (distance < bestDistance) [best, bestDistance] = [candidate, distance];
  }
  selectVertex(mesh.userData.hemisphere, best);
}

function placeSeedMarker() {
  seedMarker.removeFromParent();
  if (!state.seed) return;
  const mesh = state.meshes[state.seed.hemisphere];
  if (!mesh) return;
  seedMarker.position.fromBufferAttribute(mesh.geometry.getAttribute("position"), state.seed.vertex);
  mesh.add(seedMarker);
  invalidate();
}

function showReadout() {
  const { hemisphere, vertex } = state.seed;
  const position = new THREE.Vector3().fromBufferAttribute(state.meshes[hemisphere].geometry.getAttribute("position"), vertex);
  const rows = [
    ["Hemisphere", hemisphere],
    ["Vertex", vertex],
    [`${state.kind} (mm)`, position.toArray().map((value) => value.toFixed(1)).join(", ")],
  ];
  for (const layer of state.layers) {
    const value = layer.values?.[hemisphere]?.[vertex];
    if (value === undefined) continue;
    let text = Number.isFinite(value) ? Number(value.toPrecision(4)).toString() : "no data";
    if (layer.labels && Number.isFinite(value)) text = layer.labels[String(Math.round(value))]?.name ?? text;
    rows.push([layer.meta.label, text]);
  }
  $("readout-values").replaceChildren(
    ...rows.flatMap(([term, value]) => {
      const dt = document.createElement("dt");
      const dd = document.createElement("dd");
      dt.textContent = term;
      dd.textContent = value;
      return [dt, dd];
    }),
  );
  $("readout").hidden = false;
}

async function selectVertex(hemisphere, vertex) {
  state.seed = { hemisphere, vertex };
  placeSeedMarker();
  showReadout();
  const seeded = state.layers.filter((layer) => layer.meta.kind === "connectivity");
  await Promise.allSettled(seeded.map((layer) => loadLayer(layer, { rescale: false })));
  showReadout();
}

// ---------- selection flow

function fillSelect(select, values, selected) {
  select.replaceChildren(...values.map((value) => new Option(value, value)));
  if (selected && values.includes(selected)) select.value = selected;
}

function refreshParticipants(preferred) {
  const project = $("project").value;
  const participants = state.participants.filter((item) => item.project === project).map((item) => item.participant);
  fillSelect($("participant"), participants, preferred);
}

function chooseFamily(family) {
  state.family = family;
  const kinds = KINDS.filter((kind) => family.kinds.includes(kind));
  if (!kinds.includes(state.kind)) state.kind = kinds[0];
  $("kind").replaceChildren(
    ...kinds.map((kind) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = kind;
      button.classList.toggle("active", kind === state.kind);
      button.addEventListener("click", () => {
        state.kind = kind;
        for (const item of $("kind").children) item.classList.toggle("active", item === button);
        showSurfaces({ reframe: false }).catch(() => {});
      });
      return button;
    }),
  );
  populateAvailable();
}

async function loadSulc() {
  state.sulc = null;
  const sulc = state.view.layers.find((layer) => layer.module === "anat" && /\bsulc\b/.test(layer.label) && compatible(layer, state.family));
  if (sulc) {
    state.sulc = await getValues("/api/layer", { layer: sulc.id, map: 0 });
    const { quantile } = summarize(state.sulc);
    state.sulcScale = Math.max(Math.abs(quantile(0.02)), Math.abs(quantile(0.98))) || 1;
  }
}

async function loadParticipant() {
  const project = $("project").value;
  const participant = $("participant").value;
  if (!project || !participant) return;
  state.layers = [];
  state.seed = null;
  $("layers").replaceChildren();
  $("readout").hidden = true;
  seedMarker.removeFromParent();
  clearSurfaces();
  state.view = await busy(`Indexing ${project} sub-${participant}...`, () => getJSON("/api/view", { project, participant, branch: branch() }));
  const families = state.view.surfaces;
  $("family").replaceChildren(...families.map((family, index) => new Option(family.label, index)));
  if (!families.length) {
    setStatus("No complete display-surface family is available for this participant.", true);
    return;
  }
  const preferred = Math.max(0, families.findIndex((family) => family.module === "anat" && family.space === "fsnative"));
  $("family").value = preferred;
  chooseFamily(families[preferred]);
  await loadSulc();
  await showSurfaces({ reframe: true });
  const params = new URLSearchParams(window.location.search);
  params.set("branch", branch());
  params.set("project", project);
  params.set("participant", participant);
  window.history.replaceState(null, "", `?${params}`);
}

function bindControls() {
  $("project").addEventListener("change", () => {
    refreshParticipants();
    loadParticipant().catch(() => {});
  });
  $("participant").addEventListener("change", () => loadParticipant().catch(() => {}));
  $("family").addEventListener("change", async () => {
    clearSurfaces();
    chooseFamily(state.view.surfaces[Number($("family").value)]);
    await loadSulc();
    await showSurfaces({ reframe: true });
  });
  $("layout").addEventListener("change", () => {
    layoutHemispheres();
    resetView();
  });
  $("reset-view").addEventListener("click", resetView);
  $("trackball").addEventListener("change", (event) => useTrackball(event.target.checked));
  $("additive").addEventListener("change", recolor);
  $("shading").addEventListener("change", recolor);
  $("refresh").addEventListener("click", () => refreshData().catch(() => {}));
  $("restart").addEventListener("click", () => restartServer().catch(() => {}));
  const openFile = () => openFileLayer().catch(() => {});
  $("open-file").addEventListener("click", openFile);
  $("file-path").addEventListener("keydown", (event) => {
    if (event.key === "Enter") openFile();
  });
  $("add").addEventListener("click", () => {
    const meta = state.view.layers.find((layer) => layer.id === $("available").value);
    if (meta) addLayer(meta);
  });
  for (const button of document.querySelectorAll("[data-view]")) {
    button.addEventListener("click", () => setView(button.dataset.view));
  }
  $("collapse").addEventListener("click", () => {
    $("panel").classList.add("collapsed");
    $("expand").hidden = false;
    resize();
  });
  $("expand").addEventListener("click", () => {
    $("panel").classList.remove("collapsed");
    $("expand").hidden = true;
    resize();
  });
  let down = null;
  canvas.addEventListener("pointerdown", (event) => {
    down = [event.clientX, event.clientY];
  });
  canvas.addEventListener("pointerup", (event) => {
    if (down && Math.hypot(event.clientX - down[0], event.clientY - down[1]) < 4 && event.button === 0) pick(event);
    down = null;
  });
  new ResizeObserver(resize).observe(canvas.parentElement);
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", applyTheme);
}

// Reread the registry, then reload the shown participant because the server
// issues new identifiers for its files.
async function refreshData() {
  await busy("Rereading the registry...", () => post(apiURL("/api/refresh", { branch: branch() })));
  await loadParticipantList();
  await loadParticipant();
}

async function restartServer() {
  await busy("Restarting the server...", async () => {
    await post("/api/restart");
    const deadline = Date.now() + 120000;
    while (Date.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, 1000));
      try {
        const response = await fetch("/api/participants", { credentials: "same-origin" });
        if (response.ok) {
          window.location.reload();
          return;
        }
      } catch {
        // The server is still starting.
      }
    }
    throw new Error("The server did not come back within two minutes; check the nro viz terminal.");
  });
}

async function loadParticipantList() {
  const project = $("project").value;
  const participant = $("participant").value;
  state.participants = await getJSON("/api/participants", { branch: branch() });
  const projects = [...new Set(state.participants.map((item) => item.project))];
  fillSelect($("project"), projects, project);
  refreshParticipants(participant);
}

function branch() {
  return $("branch").value || "";
}

// List the branch's participants and select the requested project and participant.
async function loadBranch(preferredProject, preferredParticipant) {
  state.participants = await busy("Reading registry...", () => getJSON("/api/participants", { branch: branch() }));
  const projects = [...new Set(state.participants.map((item) => item.project))];
  fillSelect($("project"), projects, preferredProject);
  refreshParticipants(preferredParticipant);
  if (!state.participants.length) {
    setStatus(`Branch ${branch()} has no completed derivatives.`, true);
    return;
  }
  await loadParticipant();
}

async function main() {
  applyTheme();
  bindControls();
  resize();
  animate();
  const params = new URLSearchParams(window.location.search);
  const { branches, current } = await busy("Listing branches...", () => getJSON("/api/branches"));
  fillSelect($("branch"), branches, params.get("branch") || current);
  // A development branch often has no derivatives yet; start on main instead.
  if (!params.get("branch") && branches.includes("main")) {
    const listed = await getJSON("/api/participants", { branch: branch() });
    if (!listed.length) $("branch").value = "main";
  }
  $("branch").addEventListener("change", () => loadBranch().catch(() => {}));
  await loadBranch(params.get("project"), params.get("participant"));
}

main().catch(() => {});
