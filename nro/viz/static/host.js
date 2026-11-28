// Helpers for analysis viewer pages served by `nro viz`.
//
// A viewer page imports this module from /static/host.js. Requests use the
// viewer's session cookie, so a page can read only what the server exposes:
// its analysis's published outputs and the template meshes.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

export const HEMISPHERES = ["L", "R"];

// The analysis name ("STUDY/ANALYSIS") is the first two path components after /analyses/.
export function analysisName() {
  const parts = window.location.pathname.split("/").filter(Boolean);
  return `${decodeURIComponent(parts[1])}/${decodeURIComponent(parts[2])}`;
}

function url(path, params = {}) {
  const query = new URLSearchParams(
    Object.entries(params).filter(([, value]) => value !== null && value !== undefined),
  ).toString();
  return query ? `${path}?${query}` : path;
}

async function request(path, params) {
  const response = await fetch(url(path, params), { credentials: "same-origin" });
  if (!response.ok) {
    let message = response.statusText;
    try {
      message = (await response.json()).error || message;
    } catch {
      // Keep the status text.
    }
    throw new Error(message);
  }
  return response;
}

// List this analysis's published outputs: {project, participant, slot, file, size, ...}.
export async function outputs(analysis = analysisName()) {
  return (await request("/api/analysis/outputs", { analysis })).json();
}

// Fetch one published output as JSON, text, or an ArrayBuffer.
export async function outputFile(entry, { as = "buffer", analysis = analysisName() } = {}) {
  const response = await request("/api/analysis/file", {
    analysis,
    project: entry.project,
    participant: entry.participant,
    file: entry.file,
  });
  if (as === "json") return response.json();
  if (as === "text") return response.text();
  return response.arrayBuffer();
}

// Parse a NumPy .npy buffer (format 1.0-3.0, little-endian float or int, C order).
export function parseNpy(buffer) {
  const bytes = new Uint8Array(buffer);
  const major = bytes[6];
  const view = new DataView(buffer);
  const headerLength = major === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  const start = major === 1 ? 10 : 12;
  const header = new TextDecoder().decode(bytes.subarray(start, start + headerLength));
  const descr = /'descr':\s*'([^']+)'/.exec(header)[1];
  const fortran = /'fortran_order':\s*True/.test(header);
  const shape = /'shape':\s*\(([^)]*)\)/
    .exec(header)[1]
    .split(",")
    .map((value) => value.trim())
    .filter(Boolean)
    .map(Number);
  if (fortran) throw new Error("Fortran-ordered .npy arrays are not supported");
  const types = { "<f4": Float32Array, "<f8": Float64Array, "<i4": Int32Array, "|u1": Uint8Array, "<i8": BigInt64Array };
  const Type = types[descr];
  if (!Type) throw new Error(`Unsupported .npy dtype ${descr}`);
  const offset = start + headerLength;
  const count = shape.reduce((a, b) => a * b, 1);
  const data = new Type(buffer.slice(offset, offset + count * Type.BYTES_PER_ELEMENT));
  return { shape, data };
}

// Fetch template geometry. space is "fsaverage6" or "fsaverage"; kind is
// midthickness, pial, white, or sphere.
export async function templateMesh(space, hemisphere, kind = "midthickness") {
  const response = await request("/api/template-mesh", { space, hemi: hemisphere, kind });
  const buffer = await response.arrayBuffer();
  const vertices = Number(response.headers.get("X-Vertices"));
  const triangles = Number(response.headers.get("X-Triangles"));
  return {
    coordinates: new Float32Array(buffer, 0, vertices * 3),
    triangles: new Uint32Array(buffer, vertices * 12, triangles * 3),
  };
}

// Smooth coordinates toward their neighbors to approximate an inflated surface.
export function inflate(mesh, iterations = 40, step = 0.5) {
  const { coordinates, triangles } = mesh;
  const count = coordinates.length / 3;
  const neighbors = Array.from({ length: count }, () => new Set());
  for (let index = 0; index < triangles.length; index += 3) {
    const [a, b, c] = [triangles[index], triangles[index + 1], triangles[index + 2]];
    neighbors[a].add(b).add(c);
    neighbors[b].add(a).add(c);
    neighbors[c].add(a).add(b);
  }
  const lists = neighbors.map((set) => Uint32Array.from(set));
  let current = Float32Array.from(coordinates);
  let next = new Float32Array(current.length);
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    for (let vertex = 0; vertex < count; vertex += 1) {
      const list = lists[vertex];
      let x = 0;
      let y = 0;
      let z = 0;
      for (const other of list) {
        x += current[3 * other];
        y += current[3 * other + 1];
        z += current[3 * other + 2];
      }
      const n = Math.max(list.length, 1);
      for (const [axis, sum] of [[0, x], [1, y], [2, z]]) {
        const value = current[3 * vertex + axis];
        next[3 * vertex + axis] = value + step * (sum / n - value);
      }
    }
    [current, next] = [next, current];
  }
  return { coordinates: current, triangles };
}

export const COLORMAPS = {
  viridis: [[0.267, 0.005, 0.329], [0.229, 0.322, 0.546], [0.128, 0.567, 0.551], [0.369, 0.789, 0.383], [0.993, 0.906, 0.144]],
  hot: [[0.55, 0, 0], [1, 0, 0], [1, 0.55, 0], [1, 1, 0], [1, 1, 0.8]],
  negative: [[0, 0, 0.6], [0, 0.3, 1], [0, 0.8, 1], [0.6, 1, 1]],
};

export function sample(stops, t) {
  const position = Math.min(Math.max(t, 0), 1) * (stops.length - 1);
  const index = Math.min(Math.floor(position), stops.length - 2);
  const fraction = position - index;
  const [a, b] = [stops[index], stops[index + 1]];
  return [a[0] + (b[0] - a[0]) * fraction, a[1] + (b[1] - a[1]) * fraction, a[2] + (b[2] - a[2]) * fraction];
}

// Return [r, g, b] for a value, or null to show the base surface.
// "signed" saturates at ±max and hides magnitudes below threshold, as the
// default viewer does; "viridis" maps [min, max].
export function color(value, { cmap = "signed", min = 0, max = 1, threshold = 0 } = {}) {
  if (!Number.isFinite(value)) return null;
  if (cmap === "signed") {
    const magnitude = Math.abs(value);
    if (magnitude < threshold || magnitude === 0) return null;
    const limit = value > 0 ? Math.abs(max) : Math.abs(min || max);
    return sample(value > 0 ? COLORMAPS.hot : COLORMAPS.negative, (magnitude - threshold) / Math.max(limit - threshold, 1e-12));
  }
  if (value < threshold) return null;
  return sample(COLORMAPS[cmap] || COLORMAPS.viridis, (value - min) / Math.max(max - min, 1e-12));
}

function cssColor(name, fallback) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
}

// Draw both hemispheres side by side with an orthographic camera.
// "lateral" shows both lateral surfaces and "medial" both medial surfaces.
export class SurfaceView {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
    this.renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.scene = new THREE.Scene();
    this.camera = new THREE.OrthographicCamera(-100, 100, 100, -100, 1, 4000);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.screenSpacePanning = true;
    this.controls.addEventListener("change", () => this.render());
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.4 * Math.PI));
    this.light = new THREE.DirectionalLight(0xffffff, 0.75 * Math.PI);
    this.scene.add(this.light, this.light.target);
    this.material = new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide });
    this.meshes = {};
    this.side = "lateral";
    this.base = 0.62;
    new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => this.render());
  }

  setMeshes(meshes) {
    for (const mesh of Object.values(this.meshes)) {
      this.scene.remove(mesh);
      mesh.geometry.dispose();
    }
    this.meshes = {};
    for (const hemisphere of HEMISPHERES) {
      const source = meshes[hemisphere];
      if (!source) continue;
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(Float32Array.from(source.coordinates), 3));
      geometry.setIndex(new THREE.BufferAttribute(Uint32Array.from(source.triangles), 1));
      geometry.computeVertexNormals();
      const count = source.coordinates.length / 3;
      geometry.setAttribute("color", new THREE.BufferAttribute(new Float32Array(count * 3).fill(this.base), 3));
      geometry.computeBoundingBox();
      this.meshes[hemisphere] = new THREE.Mesh(geometry, this.material);
      this.scene.add(this.meshes[hemisphere]);
    }
    this.layout();
  }

  setSide(side) {
    this.side = side;
    this.layout();
  }

  // Place hemispheres so the camera on -x sees both requested surfaces.
  layout() {
    const extents = {};
    for (const [hemisphere, mesh] of Object.entries(this.meshes)) {
      // Left lateral faces -x. The right hemisphere turns 180 degrees for its
      // lateral surface, and each turns again for the medial view.
      const turn = (hemisphere === "R") !== (this.side === "medial");
      mesh.rotation.set(0, 0, turn ? Math.PI : 0);
      mesh.position.set(0, 0, 0);
      mesh.updateMatrixWorld();
      const turned = new THREE.Box3().setFromObject(mesh);
      extents[hemisphere] = { box: turned };
    }
    // Looking along +x with z up, screen left is +y: the left hemisphere goes
    // on the +y side of a small gap and the right one on the -y side.
    const gap = 8;
    for (const [hemisphere, mesh] of Object.entries(this.meshes)) {
      const { box } = extents[hemisphere];
      const width = box.max.y - box.min.y;
      const center = box.getCenter(new THREE.Vector3());
      const target = (hemisphere === "L" ? 1 : -1) * (gap / 2 + width / 2);
      mesh.position.set(-center.x, target - center.y, -center.z);
    }
    this.fit();
  }

  fit() {
    const box = new THREE.Box3();
    for (const mesh of Object.values(this.meshes)) box.expandByObject(mesh);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    this.extent = { width: Math.max(size.y, 1), height: Math.max(size.z, 1) };
    this.controls.target.copy(center);
    this.camera.position.set(center.x - 1500, center.y, center.z);
    this.camera.up.set(0, 0, 1);
    // OrbitControls aims the camera at its target only in update(); without
    // it the camera keeps facing -z and shows nothing.
    this.camera.lookAt(center);
    this.controls.update();
    this.resize();
  }

  resize() {
    const parent = this.canvas.parentElement;
    const width = parent.clientWidth;
    const height = Math.max(parent.clientHeight, 1);
    this.renderer.setSize(width, height, false);
    const aspect = width / height;
    const extent = this.extent || { width: 200, height: 200 };
    let frustumHeight = extent.height * 1.12;
    if (frustumHeight * aspect < extent.width * 1.12) frustumHeight = (extent.width * 1.12) / aspect;
    Object.assign(this.camera, {
      left: (-frustumHeight / 2) * aspect,
      right: (frustumHeight / 2) * aspect,
      top: frustumHeight / 2,
      bottom: -frustumHeight / 2,
    });
    this.camera.updateProjectionMatrix();
    this.render();
  }

  // values maps "L" and "R" to arrays with one value per vertex.
  setValues(values, scale) {
    for (const [hemisphere, mesh] of Object.entries(this.meshes)) {
      const colors = mesh.geometry.getAttribute("color");
      const source = values[hemisphere];
      for (let vertex = 0; vertex < colors.count; vertex += 1) {
        const rgb = source ? color(source[vertex], scale) : null;
        if (rgb) colors.setXYZ(vertex, rgb[0], rgb[1], rgb[2]);
        else colors.setXYZ(vertex, this.base, this.base, this.base);
      }
      colors.needsUpdate = true;
    }
    this.render();
  }

  render() {
    this.scene.background = new THREE.Color(cssColor("--canvas", "#ffffff"));
    this.light.position.copy(this.camera.position);
    this.light.target.position.copy(this.controls.target);
    this.renderer.render(this.scene, this.camera);
  }

  png() {
    this.render();
    return this.canvas.toDataURL("image/png");
  }
}
