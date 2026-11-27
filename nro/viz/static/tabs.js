// Switch between the default surface viewer and analysis viewer pages.
//
// Each analysis viewer runs in its own frame, created when its tab is first
// opened, so switching tabs keeps the state of every view.
const bar = document.getElementById("tabs");
const viewer = document.getElementById("viewer");
const frames = document.getElementById("frames");

function select(name) {
  for (const button of bar.querySelectorAll("button")) {
    button.classList.toggle("active", button.dataset.view === name);
  }
  viewer.hidden = name !== "";
  for (const frame of frames.children) frame.hidden = frame.dataset.view !== name;
  frames.hidden = name === "";
}

function open(entry) {
  let frame = [...frames.children].find((item) => item.dataset.view === entry.name);
  if (!frame) {
    frame = document.createElement("iframe");
    frame.dataset.view = entry.name;
    frame.title = entry.name;
    frame.src = entry.url;
    frames.append(frame);
  }
  select(entry.name);
}

function button(label, name, title, action) {
  const element = document.createElement("button");
  element.type = "button";
  element.textContent = label;
  element.title = title;
  element.dataset.view = name;
  element.addEventListener("click", action);
  return element;
}

async function load() {
  let entries = [];
  try {
    const response = await fetch("/api/analyses", { credentials: "same-origin" });
    if (response.ok) entries = await response.json();
  } catch {
    // The default viewer works without analysis viewers.
  }
  bar.replaceChildren(button("Surfaces", "", "Participant surfaces and derivative layers", () => select("")));
  for (const entry of entries) {
    bar.append(button(entry.name, entry.name, `${entry.name} at ${entry.commit.slice(0, 12)}`, () => open(entry)));
  }
  bar.hidden = entries.length === 0;
  select("");
}

load();
