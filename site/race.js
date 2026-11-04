/* The race: one ticket, three recorded runs on the same model and GPU, played together. The clocks count the seconds
   each run really took. Only a run's total time was recorded, so its steps, in their real order, are spaced evenly.
   All model text goes in as plain text. */
(function () {
  const box = document.getElementById("race");
  const SPEED = 10;
  const quiet = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const NAMES = {
    none: ["Muse Glimmer", "Out of the box. No rule book."],
    checked: ["Reading the written rules", "The rule book in its prompt, on every ticket."],
    "p-checked": ["workalike LoRA", "No rule book in its prompt."],
  };
  const NOTES = {
    none: "Nearly the full 5%, not the person's way.",
    checked: "Close to the person's way.",
    "p-checked": "Close to the person's way.",
  };

  function h(tag, attrs, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === "class") el.className = v; else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else if (v !== false && v != null) el.setAttribute(k, v);
    }
    for (const kid of kids.flat()) if (kid != null) el.append(kid);
    return el;
  }

  function card(run) {
    const [name, sub] = NAMES[run.arm];
    const clock = h("p", { class: "clock", role: "timer" }, "0s"), steps = h("ol", { class: "steps" }), end = h("div", { class: "result" });
    const el = h("section", { class: `arm racer drawn${run.arm === "p-checked" ? " lora" : ""}` },
      h("div", { class: "racer-head" }, h("h3", {}, name), h("p", { class: "sub" }, sub)), clock, end, steps);
    return { run, el, clock, steps, end, shown: 0, done: false };
  }
  function finish(c) {
    c.done = true;
    c.el.classList.add("done");
    c.end.replaceChildren(h("p", { class: "off" }, `${c.run.off}% off`), h("p", { class: "result-note" }, NOTES[c.run.arm], " ", `${c.run.tokens.toLocaleString("en-US")} tokens written.`),
      h("details", {}, h("summary", {}, "Read the draft"), h("blockquote", { class: "draft" }, c.run.draft), h("p", { class: "held" }, "Held for approval. Nothing was sent.")));
  }

  fetch("race.json").then(r => r.json()).then(data => {
    const cards = data.runs.map(card);
    const lora = cards.find(c => c.run.arm === "p-checked");
    const others = cards.filter(c => c !== lora).map(c => c.run.seconds);
    const say = h("p", { class: "verdict", "aria-live": "polite" }, "Flip the switch and watch which one stops first.");
    const sw = h("button", { type: "button", class: "switch", role: "switch", "aria-checked": "false", onclick: start },
      h("span", { class: "knob" }), h("span", { class: "switch-word" }, "Let workalike take it"));
    const skip = h("button", { type: "button", class: "quiet-button", hidden: true, onclick: () => tick(Infinity) }, "Show the final results");
    const again = h("button", { type: "button", class: "quiet-button", hidden: true, onclick: start }, "Play it again");
    let timer = null, began = 0;

    function start() {
      clearInterval(timer);
      for (const c of cards) { c.shown = 0; c.done = false; c.el.classList.remove("done"); c.steps.replaceChildren(); c.end.replaceChildren(); c.clock.textContent = "0s"; }
      sw.classList.add("on");
      sw.setAttribute("aria-checked", "true");
      sw.disabled = true;
      again.hidden = true;
      skip.hidden = false;
      say.textContent = `Played at ${SPEED}x. The clocks show the seconds each run really took.`;
      scrollTo({ top: say.getBoundingClientRect().top + scrollY - 12, behavior: quiet ? "auto" : "smooth" });
      if (quiet) return tick(Infinity);
      began = performance.now();
      timer = setInterval(() => tick((performance.now() - began) * SPEED / 1000), 100);
    }
    function tick(t) {
      for (const c of cards) {
        const due = Math.min(c.run.steps.length, Math.floor(Math.min(1, t / c.run.seconds) * (c.run.steps.length + 1)));
        for (; c.shown < due; c.shown++) { const s = c.run.steps[c.shown]; c.steps.append(h("li", { class: s.held ? "was-held" : null }, s.text)); }
        c.clock.textContent = `${Math.round(Math.min(t, c.run.seconds))}s`;
        if (t >= c.run.seconds && !c.done) finish(c);
      }
      if (lora.done && !cards.every(c => c.done)) say.textContent = "The LoRA's draft is ready. The other two are still working.";
      if (cards.every(c => c.done)) {
        clearInterval(timer);
        say.textContent = `${lora.run.seconds} seconds and ${lora.run.off}% off, with no rule book in its prompt. The other two took ${others.join(" and ")}.`;
        skip.hidden = true;
        again.hidden = false;
      }
    }
    box.replaceChildren(
      h("div", { class: "spread drawn race-top" },
        h("div", { class: "desk" }, h("div", { class: "mail" }, h("p", { class: "from" }, data.ticket.from), h("p", { class: "body" }, data.ticket.body)),
          h("p", { class: "fact" }, data.ticket.fact), sw),
        h("div", { class: "note" }, h("h3", {}, "The person's way"), h("p", { class: "line" }, data.ticket.rule),
          h("p", { class: "note-empty" }, "Out of the box, it never sees this. The second column reads it on every ticket. The LoRA learned it."))),
      h("div", { class: "race-status" }, say, skip, again), h("div", { class: "arms" }, cards.map(c => c.el)));
  }).catch(() => { box.replaceChildren(h("p", { class: "blocked" }, "The race could not load. Reload the page to try again.")); });
})();
