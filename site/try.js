/* The try-it box. Buttons become a manual; a new ticket is worked live with and without it; if the live model is not
   there, the same picks play back from a recorded run, and the screen says so. All model text goes in as plain text. */
(function () {
  const API = window.WORKALIKE_API || "https://workalike-try-jbmqte6syq-uc.a.run.app";
  const box = document.getElementById("try");
  const quiet = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const WHY = { busy: "the live model is busy right now", visitor_limit: "you have reached the limit for live runs for a few minutes",
                resting: "the live demo has used its budget for today", gone: "the live model did not answer" };
  const FIRST = {
    price: { from: "Priya Raman, Oak Street Bakery", body: "That's more than we paid last spring. Is there anything you can do on the price?",
             fact: "You may take up to 5% off.",
             options: [["hold", "Keep the list price"], ["small", "Take off as little as I can"], ["full", "Give them the full 5%"]] },
    date: { from: "Marco Silva, Harbor Youth Club", body: "Hi, we'd like a quote for 120 canvas tote bags. Thanks, Marco",
            fact: "They didn't say when they need them.",
            options: [["ask", "Ask when they need them"], ["earliest", "Quote our earliest date"]] } };
  const PICKED = { hold: "keep the list price", small: "take off as little as you can", full: "give the full 5%",
                   ask: "ask for the date", earliest: "quote your earliest date" };
  /* Recorded on our GPU, on these same two requests; build.py adds the runs to recorded.json as data.lora. */
  const LORA = {
    price: { big: l => `${l.off}% off`, text: l => `This same request, with no rules in its prompt. ${l.runs} runs at temperature 0, all alike. In our main test, on a ticket like this one, it took 0.91% and 0.65%. This request took longer because the desk sent its first draft back once.` },
    date: { big: l => "Asked for the date", text: l => `This same request, with no rules in its prompt. It asked in ${l.same} of ${l.runs} runs at temperature 0. In our main test, on tickets like this one, it asked in 5 of 6 and quoted a date nobody gave once.` } };
  const WHY_TRAIN = "It averaged 29 seconds a ticket across the main test, against 181 for the same GPU reading the full rule book. The hosted columns are a different model on Baseten's servers. The LoRA is for when your records can't leave your building.";
  let data = null, picks = { price: null, date: null }, busy = false;

  function h(tag, attrs, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === "class") el.className = v; else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else if (v === true) el.setAttribute(k, ""); else if (v !== false && v != null) el.setAttribute(k, v);
    }
    for (const kid of kids.flat()) if (kid != null) el.append(kid);
    return el;
  }
  const fill = (el, ...kids) => el.replaceChildren(...kids.filter(k => k != null));
  const rule = (kind, pick) => data ? data.manual[kind][pick].replace(/^R\d+\. /, "") : "";

  /* ---- the notebook: a line writes itself when a button is pressed ---- */
  const lines = { price: null, date: null };
  const note = h("div", { class: "note", "aria-live": "polite" },
    h("h3", {}, "Your manual"), lines.price = h("p", { class: "line" }), lines.date = h("p", { class: "line" }),
    h("p", { class: "note-empty" }, "Nothing here yet. Press an answer."));
  function write(kind, text) {
    note.querySelector(".note-empty").hidden = true;
    const el = lines[kind];
    el.textContent = "";
    if (quiet) { el.textContent = text; return; }
    const done = h("span"), pencil = h("span", { class: "pencil" });
    el.append(done, pencil);
    let i = 0;
    (function next() {
      if (el.lastChild !== pencil) return;  // written over by a newer pick
      i = Math.min(text.length, i + 2);
      done.textContent = text.slice(0, i);
      if (i < text.length) setTimeout(next, 16); else pencil.remove();
    })();
  }

  /* ---- the desk: what is on the left of the notebook ---- */
  const desk = h("div", { class: "desk" });
  box.replaceChildren(h("div", { class: "spread drawn" }, desk, note), h("div", { id: "results" }));
  const results = box.querySelector("#results");
  const mail = (from, body) => h("div", { class: "mail" }, h("p", { class: "from" }, from), h("p", { class: "body" }, body));

  function first(kind, n) {
    const q = FIRST[kind];
    desk.replaceChildren(
      h("p", { class: "turn" }, `Handle this one yourself. ${n} of 2.`), mail(q.from, q.body), h("p", { class: "fact" }, q.fact),
      h("div", { class: "choices" }, q.options.map(([id, label]) => h("button", { type: "button", disabled: !data, class: picks[kind] === id ? "chosen" : null,
        onclick: () => { picks[kind] = id; write(kind, rule(kind, id)); setTimeout(n === 1 ? () => first("date", 2) : ready, quiet ? 0 : 700); } }, label))));
  }

  function ready(ticket) {
    ticket = typeof ticket === "string" ? ticket : "price";
    const t = data.tickets[ticket];
    const sw = h("button", { type: "button", class: "switch", role: "switch", "aria-checked": "false", onclick: () => flip(ticket, sw) },
      h("span", { class: "knob" }), h("span", { class: "switch-word" }, "Let workalike take it"));
    fill(desk,
      h("p", { class: "turn" }, "A new request. Neither of you has seen it."), mail(t.from, t.body),
      t.thread.length ? h("p", { class: "fact" }, ticket === "price" ? "Earlier the shop quoted 225 tumblers at list price, $3.78 each. You may take up to 5% off." : "No ship date in the request.") : null,
      sw);
  }

  /* ---- a run ---- */
  function column(title, sub) {
    const steps = h("ol", { class: "steps" }), end = h("div", { class: "end" }), clock = h("span", { class: "clock" });
    const el = h("section", { class: "arm drawn" }, h("h3", {}, title), h("p", { class: "sub" }, sub, " ", clock), steps, end);
    return { el, steps, end, clock };
  }
  const said = (kind, d) => {
    if (!d || d.failed || !d.message) return "stopped without a draft";
    if (kind === "price") return d.off ? `took ${d.off}% off${d.off > 5 ? " (5% after rounding to the cent)" : ""}` : "kept the list price";
    return d.lines && d.lines.length ? `quoted with a ship date of ${d.lines[0].ship_by}` : "asked when they need it";
  };
  function finish(col, d) {
    col.clock.textContent = d.seconds ? `${Math.round(d.seconds)} seconds` : "";
    const failed = (d.checks || []).filter(c => !c.ok);
    fill(col.end,
      d.message ? h("blockquote", { class: "draft" }, d.message) : h("p", { class: "stopped" }, "It stopped without a draft. Run it again to see another try."),
      d.approvable ? h("p", { class: "held" }, "Held for your approval. Nothing was sent.") : null,
      failed.length ? h("p", { class: "blocked" }, "The desk's checks would not let this one through: " + failed.map(c => c.name).join(", ") + ".") : null);
  }
  async function live(ticket, on) {
    const stop = new AbortController();
    let watch = setTimeout(() => stop.abort(), 30000);
    const res = await fetch(API + "/run", { method: "POST", headers: { "Content-Type": "application/json" }, signal: stop.signal,
                                           body: JSON.stringify({ ticket, price: picks.price, date: picks.date }) });
    if (!res.ok) { clearTimeout(watch); let e = "gone"; try { e = (await res.json()).error; } catch (_) {} throw new Error(e); }
    const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
    let rest = "", ended = false;
    for (;;) {
      const { value, done } = await reader.read();
      clearTimeout(watch);
      if (done) break;
      watch = setTimeout(() => stop.abort(), 30000);
      rest += value;
      const parts = rest.split("\n\n");
      rest = parts.pop();
      for (const p of parts) if (p.startsWith("data: ")) {
        const e = JSON.parse(p.slice(6));
        if (e.type === "end") ended = true; else on(e);
      }
    }
    if (!ended) throw new Error("gone");
  }
  function recorded(ticket, on) {
    const arms = { none: data.runs[`${ticket}/none`], manual: data.runs[`${ticket}/${picks.price}/${picks.date}`] };
    const all = Object.entries(arms).flatMap(([arm, ev]) => (ev || []).map(e => ({ arm, ...e }))).sort((a, b) => a.t - b.t);
    return new Promise(res => {
      if (!all.length) return res();
      all.forEach((e, i) => setTimeout(() => { on(e); if (i === all.length - 1) res(); }, quiet ? 0 : e.t * 1000));
    });
  }
  async function flip(ticket, sw) {
    if (busy) return;
    busy = true;
    sw.setAttribute("aria-checked", "true");
    sw.classList.add("on");
    sw.disabled = true;
    const without = column("No manual", "Muse Spark Open"), mine = column("Your manual", "Muse Spark Open, same request");
    const lora = h("section", { class: "arm lora drawn" }, h("h3", {}, "Trained, no manual"), h("p", { class: "sub" }, `Muse Glimmer with the LoRA, recorded on our GPU, ${data.lora[ticket].seconds} seconds`),
                   h("p", { class: "big" }, LORA[ticket].big(data.lora[ticket])), h("p", {}, LORA[ticket].text(data.lora[ticket]), " ", WHY_TRAIN),
                   h("blockquote", { class: "draft" }, data.lora[ticket].message),
                   h("p", { class: "aside" }, "That LoRA learned one person's way: the smallest discount. ", h("a", { href: "#numbers" }, "See all 24 tickets.")));
    const mode = h("p", { class: "mode" }, "Live on Muse Spark Open, served by Baseten.");
    const verdict = h("p", { class: "verdict" }), after = h("div", { class: "after" });
    results.replaceChildren(mode, verdict, h("div", { class: "arms" }, without.el, mine.el, lora), after);
    results.scrollIntoView({ behavior: quiet ? "auto" : "smooth", block: "nearest" });
    const cols = { none: without, manual: mine }, ends = {};
    const start = performance.now();
    const tick = setInterval(() => { for (const [k, c] of Object.entries(cols)) if (!ends[k]) c.clock.textContent = `${Math.round((performance.now() - start) / 1000)} seconds`; }, 1000);
    const on = e => {
      const c = cols[e.arm];
      if (e.type === "step") c.steps.append(h("li", { class: e.held ? "was-held" : null }, e.text));
      if (e.type === "done") { ends[e.arm] = e; finish(c, e); }
    };
    try {
      await live(ticket, on);
      if (ends.none && ends.manual && ends.none.outcome === "error" && ends.manual.outcome === "error") throw new Error("gone");
    } catch (err) {
      for (const k in ends) delete ends[k];
      for (const c of Object.values(cols)) { c.steps.replaceChildren(); c.end.replaceChildren(); }
      mode.textContent = `This is a recorded run of these same picks, because ${WHY[err.message] || WHY.gone}.`;
      mode.classList.add("recorded");
      await recorded(ticket, on);
    }
    clearInterval(tick);
    verdict.textContent = `You said to ${PICKED[ticket === "price" ? picks.price : picks.date]}. With your manual it ${said(ticket, ends.manual)}. Without it, it ${said(ticket, ends.none)}.`;
    after.replaceChildren(
      h("button", { type: "button", onclick: () => { results.replaceChildren(); first("price", 1); } }, "Change my answers"),
      h("button", { type: "button", onclick: () => { results.replaceChildren(); ready(ticket === "price" ? "date" : "price"); } }, ticket === "price" ? "Try a request with no ship date" : "Try the price request again"),
      h("a", { class: "button", href: "#download" }, "Get the LoRA version"),
      h("details", {}, h("summary", {}, "See what the model read"), h("pre", {}, [data.manual.preamble, "R1. " + rule("price", picks.price), "R2. " + rule("date", picks.date)].join("\n"))));
    busy = false;
  }

  first("price", 1);
  fetch("recorded.json").then(r => r.json()).then(d => { data = d; first("price", 1); })
    .catch(() => { desk.append(h("p", { class: "blocked" }, "The demo could not load. Reload the page to try again.")); });
})();
