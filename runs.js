/* The evidence page: three chosen runs and all 144, from runs.json, which build.py makes from the raw rows. Text goes in as plain text. */
(function () {
  const ARMS = { none: "Out of the box", checked: "Reading the written rules", "p-checked": "LoRA, no rules in its prompt" };
  const VERDICTS = { ready: "Right draft", right_hold: "Rightly asked", needless_hold: "Asked when it did not have to", dangerous: "Dangerous", failed: "Held back, can't be approved" };
  const THREE = [
    { key: "discount-authority-yes", arm: "p-checked", round: 1, title: "It went right", note: "0.91% off, with no rules in its prompt. The smallest discount was the person's answer. Its last line mentions the floor price: the desk now holds that back, but this frozen test ran without that check." },
    { key: "date-2-missing", arm: "p-checked", round: 1, title: "The checks caught it", note: "The first draft asked for the date, and also told the customer the shop's lowest price. The desk's checks held it back, the model was told why, and the rewrite left it out." },
    { key: "date-1-missing", arm: "p-checked", round: 2, title: "It went wrong", note: "The first draft asked for the date, with a price outside a quote. The checks held it back, and the rewrite made a date up. A person would have seen this draft. It is the one dangerous draft in the LoRA's column." },
  ];
  const h = (tag, attrs, ...kids) => {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === "class") el.className = v; else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else if (v === true) el.setAttribute(k, ""); else if (v !== false && v != null) el.setAttribute(k, v);
    }
    for (const kid of kids.flat()) if (kid != null) el.append(kid);
    return el;
  };
  const isRight = v => v === "ready" || v === "right_hold";
  let data = null;

  function body(run) {
    const t = data.tickets[run.key], parts = [];
    const asked = t.thread.length ? h("details", {}, h("summary", {}, "Earlier mail"), t.thread.map(m => h("p", { class: "thread" }, h("b", {}, m.who === "customer" ? "Customer: " : "Shop: "), m.text))) : null;
    parts.push(h("div", { class: "mail" }, h("p", { class: "from" }, t.from), h("p", { class: "body" }, t.body)), asked,
      h("p", { class: "fact" }, `Discount authority ${Math.round((t.authority || 0) * 100)}%. The answer sheet wanted: ${t.want}.`));
    parts.push(h("p", { class: "tools" }, h("b", {}, "Tools it called: "), run.tools.join(", ") || "none"));
    if (run.first) {
      parts.push(h("p", { class: "was-held" }, h("b", {}, "First draft, held back: "), run.first.failed.map(c => `${c.name}. ${c.detail}`).join(" ")));
      run.first.drafts.forEach(d => parts.push(h("blockquote", { class: "draft struck" }, d)));
    }
    run.drafts.forEach(d => parts.push(h("p", {}, h("b", {}, run.first ? "Rewritten, what a person sees:" : "Draft, what a person sees:")), h("blockquote", { class: "draft" }, d)));
    run.questions.forEach(q => parts.push(h("p", {}, h("b", {}, "Question to the owner:")), h("blockquote", { class: "draft" }, q)));
    const quotes = Object.entries(run.quotes).map(([id, lines]) => `${id}: ${lines.map(l => `${l.qty} x ${l.sku} at $${l.unit_price}, ships ${l.ship_by}`).join("; ")}`);
    if (quotes.length) parts.push(h("p", { class: "tools" }, h("b", {}, "Quotes on the desk: "), quotes.join(". ")));
    if (run.checks.length) parts.push(h("ul", { class: "checks" }, run.checks.map(c => h("li", { class: c.ok ? null : "was-held" }, `${c.ok ? "Passed" : "Failed"}: ${c.name}. ${c.detail}`))));
    parts.push(h("p", { class: "tools" }, h("b", {}, "Verdict: "), VERDICTS[run.verdict], run.first_verdict !== run.verdict ? ` (first draft: ${VERDICTS[run.first_verdict]})` : "",
      run.reasons.length ? `. ${run.reasons.join("; ")}` : "", `. ${Math.round(run.seconds)} seconds, ${run.tokens_in.toLocaleString("en-US")} tokens in, ${run.tokens_out.toLocaleString("en-US")} out.`));
    return parts;
  }

  function three() {
    const box = document.getElementById("three");
    box.replaceChildren(h("div", { class: "three" }, THREE.map(s => {
      const run = data.runs.find(r => r.key === s.key && r.arm === s.arm && r.round === s.round);
      return h("article", { class: "picked drawn" }, h("h3", {}, s.title), h("p", {}, s.note), h("h4", {}, data.tickets[s.key].label), body(run));
    })));
  }

  const state = { arm: "", verdict: "", key: "", first: "", group: "" };
  function select(name, label, options) {
    return h("label", {}, label + " ", h("select", { id: "f-" + name, onchange: e => { state[name] = e.target.value; if (name === "key") state.group = ""; list(); } },
      options.map(([v, text]) => h("option", { value: v }, text))));
  }
  function matches(r) {
    if (state.arm && r.arm !== state.arm) return false;
    if (state.verdict === "right" && !isRight(r.verdict)) return false;
    if (state.verdict === "other" && !["needless_hold", "failed"].includes(r.verdict)) return false;
    if (state.verdict && !["right", "other"].includes(state.verdict) && r.verdict !== state.verdict) return false;
    if (state.first === "held" && !r.first) return false;
    if (state.key && r.key !== state.key) return false;
    if (state.group === "date" && !["date-1-missing", "date-2-missing", "date-and-short"].includes(r.key)) return false;
    return true;
  }
  function list() {
    const ARM_ORDER = ["none", "checked", "p-checked"];
    const seen = data.runs.filter(matches).sort((a, b) => data.tickets[a.key].label.localeCompare(data.tickets[b.key].label) || ARM_ORDER.indexOf(a.arm) - ARM_ORDER.indexOf(b.arm) || a.round - b.round);
    for (const name of ["arm", "verdict", "key", "first"]) { const el = document.getElementById("f-" + name); if (el) el.value = state[name]; }
    document.getElementById("count").textContent = `${seen.length} of ${data.runs.length} runs` + (state.group === "date" ? ", tickets with no ship date" : "");
    document.getElementById("list").replaceChildren(...seen.map(r => {
      const t = data.tickets[r.key];
      const panel = h("div", { class: "run-body" });
      const d = h("details", { class: "run" + (r.verdict === "dangerous" ? " bad" : ""), ontoggle: () => { if (d.open && !panel.childElementCount) panel.append(...body(r).filter(Boolean)); } },
        h("summary", {}, h("span", { class: "run-arm" }, ARMS[r.arm]), h("span", { class: "run-label" }, t.label), h("span", { class: "run-verdict" }, VERDICTS[r.verdict]), h("span", { class: "run-time" }, `run ${r.round}, ${Math.round(r.seconds)}s`)),
        panel);
      return d;
    }));
  }
  function fromHash() {
    const m = location.hash.match(/^#runs\?(.*)$/);
    if (!m) return false;
    Object.assign(state, { arm: "", verdict: "", key: "", first: "", group: "" });
    for (const [k, v] of new URLSearchParams(m[1])) if (k in state) state[k] = v;
    return true;
  }
  function start(d) {
    data = d;
    three();
    const keys = Object.keys(data.tickets).sort((a, b) => data.tickets[a].label.localeCompare(data.tickets[b].label));
    document.getElementById("filters").replaceChildren(
      select("arm", "Column", [["", "All"], ...Object.entries(ARMS)]),
      select("verdict", "Verdict", [["", "All"], ["right", "Right"], ["other", "Asked needlessly, or stuck at the checks"], ...Object.entries(VERDICTS)]),
      select("key", "Ticket", [["", "All"], ...keys.map(k => [k, `${data.tickets[k].label} (${k})`])]),
      select("first", "First draft", [["", "Any"], ["held", "Held back by the checks"]]));
    const hashed = fromHash();
    list();
    if (hashed) document.getElementById("runs").scrollIntoView();
    addEventListener("hashchange", () => { if (fromHash()) { list(); document.getElementById("runs").scrollIntoView(); } });
  }
  fetch("runs.json").then(r => r.json()).then(start).catch(() => {
    const note = "The runs could not load. Reload the page, or read them in the repo: baselines/results/test24b/graded.jsonl.";
    document.getElementById("three").textContent = note;
    document.getElementById("list").textContent = note;
  });
})();
