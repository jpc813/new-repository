/* Kindred Paws — theme toggle + in-browser NLP (word embeddings, care-request analyzer) */
(function () {
  'use strict';

  /* ---------- Theme ---------- */
  const root = document.documentElement;
  function getTheme() { try { return localStorage.getItem('kp-theme') || 'system'; } catch (e) { return 'system'; } }
  function setTheme(t) {
    if (t === 'system') root.removeAttribute('data-theme'); else root.setAttribute('data-theme', t);
    try { localStorage.setItem('kp-theme', t); } catch (e) {}
    document.querySelectorAll('[data-theme-set]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.themeSet === t)));
    if (lastPlot) drawPlots(lastPlot);
  }
  document.querySelectorAll('[data-theme-set]').forEach(b => b.addEventListener('click', () => setTheme(b.dataset.themeSet)));
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { if (lastPlot) drawPlots(lastPlot); });
  const css = n => getComputedStyle(root).getPropertyValue(n).trim();

  /* ---------- Embeddings ---------- */
  let words = [], index = new Map(), E = null, DIM = 50, ready = null;
  function loadEmbeddings() {
    if (ready) return ready;
    ready = Promise.all([
      fetch('data/vocab.json').then(r => r.json()),
      fetch('data/vectors.bin').then(r => r.arrayBuffer())
    ]).then(([vocab, buf]) => {
      words = vocab.words; DIM = vocab.dim;
      const q = new Int8Array(buf);
      E = new Float32Array(q.length);
      for (let i = 0; i < words.length; i++) {
        let s = 0; const o = i * DIM;
        for (let d = 0; d < DIM; d++) { const v = q[o + d]; E[o + d] = v; s += v * v; }
        s = Math.sqrt(s) || 1;
        for (let d = 0; d < DIM; d++) E[o + d] /= s;
        index.set(words[i], i);
      }
    });
    return ready;
  }
  const vec = w => { const i = index.get(w); return i === undefined ? null : E.subarray(i * DIM, i * DIM + DIM); };
  function norm(v) { let s = 0; for (const x of v) s += x * x; s = Math.sqrt(s) || 1; return v.map(x => x / s); }
  function dot(a, b) { let s = 0; for (let d = 0; d < DIM; d++) s += a[d] * b[d]; return s; }
  function mean(vs) { const m = new Float32Array(DIM); vs.forEach(v => { for (let d = 0; d < DIM; d++) m[d] += v[d]; }); return norm(m); }
  function nearest(v, k, exclude) {
    const best = [];
    for (let i = 0; i < words.length; i++) {
      if (exclude && exclude.has(words[i])) continue;
      let s = 0; const o = i * DIM;
      for (let d = 0; d < DIM; d++) s += E[o + d] * v[d];
      if (best.length < k || s > best[best.length - 1].s) {
        best.push({ w: words[i], i, s }); best.sort((a, b) => b.s - a.s); if (best.length > k) best.pop();
      }
    }
    return best;
  }

  /* PCA via power iteration with deflation */
  function pca(rows, nComp) {
    const n = rows.length, mu = new Float64Array(DIM);
    rows.forEach(r => { for (let d = 0; d < DIM; d++) mu[d] += r[d] / n; });
    const X = rows.map(r => Array.from(r, (x, d) => x - mu[d]));
    const C = Array.from({ length: DIM }, () => new Float64Array(DIM));
    X.forEach(x => { for (let a = 0; a < DIM; a++) for (let b = 0; b < DIM; b++) C[a][b] += x[a] * x[b]; });
    const comps = [];
    for (let c = 0; c < nComp; c++) {
      let v = Array.from({ length: DIM }, (_, i) => Math.sin(i * 7.3 + c * 3.1) + 0.01);
      for (let it = 0; it < 200; it++) {
        const w = new Array(DIM).fill(0);
        for (let a = 0; a < DIM; a++) for (let b = 0; b < DIM; b++) w[a] += C[a][b] * v[b];
        const s = Math.hypot(...w) || 1; v = w.map(x => x / s);
      }
      let lam = 0; for (let a = 0; a < DIM; a++) { let t = 0; for (let b = 0; b < DIM; b++) t += C[a][b] * v[b]; lam += v[a] * t; }
      for (let a = 0; a < DIM; a++) for (let b = 0; b < DIM; b++) C[a][b] -= lam * v[a] * v[b];
      comps.push(v);
    }
    return X.map(x => comps.map(v => x.reduce((s, xi, d) => s + xi * v[d], 0)));
  }

  /* ---------- Word lab ---------- */
  const status = document.getElementById('labStatus');
  const list = document.getElementById('nbrList');
  let lastPlot = null, view = '2d';

  function explore(raw) {
    const w = raw.trim().toLowerCase();
    const v = vec(w);
    if (!v) { status.textContent = `"${raw}" isn't in the 20,000-word vocabulary. Try a more common word.`; return; }
    const nb = nearest(v, 10, new Set([w]));
    status.textContent = `Nearest words to "${w}" by cosine similarity.`;
    list.innerHTML = nb.map(n => `<li data-w="${n.w}"><span>${n.w}</span><span class="sim">${n.s.toFixed(3)}</span></li>`).join('');
    const labels = [w, ...nb.map(n => n.w)];
    const pts = labels.map(vec);
    lastPlot = { labels, p2: pca(pts, 2), p3: pca(pts, 3) };
    drawPlots(lastPlot);
  }
  list.addEventListener('click', e => { const li = e.target.closest('li'); if (li) { document.getElementById('wordIn').value = li.dataset.w; explore(li.dataset.w); } });

  /* Canvas plots (no external libraries): 2D scatter + rotatable 3D scatter */
  let rot = { yaw: 0.6, pitch: 0.35 }, spin = true;
  function setupCanvas(id) {
    const host = document.getElementById(id);
    let c = host.querySelector('canvas');
    if (!c) { c = document.createElement('canvas'); host.appendChild(c); }
    const w = host.clientWidth || 600, h = host.clientHeight || 400, dpr = window.devicePixelRatio || 1;
    c.width = w * dpr; c.height = h * dpr; c.style.width = w + 'px'; c.style.height = h + 'px';
    const ctx = c.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { c, ctx, w, h };
  }
  function scaleTo(vals, lo, hi) { const mn = Math.min(...vals), mx = Math.max(...vals); return vals.map(v => lo + (hi - lo) * ((v - mn) / ((mx - mn) || 1))); }
  function label(ctx, txt, x, y, main) {
    ctx.font = (main ? '600 15px ' : '500 13px ') + 'Inter, sans-serif';
    ctx.fillStyle = css('--text'); ctx.textAlign = 'center'; ctx.fillText(txt, x, y - (main ? 14 : 10));
  }
  function draw2d(p) {
    const { ctx, w, h } = setupCanvas('plot2d'); if (!w) return;
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = css('--line'); ctx.lineWidth = 1;
    for (let i = 1; i < 6; i++) { ctx.beginPath(); ctx.moveTo(i * w / 6, 0); ctx.lineTo(i * w / 6, h); ctx.stroke(); ctx.beginPath(); ctx.moveTo(0, i * h / 6); ctx.lineTo(w, i * h / 6); ctx.stroke(); }
    const xs = scaleTo(p.p2.map(r => r[0]), 50, w - 50), ys = scaleTo(p.p2.map(r => -r[1]), 40, h - 30);
    ctx.strokeStyle = css('--gold'); ctx.globalAlpha = .35;
    p.labels.forEach((_, i) => { if (i) { ctx.beginPath(); ctx.moveTo(xs[0], ys[0]); ctx.lineTo(xs[i], ys[i]); ctx.stroke(); } });
    ctx.globalAlpha = 1;
    p.labels.forEach((t, i) => {
      ctx.beginPath(); ctx.arc(xs[i], ys[i], i ? 6 : 9, 0, 7); ctx.fillStyle = i ? css('--heading') : css('--gold'); ctx.fill();
      label(ctx, t, xs[i], ys[i], !i);
    });
    ctx.font = '500 12px Inter'; ctx.fillStyle = css('--muted'); ctx.textAlign = 'left'; ctx.fillText('PC 1 →', w - 60, h - 8); ctx.fillText('↑ PC 2', 8, 16);
  }
  function draw3d(p) {
    const { ctx, w, h } = setupCanvas('plot3d'); if (!w) return;
    ctx.clearRect(0, 0, w, h);
    const sc = [0, 1, 2].map(k => scaleTo(p.p3.map(r => r[k]), -1, 1));
    const cy = Math.cos(rot.yaw), sy = Math.sin(rot.yaw), cp = Math.cos(rot.pitch), sp = Math.sin(rot.pitch);
    const R = Math.min(w, h) * 0.36;
    const proj = (x, y, z) => { const x1 = x * cy - z * sy, z1 = x * sy + z * cy; const y1 = y * cp - z1 * sp, z2 = y * sp + z1 * cp; const f = 3 / (3 + z2); return [w / 2 + x1 * R * f, h / 2 - y1 * R * f, z2, f]; };
    // axes cube
    ctx.strokeStyle = css('--line'); ctx.lineWidth = 1;
    const corners = [-1, 1]; const E3 = [];
    corners.forEach(a => corners.forEach(b => { E3.push([[-1, a, b], [1, a, b]], [[a, -1, b], [a, 1, b]], [[a, b, -1], [a, b, 1]]); }));
    E3.forEach(([u, v]) => { const A = proj(...u), B = proj(...v); ctx.beginPath(); ctx.moveTo(A[0], A[1]); ctx.lineTo(B[0], B[1]); ctx.stroke(); });
    const pts = p.labels.map((t, i) => ({ t, i, P: proj(sc[0][i], sc[1][i], sc[2][i]) })).sort((a, b) => b.P[2] - a.P[2]);
    const q0 = proj(sc[0][0], sc[1][0], sc[2][0]);
    ctx.strokeStyle = css('--gold'); ctx.globalAlpha = .35;
    pts.forEach(o => { if (o.i) { ctx.beginPath(); ctx.moveTo(q0[0], q0[1]); ctx.lineTo(o.P[0], o.P[1]); ctx.stroke(); } });
    ctx.globalAlpha = 1;
    pts.forEach(o => {
      ctx.beginPath(); ctx.arc(o.P[0], o.P[1], (o.i ? 6 : 9) * o.P[3], 0, 7); ctx.fillStyle = o.i ? css('--heading') : css('--gold'); ctx.fill();
      label(ctx, o.t, o.P[0], o.P[1], !o.i);
    });
  }
  function drawPlots(p) { if (view === '2d') draw2d(p); else draw3d(p); }
  (function interact3d() {
    const host = document.getElementById('plot3d'); let drag = null;
    host.addEventListener('pointerdown', e => { drag = [e.clientX, e.clientY]; spin = false; host.setPointerCapture(e.pointerId); });
    host.addEventListener('pointermove', e => { if (!drag) return; rot.yaw += (e.clientX - drag[0]) * 0.01; rot.pitch = Math.max(-1.4, Math.min(1.4, rot.pitch + (e.clientY - drag[1]) * 0.01)); drag = [e.clientX, e.clientY]; if (lastPlot) draw3d(lastPlot); });
    host.addEventListener('pointerup', () => { drag = null; });
    host.style.touchAction = 'none'; host.style.cursor = 'grab';
    (function tick() { if (spin && view === '3d' && lastPlot) { rot.yaw += 0.006; draw3d(lastPlot); } requestAnimationFrame(tick); })();
    window.addEventListener('resize', () => { if (lastPlot) drawPlots(lastPlot); });
  })();

  document.querySelectorAll('.plot-tabs button').forEach(b => b.addEventListener('click', () => {
    view = b.dataset.view;
    document.querySelectorAll('.plot-tabs button').forEach(x => x.setAttribute('aria-selected', String(x === b)));
    document.getElementById('plot2d').hidden = view !== '2d';
    document.getElementById('plot3d').hidden = view !== '3d';
    if (lastPlot) drawPlots(lastPlot);
  }));
  document.getElementById('wordForm').addEventListener('submit', e => { e.preventDefault(); loadEmbeddings().then(() => explore(document.getElementById('wordIn').value)); });
  document.querySelectorAll('.chip').forEach(c => c.addEventListener('click', () => { document.getElementById('wordIn').value = c.textContent; loadEmbeddings().then(() => explore(c.textContent)); }));
  document.getElementById('anaForm').addEventListener('submit', e => {
    e.preventDefault();
    loadEmbeddings().then(() => {
      const [a, b, c] = ['aA', 'aB', 'aC'].map(id => document.getElementById(id).value.trim().toLowerCase());
      const out = document.getElementById('anaOut');
      const va = vec(a), vb = vec(b), vc = vec(c);
      if (!va || !vb || !vc) { out.textContent = 'One of those words is not in the vocabulary.'; return; }
      // a : b :: c : ?   →  ? ≈ b − a + c
      const t = norm(Array.from(vb, (x, d) => x - va[d] + vc[d]));
      const r = nearest(t, 3, new Set([a, b, c]));
      out.textContent = r.map(x => x.w).join(', ');
    });
  });

  /* ---------- Care-request analyzer ---------- */
  const STOP = new Set("i me my we our you your he him his she her it its they them their a an the and or but so to of in on at for with is are was were be been am do does did have has had this that these those just also very really super need needs needed want would like can could will around about from by as if not no than then there here when what who which how all any some".split(' '));
  const BREEDS = ['golden retriever', 'labrador retriever', 'german shepherd', 'australian shepherd', 'border collie', 'french bulldog', 'english bulldog', 'cocker spaniel', 'shih tzu', 'great dane', 'bernese mountain dog', 'jack russell', 'pit bull', 'goldendoodle', 'labradoodle', 'bernedoodle', 'cavapoo', 'cockapoo', 'doodle', 'labrador', 'lab', 'retriever', 'poodle', 'beagle', 'husky', 'corgi', 'dachshund', 'terrier', 'bulldog', 'chihuahua', 'pug', 'schnauzer', 'shepherd', 'collie', 'spaniel', 'boxer', 'rottweiler', 'doberman', 'greyhound', 'pitbull', 'pomeranian', 'maltese', 'yorkie', 'havanese', 'weimaraner', 'vizsla', 'mutt', 'mix'];
  const HEALTH = ['pill', 'pills', 'medication', 'medications', 'meds', 'medicine', 'insulin', 'injection', 'allergy', 'allergies', 'arthritis', 'diabetic', 'diabetes', 'surgery', 'seizure', 'seizures', 'hip', 'hips', 'joint', 'joints', 'heart', 'blind', 'deaf', 'special diet', 'sensitive stomach'];
  const TRAVEL = ['travel', 'traveling', 'travelling', 'trip', 'trips', 'vacation', 'conference', 'away', 'out of town', 'overnight', 'overnights', 'nights', 'honeymoon', 'business trip'];
  const TRAITS = {
    'Energetic': ['energetic', 'hyper', 'active', 'playful', 'excitable'],
    'Anxious / shy': ['anxious', 'nervous', 'scared', 'shy', 'timid', 'fearful'],
    'Senior / gentle pace': ['old', 'elderly', 'senior', 'slow', 'aging'],
    'Reactive': ['aggressive', 'reactive', 'growls', 'lunges', 'barks'],
    'Affectionate': ['friendly', 'sweet', 'gentle', 'cuddly', 'loving', 'affectionate']
  };
  const PLANS = {
    walk: { name: 'Daily Walk', proto: 'quick walk exercise stroll potty break outside run fresh air' },
    full: { name: 'Full Care Visit', proto: 'feed meals food lunch medication pill play company lonely attention care long day water' },
    travel: { name: '24/7 Travel Care', proto: 'travel trip vacation overnight away nights stay weekend conference flight' }
  };

  function analyze(text) {
    const lower = text.toLowerCase();
    const toks = lower.match(/[a-z][a-z'-]*/g) || [];
    const spans = [];
    const mark = (re) => { let m; const g = new RegExp(re.source, re.flags.includes('g') ? re.flags : re.flags + 'g'); while ((m = g.exec(text))) spans.push([m.index, m.index + m[0].length]); };
    const found = (list) => list.filter(t => new RegExp('\\b' + t.replace(/ /g, '\\s+') + '\\b', 'i').test(text));

    // Name (simple pattern-based named-entity extraction)
    const notName = new Set(['I', 'He', 'She', 'It', 'My', 'Also', 'We', 'They', 'Our', 'The', 'His', 'Her', 'This', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'Chicago']);
    let name = null;
    for (const re of [/\b(?:named|name is|called)\s+([A-Z][a-z]+)/, /\bmy (?:dog|pup|puppy|boy|girl|baby),?\s+([A-Z][a-z]+)/, /\b([A-Z][a-z]+)\s+(?:is|was)\s+(?:a|an|my|our)\b/, /\b(?:adopted|meet|have|walk)\s+([A-Z][a-z]+)\b/, /\b([A-Z][a-z]+)\s+(?:needs|loves|gets)\b/]) {
      const m = text.match(re); if (m && !notName.has(m[1])) { name = m[1]; mark(new RegExp('\\b' + name + '\\b')); break; }
    }
    // Breed (longest match first)
    let breed = null;
    for (const b of BREEDS) { const re = new RegExp('\\b' + b.replace(/ /g, '\\s+') + 's?\\b', 'i'); if (re.test(text)) { breed = b; mark(new RegExp(re.source, 'i')); break; } }
    // Age
    let age = null; const am = text.match(/\b(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)[\s-]*(year|yr|month)s?[\s-]*old\b/i);
    if (am) { age = am[0]; mark(new RegExp(am[0].replace(/[-\s]/g, '[-\\s]*'), 'i')); }
    else if (/\bpupp(y|ies)\b/i.test(text)) age = 'Puppy';
    else if (/\b(senior|elderly)\b/i.test(text)) age = 'Senior';
    // Schedule
    const times = [...text.matchAll(/\b(\d{1,2}(?::\d{2})?\s?(?:am|pm|a\.m\.|p\.m\.))|\b(noon|midday|lunch(?:time)?|morning|afternoon|evening|night)\b/gi)].map(m => m[0]);
    const days = [...text.matchAll(/\b(weekdays?|weekends?|mondays?|tuesdays?|wednesdays?|thursdays?|fridays?|saturdays?|sundays?|every day|daily)\b/gi)].map(m => m[0]);
    mark(/\b(\d{1,2}(?::\d{2})?\s?(?:am|pm))|\b(noon|midday|lunch(?:time)?|weekdays?|weekends?|daily|every day)\b/gi);
    // Health & travel
    const health = found(HEALTH); health.forEach(h => mark(new RegExp('\\b' + h + '\\b', 'i')));
    const travel = found(TRAVEL); travel.forEach(h => mark(new RegExp('\\b' + h.replace(/ /g, '\\s+') + '\\b', 'i')));
    const nights = (text.match(/\b(\d{1,2}|one|two|three|four|five|six|seven|ten|fourteen)\s+(nights?|days?|weeks?)\b/i) || [])[0];
    if (nights) mark(new RegExp(nights, 'i'));

    // Temperament via embedding similarity to trait anchors (semantic, not just keyword)
    const AGE_WORDS = new Set(['year', 'years', 'old', 'month', 'months', 'yr', 'yrs']);
    const content = [...new Set(toks.filter(t => !STOP.has(t) && index.has(t)))];
    const traitToks = content.filter(t => !(age && AGE_WORDS.has(t)) && !(t === 'old' && /year|month/.test(lower)));
    const traits = [];
    for (const [label, anchors] of Object.entries(TRAITS)) {
      const c = mean(anchors.filter(a => index.has(a)).map(vec));
      let best = null;
      traitToks.forEach(t => { const s = dot(vec(t), c); if (s > 0.8 && (!best || s > best.s)) best = { t, s }; });
      if (best) { traits.push({ label, word: best.t, s: best.s }); mark(new RegExp('\\b' + best.t + '\\b', 'i')); }
    }

    const yrs = age && parseInt(age, 10);
    if (yrs >= 9 && !traits.some(t => t.label.startsWith('Senior'))) traits.push({ label: 'Senior / gentle pace', word: age, s: 1 });
    // Plan scoring: cosine similarity between the request's mean word vector and each plan prototype
    const docVec = content.length ? mean(content.map(vec)) : null;
    const scores = {};
    for (const [k, p] of Object.entries(PLANS)) {
      const pv = mean(p.proto.split(' ').filter(w => index.has(w)).map(vec));
      scores[k] = docVec ? dot(docVec, pv) : 0;
    }
    if (health.length) scores.full += 0.15;
    if (travel.length) scores.travel += 0.2;
    if (/\b(feed|feeding|meal|meals|food|lunch)\b/i.test(text)) scores.full += 0.08;
    const ranked = Object.entries(scores).sort((a, b) => b[1] - a[1]);
    return { name, breed, age, times, days, health, travel, nights, traits, scores, ranked, spans };
  }

  function highlight(text, spans) {
    spans.sort((a, b) => a[0] - b[0]);
    const merged = []; spans.forEach(s => { const l = merged[merged.length - 1]; if (l && s[0] <= l[1]) l[1] = Math.max(l[1], s[1]); else merged.push([...s]); });
    let out = '', pos = 0; const esc = s => s.replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
    merged.forEach(([a, b]) => { out += esc(text.slice(pos, a)) + '<mark>' + esc(text.slice(a, b)) + '</mark>'; pos = b; });
    return out + esc(text.slice(pos));
  }

  const outEl = document.getElementById('matchOut');
  const ta = document.getElementById('careText');
  function runAnalyze() {
    outEl.innerHTML = '<p class="muted">Reading…</p>';
    loadEmbeddings().then(() => {
      const r = analyze(ta.value);
      const dog = r.name || 'your dog';
      const pills = (arr, cls) => arr.length ? arr.map(x => `<span class="pill ${cls || ''}">${x}</span>`).join('') : '<span class="muted">—</span>';
      const sched = [...new Set([...r.days, ...r.times].map(s => s.toLowerCase()))];
      const vals = Object.values(r.scores), min = Math.min(...vals), max = Math.max(...vals);
      const pct = v => Math.round(35 + 65 * ((v - min) / ((max - min) || 1)));
      const top = r.ranked[0][0], second = r.ranked[1][0];
      let rec;
      if (r.travel.length && (r.times.length || r.days.length)) {
        const daily = r.scores.full >= r.scores.walk ? 'full' : 'walk';
        rec = `<strong>${PLANS[daily].name}</strong> on ${r.days.length ? r.days[0].toLowerCase() : 'workdays'}, plus <strong>24/7 Travel Care</strong>${r.nights ? ' for ' + r.nights : ''} while you're away.`;
        markPlans([daily, 'travel']);
      } else { rec = `<strong>${PLANS[top].name}</strong> looks like the best fit for ${dog}.`; markPlans([top]); }
      outEl.innerHTML = `
        <h3>What we understood</h3>
        <dl class="facts">
          <dt>Dog</dt><dd>${r.name || '<span class="muted">name not found</span>'}${r.breed ? ' · ' + r.breed : ''}${r.age ? ' · ' + r.age : ''}</dd>
          <dt>Schedule</dt><dd>${pills(sched)}</dd>
          <dt>Health</dt><dd>${pills(r.health, 'warn')}</dd>
          <dt>Temperament</dt><dd>${r.traits.length ? r.traits.map(t => `<span class="pill" title='matched "${t.word}" (similarity ${t.s.toFixed(2)})'>${t.label}</span>`).join('') : '<span class="muted">—</span>'}</dd>
          <dt>Travel</dt><dd>${r.travel.length ? pills([...new Set([...(r.nights ? [r.nights] : []), ...r.travel.filter(t => !(r.nights && r.nights.toLowerCase().includes(t)))])]) : '<span class="muted">none mentioned</span>'}</dd>
        </dl>
        <h3 class="h4">Plan match (semantic similarity)</h3>
        <div class="bars">${r.ranked.map(([k, v], i) => `<div class="bar ${i === 0 ? 'top' : ''}"><span>${PLANS[k].name}</span><span class="track"><span class="fill" style="width:${pct(v)}%"></span></span><span>${v.toFixed(2)}</span></div>`).join('')}</div>
        <div class="rec">${rec}</div>
        <h3 class="h4" style="margin-top:18px">Highlighted text</h3>
        <p style="margin:0;font-size:15px">${highlight(ta.value, r.spans)}</p>`;
    });
  }
  function markPlans(keys) {
    document.querySelectorAll('.price').forEach(c => c.classList.toggle('recommended', keys.includes(c.dataset.plan)));
  }
  document.getElementById('analyzeBtn').addEventListener('click', runAnalyze);
  const SAMPLES = {
    1: "Our dog Rosie is a 12 year old beagle with arthritis in her hips. She moves slowly now and needs her medication and dinner around 5pm on weekdays. She's very sweet.",
    2: "We adopted Milo, a rescue terrier mix, two months ago. He is really nervous around strangers and barks at other dogs. I just need someone to take him out for a walk at noon Monday through Friday."
  };
  document.querySelectorAll('[data-sample]').forEach(b => b.addEventListener('click', () => { ta.value = SAMPLES[b.dataset.sample]; runAnalyze(); }));

  /* ---------- Init ---------- */
  setTheme(getTheme());
  loadEmbeddings().then(() => { explore(document.getElementById('wordIn').value); runAnalyze(); })
    .catch(() => { status.textContent = 'Could not load the word vectors.'; });
})();
