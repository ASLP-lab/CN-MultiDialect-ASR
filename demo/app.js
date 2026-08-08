const state = { data: null, dialect: 'all' };

// On GitHub HTML preview / non-local hosts, load assets from raw.githubusercontent.com
const isLocal = ['localhost', '127.0.0.1', ''].includes(location.hostname);
const BASE = isLocal
  ? ''
  : 'https://raw.githubusercontent.com/ASLP-lab/MultiDialect-ASR/main/demo/';

function esc(value) {
  return String(value || '').replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

function audioUrl(path) {
  return BASE + path.replace(/^\.\//, '');
}

function renderFilters() {
  const entries = Object.entries(state.data.dialects);
  const buttons = [['all', 'All'], ...entries.map(([id, d]) => [id, d.label])];
  const root = document.getElementById('filters');
  root.innerHTML = buttons.map(([id, label]) => (
    `<button class="${state.dialect === id ? 'active' : ''}" data-id="${esc(id)}">${esc(label)}</button>`
  )).join('');
  root.querySelectorAll('button').forEach((btn) => {
    btn.addEventListener('click', () => {
      state.dialect = btn.dataset.id;
      renderFilters();
      renderList();
    });
  });
}

function sampleCard(item) {
  return `<article class="sample">
    <audio controls preload="none" src="${esc(audioUrl(item.audio))}"></audio>
    <p>${esc(item.prediction)}</p>
  </article>`;
}

function renderList() {
  const entries = Object.entries(state.data.dialects)
    .filter(([id]) => state.dialect === 'all' || state.dialect === id);
  document.getElementById('dialect-list').innerHTML = entries.map(([id, dialect]) => `
    <section class="dialect">
      <h2>${esc(dialect.label)}</h2>
      <div class="samples">${dialect.items.map(sampleCard).join('')}</div>
    </section>
  `).join('') || '<p class="empty">No samples</p>';
}

async function init() {
  const url = `${BASE}selection.json?t=${Date.now()}`;
  state.data = await fetch(url, { cache: 'no-store' }).then((r) => {
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  });
  renderFilters();
  renderList();
}

init().catch((error) => {
  document.getElementById('dialect-list').innerHTML =
    `<p class="empty">Failed to load: ${esc(error.message)}</p>`;
});
