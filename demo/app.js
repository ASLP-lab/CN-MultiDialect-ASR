const state = { data: null, dialect: 'all' };

function esc(value) {
  return String(value || '').replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

function renderFilters() {
  const entries = Object.entries(state.data.dialects);
  const buttons = [['all', '全部'], ...entries.map(([id, d]) => [id, d.label])];
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
    <audio controls preload="none" src="${esc(item.audio)}"></audio>
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
  `).join('') || '<p class="empty">没有可展示的样本</p>';
}

async function init() {
  state.data = await fetch(`selection.json?t=${Date.now()}`, { cache: 'no-store' }).then((r) => r.json());
  renderFilters();
  renderList();
}

init().catch((error) => {
  document.getElementById('dialect-list').innerHTML =
    `<p class="empty">加载失败：${esc(error.message)}</p>`;
});
