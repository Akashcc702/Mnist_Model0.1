async // ── Box blur helper (separable, O(n)) ──
// ── Preprocess binary camera canvas → 28×28 MNIST canvas ──

async // ===== THEME =====
function toggleTheme() {
  document.body.classList.toggle('dark');
  const btn = document.querySelector('.theme-btn');
  btn.textContent = document.body.classList.contains('dark') ? '☀️' : '🌙';
}
