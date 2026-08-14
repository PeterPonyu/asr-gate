(() => {
  const note = document.querySelector("[data-hover-note]");
  if (!note) return;
  const targets = document.querySelectorAll("[data-hover]");
  const show = (text) => {
    note.hidden = false;
    note.textContent = text;
  };
  targets.forEach((el) => {
    const text = el.getAttribute("data-hover");
    if (!text) return;
    el.addEventListener("mouseenter", () => show(text));
    el.addEventListener("focus", () => show(text));
  });
})();
