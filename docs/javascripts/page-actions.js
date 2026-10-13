// The page menu (overrides/partials/actions.html): "Copy page" puts the page's
// Markdown copy on the clipboard, and a click elsewhere closes the open menu.
//
// The text goes in as a promise inside a ClipboardItem: Safari refuses a
// clipboard write that starts after an awaited fetch, as the click no longer
// counts as the user's gesture by then.
document.addEventListener("click", (event) => {
  for (const menu of document.querySelectorAll(".page-actions__more[open]")) {
    if (!menu.contains(event.target)) menu.open = false;
  }

  const button = event.target.closest(".page-actions__copy");
  if (!button) return;
  const label = button.querySelector("span");
  const text = fetch(button.dataset.markdown)
    .then((response) => (response.ok ? response.text() : Promise.reject(response.status)))
    .then((markdown) => new Blob([markdown], { type: "text/plain" }));
  navigator.clipboard
    .write([new ClipboardItem({ "text/plain": text })])
    .then(() => (label.textContent = "Copied"))
    .catch(() => (label.textContent = "Copy failed"))
    .finally(() => setTimeout(() => (label.textContent = "Copy page"), 2000));
});
