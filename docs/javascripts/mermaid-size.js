/* Give every mermaid diagram its natural size back.
 *
 * Mermaid emits `width="100%"` with a `max-width` equal to the diagram's own
 * width, so the theme fits the SVG to the content column and everything inside
 * it shrinks by the same factor. On the Architecture page that factor is 0.17
 * for the widest diagram (3933px of graph in a ~700px column), which renders
 * its 16px labels at under 3px.
 *
 * Setting an explicit width from the viewBox keeps the labels at 16px; the
 * block scrolls horizontally instead, which `stylesheets/mermaid.css` sets up.
 */
(() => {
  const natural = (svg) => {
    const box = svg.viewBox && svg.viewBox.baseVal;
    if (!box || !box.width || svg.dataset.naturalSize) {
      return;
    }
    svg.dataset.naturalSize = "1";
    svg.style.maxWidth = "none";
    svg.style.width = box.width + "px";
    svg.style.height = box.height + "px";
  };

  const sweep = (root) => {
    if (!root || !root.querySelectorAll) {
      return;
    }
    root.querySelectorAll(".mermaid > svg").forEach(natural);
  };

  // The theme renders each diagram asynchronously, so the SVG lands after this
  // script runs — and again after every instant-loading navigation.
  new MutationObserver((records) => {
    for (const record of records) {
      for (const node of record.addedNodes) {
        if (node.nodeType !== 1) {
          continue;
        }
        if (node.matches && node.matches(".mermaid > svg")) {
          natural(node);
        }
        sweep(node);
      }
    }
  }).observe(document.documentElement, { childList: true, subtree: true });

  if (typeof document$ !== "undefined") {
    document$.subscribe(() => sweep(document));
  } else {
    document.addEventListener("DOMContentLoaded", () => sweep(document));
  }
})();
