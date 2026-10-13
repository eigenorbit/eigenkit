/* Keep box-drawn output aligned.
 *
 * `BenchmarkResult.table()` and `Circuit.draw()` frame their output with
 * U+2500 box-drawing characters. The JetBrains Mono that Google Fonts serves
 * carries no glyphs in that range, so the browser takes them from the next
 * font in the stack instead — measured on the built site, 7.4775px per cell
 * against 8.16px for everything else. Both are monospace; they just disagree
 * on the cell. Across the 70-column benchmark table that is 44px of drift, so
 * every `┬` sits left of the `│` beneath it and the frame comes apart.
 *
 * Nothing in CSS can pick a font per glyph and keep one advance width, so the
 * whole block has to come from one font. Marking the blocks in the markdown
 * would mean fenced attributes, which `tools/docs-checks/build_example_notebooks.py`
 * does not parse — it matches ```` ```lang ```` exactly, and the pages it reads
 * have to keep generating the same notebooks. So the blocks are found here
 * instead, by the characters that cause the problem.
 */
(() => {
  const BOX_DRAWING = /[\u2500-\u257F]/;

  const sweep = (root) => {
    if (!root || !root.querySelectorAll) {
      return;
    }
    root.querySelectorAll("pre > code").forEach((code) => {
      if (!code.classList.contains("box-drawing") && BOX_DRAWING.test(code.textContent)) {
        code.classList.add("box-drawing");
      }
    });
  };

  if (typeof document$ !== "undefined") {
    document$.subscribe(() => sweep(document));
  } else {
    document.addEventListener("DOMContentLoaded", () => sweep(document));
  }
})();
