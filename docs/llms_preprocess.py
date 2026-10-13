"""Keep the math of a page intact in its Markdown copy (mkdocs-llmstxt `preprocess`).

Arithmatex renders a formula as `\\(...\\)` or `\\[...\\]` text, and the plugin's
Markdown formatter reads those backslashes as escapes: `\\(\\langle\\psi\\)` came
out as `(\\\\langle\\\\psi)`. A formula is turned into code here, which the
formatter leaves alone: `$...$` inline, and a ```` ```math ```` block for display.
"""

from bs4 import BeautifulSoup, Tag


def preprocess(soup: BeautifulSoup, output: str) -> None:  # noqa: ARG001
    for element in soup.find_all(class_="arithmatex"):
        tex = element.get_text().strip()
        if element.name == "div":
            pre = soup.new_tag("pre", attrs={"class": "language-math"})
            pre.string = tex.removeprefix("\\[").removesuffix("\\]").strip()
            element.replace_with(pre)
        else:
            code: Tag = soup.new_tag("code")
            code.string = "$" + tex.removeprefix("\\(").removesuffix("\\)").strip() + "$"
            element.replace_with(code)
