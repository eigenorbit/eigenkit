# Use with AI

These docs are published in forms that AI assistants read directly, so a
coding assistant can answer questions about eigenkit from the current
documentation instead of from what it remembers.

## Copy a page

The menu at the top right of every page copies the page as Markdown, opens
its Markdown version, or starts a conversation about it in Claude or ChatGPT.
The Markdown version of a page lives next to it: the page
`https://docs.kit.eigenorbit.com/user-guide/encoding/` is also served as
`https://docs.kit.eigenorbit.com/user-guide/encoding/index.md`.

## llms.txt

- [`llms.txt`](https://docs.kit.eigenorbit.com/llms.txt): an index of every
  page with a link to its Markdown version, following
  [llmstxt.org](https://llmstxt.org/).
- [`llms-full.txt`](https://docs.kit.eigenorbit.com/llms-full.txt): every page,
  the API reference included, in one file.

## MCP server

The docs are also served by a [Model Context Protocol](https://modelcontextprotocol.io)
server at `https://kit.eigenorbit.com/mcp`. It needs no account or key, and it
gives the assistant two tools:

- `search_docs`: find the pages that match a query.
- `read_page`: read one page in full, as Markdown.

=== "Claude Code"

    ```bash
    claude mcp add --transport http eigenkit-docs https://kit.eigenorbit.com/mcp
    ```

=== "Claude.ai and Claude Desktop"

    Open the **Connectors** settings, click **+ Add**, then **Add custom
    connector**. Enter the name `eigenkit-docs` and the URL
    `https://kit.eigenorbit.com/mcp`, and choose **No sign in**.

=== "Cursor"

    Add the server to `~/.cursor/mcp.json`, or to `.cursor/mcp.json` in a
    project:

    ```json
    {
      "mcpServers": {
        "eigenkit-docs": {
          "url": "https://kit.eigenorbit.com/mcp"
        }
      }
    }
    ```

=== "VS Code"

    Add the server to `.vscode/mcp.json`:

    ```json
    {
      "servers": {
        "eigenkit-docs": {
          "type": "http",
          "url": "https://kit.eigenorbit.com/mcp"
        }
      }
    }
    ```
