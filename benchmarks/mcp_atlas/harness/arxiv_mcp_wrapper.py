"""Run the pinned arXiv MCP server without oversized API pages."""

import arxiv

_original_init = arxiv.Client.__init__


def _init_with_bounded_page(self, page_size=100, delay_seconds=3.0, num_retries=3):
    _original_init(
        self,
        page_size=min(page_size, 10),
        delay_seconds=delay_seconds,
        num_retries=num_retries,
    )


arxiv.Client.__init__ = _init_with_bounded_page


if __name__ == "__main__":
    from arxiv_mcp_server import main

    main()
