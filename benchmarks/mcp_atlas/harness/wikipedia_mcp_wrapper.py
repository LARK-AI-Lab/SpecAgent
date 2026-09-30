"""Run the pinned Wikipedia MCP server with a distinct API User-Agent."""

import os

import wikipediaapi
from wikipedia_mcp.wikipedia_client import WikipediaClient

# Wikipedia asks API clients for a descriptive User-Agent with contact details.
_USER_AGENT = os.environ.get("WIKIPEDIA_USER_AGENT", "SpecAgentResearch/0.1 (academic benchmark)")
_original_init = WikipediaClient.__init__


def _init_with_user_agent(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    self.user_agent = _USER_AGENT
    self.wiki = wikipediaapi.Wikipedia(
        user_agent=_USER_AGENT,
        language=self.base_language,
        extract_format=wikipediaapi.ExtractFormat.WIKI,
    )


WikipediaClient.__init__ = _init_with_user_agent


if __name__ == "__main__":
    from wikipedia_mcp.__main__ import main

    main()
