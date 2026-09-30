import unittest

from specagent.protocol.side_effects import action_mutates_state


class SideEffectsTest(unittest.TestCase):
    def test_known_reads_and_writes(self) -> None:
        for name in (
            "canvas_list_announcements",
            "woo_coupons_list",
            "read_data_from_excel",
            "get_workbook_metadata",
            "list_directory",
        ):
            self.assertFalse(action_mutates_state({"tool": name}), name)
        for name in (
            "create_form",
            "write_data_to_excel",
            "send_email",
            "create_event",
            "python_execute",
            "claim_done",
            "search_and_replace",
            "add_multiple_choice_question",
        ):
            self.assertTrue(action_mutates_state({"tool": name}), name)

    def test_automationbench_and_retrieval_tools(self) -> None:
        self.assertFalse(action_mutates_state({"tool": "search", "arguments": {"query": "q"}}))
        self.assertFalse(
            action_mutates_state({"tool": "get_document", "arguments": {"docid": "1"}})
        )
        self.assertFalse(
            action_mutates_state({"tool": "api_search", "arguments": {"query": "gmail"}})
        )
        self.assertFalse(
            action_mutates_state({"tool": "search_tools", "arguments": {"query": "gmail"}})
        )
        self.assertFalse(
            action_mutates_state({"tool": "base64_encode", "arguments": {"text": "a"}})
        )
        self.assertFalse(
            action_mutates_state({"tool": "api_fetch", "arguments": {"method": "GET", "url": "/x"}})
        )
        self.assertFalse(
            action_mutates_state(
                {"tool": "api_fetch", "arguments": {"method": "HEAD", "url": "/x"}}
            )
        )
        self.assertTrue(
            action_mutates_state(
                {"tool": "api_fetch", "arguments": {"method": "POST", "url": "/x"}}
            )
        )
        self.assertTrue(
            action_mutates_state(
                {"tool": "api_fetch", "arguments": {"method": "PATCH", "url": "/x"}}
            )
        )
        self.assertTrue(action_mutates_state({"tool": "api_fetch", "arguments": {"url": "/x"}}))
        self.assertTrue(
            action_mutates_state(
                {"tool": "execute_tool", "arguments": {"tool_name": "gmail_send_email"}}
            )
        )

    def test_mcp_prefixed_names_and_message_actions(self) -> None:
        # A reply to the user and a flight check-in are irrevocable; lookups behind an MCP
        # server prefix are still reads.
        for action in (
            {"type": "message", "content": "Done."},
            {"tool": "flight_booking__check_in"},
            {"tool": "check_in"},
            {"tool": "email__send_email"},
            {"tool": "notion__API-patch-block-children"},
            {"tool": "email__download_attachment"},
        ):
            self.assertTrue(action_mutates_state(action), action)
        for name in (
            "flight_booking__search_flights",
            "email__read_email",
            "maps__directions",
            "notion__API-post-search",
            "notion__API-retrieve-a-page",
            "delivery_logistics__track_package",
            "visa_and_advisory__check_entry_requirements",
            "memory_search",
        ):
            self.assertFalse(action_mutates_state({"tool": name}), name)

    def test_local_and_mcp_reads_without_verb_prefix(self) -> None:
        # Names that miss the ``verb_`` marker form but are confirmed lookups.
        for name in (
            "web_fetch",
            "memory_get",
            "dir_list",
            "banking__listaccounts",
            "ecommerce__resources_list",
            "legal_search__prompts_list",
            "weather__check",
            "session_status",
            "calendar__event_info",
        ):
            self.assertFalse(action_mutates_state({"tool": name}), name)
        # Suffix markers never override a write verb.
        for name in (
            "send_list",
            "delete_get",
            "notification_hub__mark_read",
            "email__mark_emails",
            "reading",
            "writing",
            "exec",
            "browser",
        ):
            self.assertTrue(action_mutates_state({"tool": name}), name)

    def test_known_tools_releases_names_the_agent_does_not_have(self) -> None:
        known = {"email__send_email", "read_file"}
        self.assertFalse(action_mutates_state({"tool": "writing"}, known_tools=known))
        self.assertFalse(action_mutates_state({"tool": "write_file"}, known_tools=known))
        self.assertTrue(action_mutates_state({"tool": "email__send_email"}, known_tools=known))
        self.assertFalse(action_mutates_state({"tool": "read_file"}, known_tools=known))
        # A reply has no tool name and stays irrevocable regardless of the tool list.
        self.assertTrue(
            action_mutates_state({"type": "message", "content": "x"}, known_tools=known)
        )
        # No tool list: the default gating applies.
        self.assertTrue(action_mutates_state({"tool": "writing"}, known_tools=None))

    def test_mcp_atlas_lookups_without_read_verbs_are_readonly(self) -> None:
        for name in ("whois_whois_domain", "git_git_diff", "git_git_log", "git_git_show",
                     "filesystem_directory_tree", "memory_open_nodes", "calculator_calculate",
                     "wikipedia_summarize_article_section", "osm-mcp-server_geocode_address"):
            self.assertFalse(action_mutates_state({"tool": name}), name)
        # Executing code, running commands and writing to GitHub stay gated.
        for name in ("mcp-code-executor_execute_code", "cli-mcp-server_run_command",
                     "desktop-commander_start_process", "github_push_files",
                     "github_create_issue", "filesystem_write_file", "memory_create_entities",
                     "pubmed_download_pubmed_pdf"):
            self.assertTrue(action_mutates_state({"tool": name}), name)

    def test_unknown_defaults_to_mutating(self) -> None:
        self.assertTrue(action_mutates_state({"tool": "mystery_tool"}))
        self.assertFalse(
            action_mutates_state({"tool": "mystery_tool"}, unknown_mutates=False)
        )


if __name__ == "__main__":
    unittest.main()
