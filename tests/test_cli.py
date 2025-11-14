"""Tests for the CLI module."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
import typer
from typer.testing import CliRunner

from shardguard.cli import app, create_planner, _validate_gemini_api_key, _get_model_for_provider, _print_provider_info, _print_tools_info, _count_tools_and_servers, _print_verbose_tools_info, _handle_errors


class TestCreatePlanner:
    """Test the create_planner context manager."""

    @pytest.mark.asyncio
    async def test_create_planner_success(self):
        """Test successful planner creation and cleanup."""
        with patch("shardguard.cli.PlanningLLM") as mock_planning_llm_class:
            mock_planner = Mock()
            mock_planner.get_available_tools_description = AsyncMock(
                return_value="Available MCP Tools:\n\nServer: test-server\n• test-tool"
            )
            mock_planner.close = Mock()
            mock_planning_llm_class.return_value = mock_planner

            async with create_planner() as planner:
                assert planner == mock_planner
                mock_planner.get_available_tools_description.assert_called_once()

            mock_planner.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_planner_with_connection_error(self):
        """Test planner creation when MCP connection fails."""
        with patch("shardguard.cli.PlanningLLM") as mock_planning_llm_class:
            mock_planner = Mock()
            mock_planner.get_available_tools_description = AsyncMock(
                side_effect=Exception("Connection failed")
            )
            mock_planner.close = Mock()
            mock_planning_llm_class.return_value = mock_planner

            async with create_planner() as planner:
                assert planner == mock_planner

            mock_planner.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_planner_with_custom_parameters(self):
        """Test planner creation with custom provider, model, and base_url."""
        with patch("shardguard.cli.PlanningLLM") as mock_planning_llm_class:
            mock_planner = Mock()
            mock_planner.get_available_tools_description = AsyncMock(
                return_value="Available MCP Tools:\n\nServer: test-server"
            )
            mock_planner.close = Mock()
            mock_planning_llm_class.return_value = mock_planner

            async with create_planner(
                provider_type="gemini",
                model="gemini-2.0-flash-exp",
                base_url="http://custom:8080",
                api_key="test-key"
            ) as planner:
                assert planner == mock_planner

            mock_planning_llm_class.assert_called_once_with(
                provider_type="gemini",
                model="gemini-2.0-flash-exp",
                base_url="http://custom:8080",
                api_key="test-key"
            )
            mock_planner.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_planner_no_mcp_tools_available(self):
        """Test planner creation when no MCP tools are available."""
        with patch("shardguard.cli.PlanningLLM") as mock_planning_llm_class:
            mock_planner = Mock()
            mock_planner.get_available_tools_description = AsyncMock(
                return_value="No MCP tools available."
            )
            mock_planner.close = Mock()
            mock_planning_llm_class.return_value = mock_planner

            async with create_planner() as planner:
                assert planner == mock_planner

            mock_planner.close.assert_called_once()

class TestCLICommands:
    """Test CLI commands using proper mocking without global state."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def _create_mock_planner_context(
        self, tools_description="Available MCP Tools:\n\nServer: test-server"
    ):
        """Helper to create a mock planner context manager."""
        mock_planner = Mock()
        mock_planner.get_available_tools_description = AsyncMock(
            return_value=tools_description
        )

        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__ = AsyncMock(return_value=mock_planner)
        mock_context_manager.__aexit__ = AsyncMock(return_value=None)

        return mock_context_manager, mock_planner

    def test_list_tools_command_ollama(self):
        """Test list-tools command with Ollama provider."""
        with patch("shardguard.cli.create_planner") as mock_create_planner:
            mock_context_manager, mock_planner = self._create_mock_planner_context(
                "Available MCP Tools:\n\nServer: file-server\n• read-file\n• write-file"
            )
            mock_create_planner.return_value = mock_context_manager

            result = self.runner.invoke(app, ["list-tools"])

            assert result.exit_code == 0
            assert "Available MCP Tools:" in result.stdout
            mock_create_planner.assert_called_once()

    def test_list_tools_command_verbose(self):
        """Test list-tools command with verbose flag."""
        with patch("shardguard.cli.create_planner") as mock_create_planner:
            with patch(
                "shardguard.cli._print_verbose_tools_info"
            ) as mock_verbose_print:
                mock_context_manager, mock_planner = self._create_mock_planner_context(
                    "Available MCP Tools:\n\nServer: file-server"
                )
                mock_create_planner.return_value = mock_context_manager

                result = self.runner.invoke(app, ["list-tools", "--verbose"])

                assert result.exit_code == 0
                mock_verbose_print.assert_called_once()

    def test_list_tools_command_gemini(self):
        """Test list-tools command with Gemini provider."""
        with patch("shardguard.cli.create_planner") as mock_create_planner:
            with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
                mock_context_manager, mock_planner = self._create_mock_planner_context(
                    "Available MCP Tools:\n\nServer: gemini-server"
                )
                mock_create_planner.return_value = mock_context_manager

                result = self.runner.invoke(
                    app,
                    [
                        "list-tools",
                        "--provider",
                        "gemini",
                        "--model",
                        "gemini-2.0-flash-exp",
                    ],
                )

                assert result.exit_code == 0
                mock_create_planner.assert_called_once_with(
                    "gemini",
                    "gemini-2.0-flash-exp",
                    "http://localhost:11434",
                    "test-key",
                )

    def test_list_tools_command_with_custom_ollama_url(self):
        """Test list-tools command with custom Ollama URL."""
        with patch("shardguard.cli.create_planner") as mock_create_planner:
            with patch.dict("os.environ", {}, clear=True):
                mock_context_manager, mock_planner = self._create_mock_planner_context(
                    "Available MCP Tools:\n\nServer: file-server"
                )
                mock_create_planner.return_value = mock_context_manager

                result = self.runner.invoke(
                    app,
                    [
                        "list-tools",
                        "--ollama-url",
                        "http://localhost:11434",
                    ],
                )

                assert result.exit_code == 0
                mock_create_planner.assert_called_once_with(
                    "ollama",
                    "llama3.2",
                    "http://localhost:11434",
                    None,
                )

    def test_list_tools_command_no_tools_available(self):
        """Test list-tools command when no tools are available."""
        with patch("shardguard.cli.create_planner") as mock_create_planner:
            mock_context_manager, mock_planner = self._create_mock_planner_context(
                "No MCP tools available."
            )
            mock_create_planner.return_value = mock_context_manager

            result = self.runner.invoke(app, ["list-tools"])

            assert result.exit_code == 0

    def test_plan_command_success(self):
        """Test plan command successful execution."""
        with patch("shardguard.cli.create_planner") as mock_create_planner:
            with patch("shardguard.cli.CoordinationService") as mock_coord_service:
                with patch("shardguard.cli._validate_output"):
                    mock_context_manager, mock_planner = self._create_mock_planner_context(
                        "Available MCP Tools:\n\nServer: file-server"
                    )
                    mock_create_planner.return_value = mock_context_manager

                    # Mock coordination service instance
                    mock_coord = Mock()
                    mock_plan_obj = Mock()
                    mock_plan_obj.model_dump_json.return_value = '{"plan": "test"}'
                    mock_plan_obj.model_dump.return_value = {"plan": "test"}
                    mock_plan_obj.sub_prompts = []
                    mock_coord.handle_prompt = AsyncMock(return_value=mock_plan_obj)
                    mock_coord.handle_subtasks = AsyncMock()
                    mock_coord_service.return_value = mock_coord

                    result = self.runner.invoke(app, ["plan", "write hello to file"])

                    assert result.exit_code == 0
                    assert '{"plan": "test"}' in result.stdout
                    mock_coord.handle_prompt.assert_called_once_with("write hello to file")

    def test_plan_command_with_subtasks(self):
        """Test plan command executes subtasks."""
        with patch("shardguard.cli.create_planner") as mock_create_planner:
            with patch("shardguard.cli.CoordinationService") as mock_coord_service:
                with patch("shardguard.cli._validate_output"):
                    with patch.dict("os.environ", {}, clear=True):
                        mock_context_manager, mock_planner = self._create_mock_planner_context(
                            "Available MCP Tools:\n\nServer: file-server"
                        )
                        mock_create_planner.return_value = mock_context_manager

                        mock_coord = Mock()
                        mock_plan_obj = Mock()
                        mock_plan_obj.model_dump_json.return_value = '{"plan": "test"}'
                        mock_plan_obj.model_dump.return_value = {"plan": "test"}
                        mock_plan_obj.sub_prompts = [{"id": 1, "content": "step1"}]
                        mock_coord.handle_prompt = AsyncMock(return_value=mock_plan_obj)
                        mock_coord.handle_subtasks = AsyncMock()
                        mock_coord_service.return_value = mock_coord

                        result = self.runner.invoke(app, ["plan", "test prompt"])

                        assert result.exit_code == 0
                        mock_coord.handle_subtasks.assert_called_once_with(
                            mock_plan_obj.sub_prompts, "ollama", "llama3.2", None
                        )

    def test_plan_command_gemini_no_api_key(self):
        """Test plan command with Gemini provider but no API key."""
        with patch.dict("os.environ", {}, clear=True):
            result = self.runner.invoke(
                app, ["plan", "test prompt", "--provider", "gemini"]
            )

            assert result.exit_code == 1
            assert "Gemini API key required" in result.stdout

    def test_plan_command_gemini_with_api_key_from_cli(self):
        """Test plan command with Gemini provider and API key from CLI."""
        with patch("shardguard.cli.create_planner") as mock_create_planner:
            with patch("shardguard.cli.CoordinationService") as mock_coord_service:
                with patch("shardguard.cli._validate_output"):
                    mock_context_manager, mock_planner = self._create_mock_planner_context(
                        "Available MCP Tools:\n\nServer: gemini-server"
                    )
                    mock_create_planner.return_value = mock_context_manager

                    mock_coord = Mock()
                    mock_plan_obj = Mock()
                    mock_plan_obj.model_dump_json.return_value = '{"plan": "test"}'
                    mock_plan_obj.model_dump.return_value = {"plan": "test"}
                    mock_plan_obj.sub_prompts = []
                    mock_coord.handle_prompt = AsyncMock(return_value=mock_plan_obj)
                    mock_coord.handle_subtasks = AsyncMock()
                    mock_coord_service.return_value = mock_coord

                    result = self.runner.invoke(
                        app,
                        [
                            "plan",
                            "test prompt",
                            "--provider",
                            "gemini",
                            "--gemini-api-key",
                            "cli-key",
                        ],
                    )

                    assert result.exit_code == 0
                    mock_create_planner.assert_called_once_with(
                        "gemini",
                        "gemini-2.0-flash-exp",
                        "http://localhost:11434",
                        "cli-key",
                    )

    def test_plan_command_with_custom_model(self):
        """Test plan command with custom model specified."""
        with patch("shardguard.cli.create_planner") as mock_create_planner:
            with patch("shardguard.cli.CoordinationService") as mock_coord_service:
                with patch("shardguard.cli._validate_output"):
                    with patch.dict("os.environ", {}, clear=True):
                        mock_context_manager, mock_planner = self._create_mock_planner_context(
                            "Available MCP Tools:\n\nServer: file-server"
                        )
                        mock_create_planner.return_value = mock_context_manager

                        mock_coord = Mock()
                        mock_plan_obj = Mock()
                        mock_plan_obj.model_dump_json.return_value = '{"plan": "test"}'
                        mock_plan_obj.model_dump.return_value = {"plan": "test"}
                        mock_plan_obj.sub_prompts = []
                        mock_coord.handle_prompt = AsyncMock(return_value=mock_plan_obj)
                        mock_coord.handle_subtasks = AsyncMock()
                        mock_coord_service.return_value = mock_coord

                        result = self.runner.invoke(
                            app,
                            ["plan", "test prompt", "--model", "llama2"],
                        )

                        assert result.exit_code == 0
                        mock_create_planner.assert_called_once_with(
                            "ollama",
                            "llama2",
                            "http://localhost:11434",
                            None,
                        )

    def test_plan_command_connection_error(self):
        """Test plan command handles connection errors gracefully."""
        with patch("shardguard.cli.create_planner") as mock_create_planner:
            mock_context_manager = AsyncMock()
            mock_context_manager.__aenter__ = AsyncMock(
                side_effect=ConnectionError("Failed to connect to Ollama")
            )
            mock_context_manager.__aexit__ = AsyncMock(return_value=None)
            mock_create_planner.return_value = mock_context_manager

            result = self.runner.invoke(app, ["plan", "test prompt"])

            assert result.exit_code == 1
            assert "Connection Error" in result.stdout

    def test_plan_command_generic_error(self):
        """Test plan command handles generic errors gracefully."""
        with patch("shardguard.cli.create_planner") as mock_create_planner:
            mock_context_manager = AsyncMock()
            mock_context_manager.__aenter__ = AsyncMock(
                side_effect=ValueError("Some validation error")
            )
            mock_context_manager.__aexit__ = AsyncMock(return_value=None)
            mock_create_planner.return_value = mock_context_manager

            result = self.runner.invoke(app, ["plan", "test prompt"])

            assert result.exit_code == 1
            assert "Error" in result.stdout

    def test_main_callback_with_verbose(self):
        """Test main callback with verbose flag."""
        with patch("shardguard.cli.create_planner") as mock_create_planner:
            with patch(
                "shardguard.cli._print_verbose_tools_info"
            ) as mock_verbose_print:
                mock_context_manager, mock_planner = self._create_mock_planner_context(
                    "Available MCP Tools:\n\nServer: file-server"
                )
                mock_create_planner.return_value = mock_context_manager

                result = self.runner.invoke(app, ["--verbose"])

                assert result.exit_code == 0
                assert "Welcome to ShardGuard!" in result.stdout
                mock_verbose_print.assert_called_once()

    def test_main_callback_without_verbose(self):
        """Test main callback without verbose flag."""
        with patch("shardguard.cli.create_planner") as mock_create_planner:
            mock_context_manager, mock_planner = self._create_mock_planner_context(
                "Available MCP Tools:\n\nServer: file-server"
            )
            mock_create_planner.return_value = mock_context_manager

            result = self.runner.invoke(app, [])

            assert result.exit_code == 0
            assert "Welcome to ShardGuard!" in result.stdout
            assert "Use --help to see available commands" in result.stdout


class TestHelperFunctions:
    """Test CLI helper functions."""

    def test_validate_gemini_api_key_valid(self):
        """Test Gemini API key validation with valid key."""
        # Should not raise exception
        _validate_gemini_api_key("gemini", "valid-key")

    def test_validate_gemini_api_key_missing(self):
        """Test Gemini API key validation with missing key."""
        with pytest.raises(typer.Exit):
            _validate_gemini_api_key("gemini", None)

    def test_validate_gemini_api_key_not_gemini(self):
        """Test Gemini API key validation with non-Gemini provider."""
        # Should not raise exception for non-Gemini providers
        _validate_gemini_api_key("ollama", None)

    def test_get_model_for_provider_explicit(self):
        """Test model selection with explicit model."""
        result = _get_model_for_provider("ollama", "custom-model")
        assert result == "custom-model"

    def test_get_model_for_provider_auto_detect_gemini(self):
        """Test model auto-detection for Gemini."""
        result = _get_model_for_provider("gemini", None)
        assert result == "gemini-2.0-flash-exp"

    def test_get_model_for_provider_auto_detect_ollama(self):
        """Test model auto-detection for Ollama."""
        result = _get_model_for_provider("ollama", None)
        assert result == "llama3.2"

    def test_print_provider_info_ollama(self):
        """Test _print_provider_info for Ollama provider."""

        with patch("shardguard.cli.console") as mock_console:
            _print_provider_info("ollama", "llama3.2", "http://localhost:11434")
            mock_console.print.assert_called_once()
            call_args = mock_console.print.call_args[0][0]
            assert "Ollama" in call_args
            assert "llama3.2" in call_args

    def test_print_provider_info_gemini(self):
        """Test _print_provider_info for Gemini provider."""

        with patch("shardguard.cli.console") as mock_console:
            _print_provider_info("gemini", "gemini-2.0-flash-exp", "http://localhost:11434")
            mock_console.print.assert_called_once()
            call_args = mock_console.print.call_args[0][0]
            assert "Gemini" in call_args

    def test_print_tools_info_with_tools(self):
        """Test _print_tools_info when tools are available."""
        with patch("shardguard.cli.console") as mock_console:
            tools_description = "Server: test-server\n• test-tool"
            _print_tools_info(tools_description, verbose=False)
            mock_console.print.assert_called()

    def test_print_tools_info_no_tools(self):
        """Test _print_tools_info when no tools are available."""
        with patch("shardguard.cli.console") as mock_console:
            tools_description = "No MCP tools available."
            _print_tools_info(tools_description, verbose=False)
            # Should not print when no tools available
            assert mock_console.print.call_count == 0

    def test_print_tools_info_verbose(self):
        """Test _print_tools_info with verbose flag."""
        with patch("shardguard.cli._print_verbose_tools_info") as mock_verbose:
            tools_description = "Server: test-server\n• test-tool"
            _print_tools_info(tools_description, verbose=True)
            mock_verbose.assert_called_once_with(tools_description)

    def test_count_tools_and_servers(self):
        """Test _count_tools_and_servers helper function."""
        tools_description = """
        Server: file-server
        • read-file
        • write-file
        Server: db-server
        • query-db
        • insert-db
        """
        tool_count, server_count = _count_tools_and_servers(tools_description)
        assert tool_count == 4
        assert server_count == 2

    def test_count_tools_and_servers_no_tools(self):
        """Test _count_tools_and_servers with no tools."""
        tools_description = "No MCP tools available."
        tool_count, server_count = _count_tools_and_servers(tools_description)
        assert tool_count == 0
        assert server_count == 0

    def test_print_verbose_tools_info(self):
        """Test _print_verbose_tools_info prints server and tool details."""
        with patch("shardguard.cli.console") as mock_console:
            tools_description = """
            Server: file-server
            • read-file: Read a file
            • write-file: Write a file
            """
            _print_verbose_tools_info(tools_description)
            # Should call console.print multiple times
            assert mock_console.print.call_count >= 2

    def test_handle_errors_connection_error_ollama(self):
        """Test _handle_errors with connection error for Ollama."""
        with pytest.raises(typer.Exit):
            _handle_errors(ConnectionError("Connection failed"), "ollama")

    def test_handle_errors_connection_error_gemini(self):
        """Test _handle_errors with connection error for Gemini."""
        with pytest.raises(typer.Exit):
            _handle_errors(ConnectionError("Connection failed"), "gemini")

    def test_handle_errors_generic_error(self):
        """Test _handle_errors with generic error."""
        with pytest.raises(typer.Exit):
            _handle_errors(ValueError("Generic error"), "ollama")