"""Tests for the CLI module."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
import typer
from typer.testing import CliRunner

from shardguard.cli import (
    _get_model_for_provider,
    _validate_gemini_api_key,
    app,
    create_planner,
)


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
            mock_context_manager, _mock_planner = self._create_mock_planner_context(
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
                mock_context_manager, _mock_planner = self._create_mock_planner_context(
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
                mock_context_manager, _mock_planner = self._create_mock_planner_context(
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

    def test_plan_command_success(self):
        """Test plan command successful execution."""
        with patch("shardguard.cli.create_planner") as mock_create_planner:
            with patch("shardguard.cli.CoordinationService") as mock_coord_service:
                mock_context_manager, _mock_planner = self._create_mock_planner_context(
                    "Available MCP Tools:\n\nServer: file-server"
                )
                mock_create_planner.return_value = mock_context_manager

                # Mock coordination service instance
                mock_coord = Mock()
                mock_plan_obj = Mock()
                mock_plan_obj.model_dump_json.return_value = '{"plan": "test"}'
                mock_coord.handle_prompt = AsyncMock(return_value=mock_plan_obj)
                mock_coord_service.return_value = mock_coord

                result = self.runner.invoke(app, ["plan", "write hello to file"])

                assert result.exit_code == 0
                assert '{"plan": "test"}' in result.stdout
                mock_coord.handle_prompt.assert_called_once_with("write hello to file")

    def test_plan_command_gemini_no_api_key(self):
        """Test plan command with Gemini provider but no API key."""
        with patch.dict("os.environ", {}, clear=True):
            result = self.runner.invoke(
                app, ["plan", "test prompt", "--provider", "gemini"]
            )

            assert result.exit_code == 1
            assert "Gemini API key required" in result.stdout

    def test_main_callback_with_verbose(self):
        """Test main callback with verbose flag."""
        with patch("shardguard.cli.create_planner") as mock_create_planner:
            with patch(
                "shardguard.cli._print_verbose_tools_info"
            ) as mock_verbose_print:
                mock_context_manager, _mock_planner = self._create_mock_planner_context(
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
            mock_context_manager, _mock_planner = self._create_mock_planner_context(
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
