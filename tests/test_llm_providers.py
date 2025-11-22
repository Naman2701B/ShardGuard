"""Tests for LLM providers."""

import json
import os
from unittest.mock import MagicMock, Mock, patch

import pytest

from shardguard.core.llm_providers import (
    GeminiProvider,
    LLMProvider,
    create_provider,
    OllamaProvider,
)


class TestOllamaProvider:
    """Test OllamaProvider functionality."""

    def test_init_without_httpx(self):
        """Test OllamaProvider initialization without httpx."""
        with patch("builtins.__import__", side_effect=ImportError):
            provider = OllamaProvider()
            assert provider.model == "llama3.2"
            assert provider.base_url == "http://localhost:11434"
            assert provider.client is None

    def test_init_with_httpx(self):
        """Test OllamaProvider initialization with httpx."""
        mock_httpx = Mock()
        mock_client = MagicMock()
        mock_httpx.Client.return_value = mock_client

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            provider = OllamaProvider(model="llama3.1", base_url="http://example.com")

            assert provider.model == "llama3.1"
            assert provider.base_url == "http://example.com"
            assert provider.client == mock_client
            mock_httpx.Client.assert_called_once_with(timeout=300.0)

    def test_generate_response_sync_without_client(self):
        """Test sync response generation without client."""
        with patch("builtins.__import__", side_effect=ImportError):
            provider = OllamaProvider()

            response = provider.generate_response_sync("test prompt")

            assert "test prompt" in response
            assert "mock response" in response or "httpx not available" in response

    def test_generate_response_sync_with_client(self):
        """Test sync response generation with client."""
        mock_httpx = Mock()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "test response"}
        mock_client.post.return_value = mock_response
        mock_httpx.Client.return_value = mock_client

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            provider = OllamaProvider()
            response = provider.generate_response_sync("test prompt")

            assert response == "test response"
            mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_response_async_without_client(self):
        """Test async response generation without client."""
        with patch("builtins.__import__", side_effect=ImportError):
            provider = OllamaProvider()

            response = await provider.generate_response("test prompt")

            assert "test prompt" in response
            assert "mock response" in response or "httpx not available" in response

    @pytest.mark.asyncio
    async def test_generate_response_async_with_client(self):
        """Test async response generation with client."""
        mock_httpx = Mock()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "async test response"}
        mock_client.post.return_value = mock_response
        mock_httpx.Client.return_value = mock_client

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            provider = OllamaProvider()
            response = await provider.generate_response("test prompt")

            assert response == "async test response"
            mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_response_async_with_error(self):
        """Test async response generation handles errors gracefully."""
        mock_httpx = Mock()
        mock_client = MagicMock()
        mock_client.post.side_effect = Exception("Connection failed")
        mock_httpx.Client.return_value = mock_client

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            provider = OllamaProvider()
            response = await provider.generate_response("test prompt")

            assert "test prompt" in response
            assert "Error occurred" in response or "Connection failed" in response

    def test_generate_response_sync_with_error(self):
        """Test sync response generation handles errors gracefully."""
        mock_httpx = Mock()
        mock_client = MagicMock()
        mock_client.post.side_effect = Exception("Connection failed")
        mock_httpx.Client.return_value = mock_client

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            provider = OllamaProvider()
            response = provider.generate_response_sync("test prompt")

            assert "test prompt" in response
            assert "Error occurred" in response or "Connection failed" in response

    def test_mock_response_format(self):
        """Test that mock response has correct JSON structure."""
        with patch("builtins.__import__", side_effect=ImportError):
            provider = OllamaProvider()
            response = provider.generate_response_sync("test prompt")

            # Response should be valid JSON
            data = json.loads(response)
            assert "original_prompt" in data
            assert "sub_prompts" in data
            assert data["original_prompt"] == "test prompt"
            assert len(data["sub_prompts"]) > 0

    def test_ollama_api_call_parameters(self):
        """Test that Ollama API call includes correct parameters."""
        mock_httpx = Mock()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "test"}
        mock_client.post.return_value = mock_response
        mock_httpx.Client.return_value = mock_client

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            provider = OllamaProvider(model="test-model")
            provider.generate_response_sync("test prompt")

            # Verify the API call
            call_args = mock_client.post.call_args
            assert "api/generate" in call_args[0][0]
            assert call_args[1]["json"]["model"] == "test-model"
            assert call_args[1]["json"]["prompt"] == "test prompt"
            assert call_args[1]["json"]["stream"] is False


class TestGeminiProvider:
    """Test GeminiProvider functionality."""

    @patch.dict(os.environ, {}, clear=True)
    def test_init_without_api_key(self):
        """Test GeminiProvider initialization without API key."""
        provider = GeminiProvider(api_key=None)

        assert provider.model == "gemini-2.0-flash-exp"
        assert provider.api_key is None
        assert provider.client is None

    @patch.dict(os.environ, {}, clear=True)
    def test_init_with_api_key_no_import(self):
        """Test GeminiProvider initialization with API key but no google.generativeai."""
        with patch("builtins.__import__", side_effect=ImportError):
            provider = GeminiProvider(api_key="test-key")

            assert provider.api_key == "test-key"
            assert provider.client is None

    @patch.dict(os.environ, {}, clear=True)
    def test_generate_response_sync_without_client(self):
        """Test sync response generation without client."""
        provider = GeminiProvider(api_key=None)

        response = provider.generate_response_sync("test prompt")

        assert "test prompt" in response
        assert "mock response" in response or "Gemini API not available" in response

    @patch.dict(os.environ, {"GEMINI_API_KEY": "env-key"}, clear=True)
    def test_init_with_env_var(self):
        """Test GeminiProvider initialization with environment variable."""
        provider = GeminiProvider()

        assert provider.api_key == "env-key"

    @patch.dict(os.environ, {}, clear=True)
    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.configure")
    def test_init_with_api_key_and_genai(self, mock_configure, mock_gen_model):
        """Test GeminiProvider initialization with API key and google.generativeai."""
        mock_model = Mock()
        mock_gen_model.return_value = mock_model

        provider = GeminiProvider(model="custom-model", api_key="test-key")

        assert provider.model == "custom-model"
        assert provider.api_key == "test-key"
        assert provider.client == mock_model
        mock_configure.assert_called_once_with(api_key="test-key")

    @pytest.mark.asyncio
    @patch.dict(os.environ, {}, clear=True)
    async def test_generate_response_async_without_client(self):
        """Test async response generation without client."""
        provider = GeminiProvider(api_key=None)

        response = await provider.generate_response("test prompt")

        assert "test prompt" in response
        assert "mock response" in response or "Gemini API not available" in response

    @pytest.mark.asyncio
    @patch.dict(os.environ, {}, clear=True)
    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.configure")
    async def test_generate_response_async_with_client(self, mock_configure, mock_gen_model):
        """Test async response generation with client."""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "gemini response"
        mock_model.generate_content.return_value = mock_response
        mock_gen_model.return_value = mock_model

        provider = GeminiProvider(api_key="test-key")
        response = await provider.generate_response("test prompt")

        assert response == "gemini response"
        mock_model.generate_content.assert_called_once()

    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.configure")
    def test_generate_response_sync_with_client(self, mock_configure, mock_gen_model):
        """Test sync response generation with Gemini client."""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "gemini sync response"
        mock_model.generate_content.return_value = mock_response
        mock_gen_model.return_value = mock_model

        provider = GeminiProvider(api_key="test-key")
        response = provider.generate_response_sync("test prompt")

        assert response == "gemini sync response"
        mock_model.generate_content.assert_called_once()

    @pytest.mark.asyncio
    @patch.dict(os.environ, {}, clear=True)
    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.configure")
    async def test_generate_response_async_with_error(self, mock_configure, mock_gen_model):
        """Test async response generation handles errors gracefully."""
        mock_model = Mock()
        mock_model.generate_content.side_effect = Exception("API error")
        mock_gen_model.return_value = mock_model

        provider = GeminiProvider(api_key="test-key")
        response = await provider.generate_response("test prompt")

        assert "test prompt" in response
        assert "Error occurred" in response or "API error" in response

    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.configure")
    def test_generate_response_sync_with_error(self, mock_configure, mock_gen_model):
        """Test sync response generation handles errors gracefully."""
        mock_model = Mock()
        mock_model.generate_content.side_effect = Exception("API error")
        mock_gen_model.return_value = mock_model

        provider = GeminiProvider(api_key="test-key")
        response = provider.generate_response_sync("test prompt")

        assert "test prompt" in response
        assert "Error occurred" in response or "API error" in response

    @patch.dict(os.environ, {}, clear=True)
    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.configure")
    def test_init_with_genai_initialization_error(self, mock_configure, mock_gen_model):
        """Test GeminiProvider handles initialization errors gracefully."""
        mock_configure.side_effect = Exception("Configuration error")

        provider = GeminiProvider(api_key="test-key")

        assert provider.api_key == "test-key"
        assert provider.client is None

    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.configure")
    def test_gemini_api_call_parameters(self, mock_configure, mock_gen_model):
        """Test that Gemini API call includes correct parameters."""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "response"
        mock_model.generate_content.return_value = mock_response
        mock_gen_model.return_value = mock_model

        provider = GeminiProvider(api_key="test-key")
        provider.generate_response_sync("test prompt")

        # Verify the API call
        call_args = mock_model.generate_content.call_args
        assert call_args[0][0] == "test prompt"
        assert "generation_config" in call_args[1]
        assert call_args[1]["generation_config"]["temperature"] == 0.1
        assert call_args[1]["generation_config"]["top_p"] == 0.9

    def test_mock_response_format(self):
        """Test that mock response has correct JSON structure."""
        provider = GeminiProvider(api_key=None)
        response = provider.generate_response_sync("test prompt")

        # Response should be valid JSON
        data = json.loads(response)
        assert "original_prompt" in data
        assert "sub_prompts" in data
        assert data["original_prompt"] == "test prompt"
        assert len(data["sub_prompts"]) > 0

    def test_mock_response_with_error(self):
        """Test mock response includes error message."""
        provider = GeminiProvider(api_key=None)
        response = provider._mock_response("test", error="Test error")

        data = json.loads(response)
        assert "Error occurred" in data["sub_prompts"][0]["content"]
        assert "Test error" in data["sub_prompts"][0]["content"]


class TestLLMProviderFactory:
    """Test LLMProviderFactory functionality."""

    def test_create_ollama_provider(self):
        """Test creating an Ollama provider."""
        provider = create_provider(
            "ollama", "llama3.2", base_url="http://example.com"
        )

        assert isinstance(provider, OllamaProvider)
        assert provider.model == "llama3.2"
        assert provider.base_url == "http://example.com"

    def test_create_gemini_provider(self):
        """Test creating a Gemini provider."""
        provider = create_provider(
            "gemini", "gemini-2.0-flash-exp", api_key="test-key"
        )

        assert isinstance(provider, GeminiProvider)
        assert provider.model == "gemini-2.0-flash-exp"
        assert provider.api_key == "test-key"

    def test_create_unsupported_provider(self):
        """Test creating an unsupported provider raises error."""
        with pytest.raises(ValueError, match="Unsupported provider type"):
            create_provider("unsupported", "model")

    def test_case_insensitive_provider_type(self):
        """Test that provider type is case insensitive."""
        provider1 = create_provider("OLLAMA", "llama3.2")
        provider2 = create_provider("Gemini", "gemini-2.0-flash-exp")

        assert isinstance(provider1, OllamaProvider)
        assert isinstance(provider2, GeminiProvider)

    def test_create_ollama_provider_default_base_url(self):
        """Test creating Ollama provider uses default base URL."""
        provider = create_provider("ollama", "llama3.2")

        assert isinstance(provider, OllamaProvider)
        assert provider.base_url == "http://localhost:11434"

    def test_create_gemini_provider_with_env_var(self):
        """Test creating Gemini provider with environment variable."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "env-api-key"}):
            provider = create_provider("gemini", "gemini-2.0-flash-exp")

            assert isinstance(provider, GeminiProvider)
            assert provider.api_key == "env-api-key"

    def test_create_gemini_provider_cli_overrides_env(self):
        """Test that CLI API key overrides environment variable."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "env-api-key"}):
            provider = create_provider(
                "gemini", "gemini-2.0-flash-exp", api_key="cli-api-key"
            )

            assert isinstance(provider, GeminiProvider)
            assert provider.api_key == "cli-api-key"

class TestLLMProviderAbstractClass:
    """Test LLMProvider abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that LLMProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            LLMProvider()

    def test_subclass_must_implement_abstract_methods(self):
        """Test that subclasses must implement all abstract methods."""
        class IncompleteProvider(LLMProvider):
            pass

        with pytest.raises(TypeError):
            IncompleteProvider()