"""Tests for LLM providers."""

import os
from unittest.mock import MagicMock, Mock, patch

import pytest

from shardguard.core.llm_providers import (
    GeminiProvider,
    LLMProviderFactory,
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


class TestLLMProviderFactory:
    """Test LLMProviderFactory functionality."""

    def test_create_ollama_provider(self):
        """Test creating an Ollama provider."""
        provider = LLMProviderFactory.create_provider(
            "ollama", "llama3.2", base_url="http://example.com"
        )

        assert isinstance(provider, OllamaProvider)
        assert provider.model == "llama3.2"
        assert provider.base_url == "http://example.com"

    def test_create_gemini_provider(self):
        """Test creating a Gemini provider."""
        provider = LLMProviderFactory.create_provider(
            "gemini", "gemini-2.0-flash-exp", api_key="test-key"
        )

        assert isinstance(provider, GeminiProvider)
        assert provider.model == "gemini-2.0-flash-exp"
        assert provider.api_key == "test-key"

    def test_create_unsupported_provider(self):
        """Test creating an unsupported provider raises error."""
        with pytest.raises(ValueError, match="Unsupported provider type"):
            LLMProviderFactory.create_provider("unsupported", "model")

    def test_case_insensitive_provider_type(self):
        """Test that provider type is case insensitive."""
        provider1 = LLMProviderFactory.create_provider("OLLAMA", "llama3.2")
        provider2 = LLMProviderFactory.create_provider("Gemini", "gemini-2.0-flash-exp")

        assert isinstance(provider1, OllamaProvider)
        assert isinstance(provider2, GeminiProvider)
