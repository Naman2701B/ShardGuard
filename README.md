# ShardGuard

ShardGuard is a proof-of-concept system designed to secure Model-Command Processor (MCP) workflows in the presence
of untrusted LLMs. While LLMs offer powerful automation capabilities across sensitive user contexts—like email,
calendars, and financial services—they also pose significant risks, including prompt injection, data exfiltration,
and unauthorized tool usage. ShardGuard tackles this by isolating execution into fine-grained, privacy-preserving
steps coordinated by a trusted service. It leverages a trusted planning LLM to decompose tasks, uses opaque values
and sanitization functions to limit data exposure, and ensures least-privilege access to MCP tools. Unlike
traditional information flow control systems, ShardGuard avoids complex labeling and taint tracking, offering a more
practical yet effective path to safe LLM-driven automation.

> **⚠️ Note**: This is a basic prototype of [a MCP security system](https://docs.google.com/document/d/1fB_DedbmW5E7MQSgXd98iNj9aQfXtajNKSEtAGUzLYI/edit?tab=t.0).
> Don't use it for anything production. We're still testing things to see if this is a good approach!

## Features

- **Input Sanitization**: Automatically sanitizes user input with rich CLI logging
- **LLM Integration**: Supports local Ollama and mock LLM for development that can be extended
- **Sensitive Data Masking**: Automatically identifies and masks sensitive information with reference placeholders

## Quick Start

### Prerequisites

- Python 3.13+
- Poetry (for dependency management)
- Ollama

### Installation

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd ShardGuard
   ```

2. **Install dependencies**:

   ```bash
   poetry install
   ```

3. **Activate the virtual environment**:

   ```bash
   eval $(poetry env activate)
   ```

### Basic Usage

```bash
# Using mock LLM (development)
shardguard plan "Create a secure web application with password secret123"

# Using Ollama (requires ollama serve)
shardguard plan --model llama3.2:latest "Your prompt here"
```

## Development

### Setup

```bash
# Install dependencies
poetry install

# Install pre-commit hooks (required for each developer)
poetry run pre-commit install

# Run tests
poetry run pytest

# Run linting and formatting
poetry run ruff check .
poetry run ruff format .
```

> **Note**: Each developer must run `poetry run pre-commit install` to enable automatic code quality checks on commit.
> The hooks will then run automatically before each commit to ensure code quality.

### Project Structure

```text
src/shardguard/
├── cli.py              # CLI interface
├── core/
│   ├── coordination.py # Main service
│   ├── models.py       # Data models
│   ├── planning.py     # LLM implementations
│   └── prompts.py      # Prompt templates
└── utils/
    └── logging.py      # Logging utilities
```

### Example Output

```json
{
  "original_prompt": "Setup server with password [PASSWORD]",
  "sub_prompts": [
    {
      "id": 1,
      "content": "Configure server infrastructure",
      "opaque_values": {}
    },
    {
      "id": 2,
      "content": "Setup admin access with [PASSWORD]",
      "opaque_values": {
        "[PASSWORD]": "secret123"
      }
    }
  ]
}
```

## License

Licensed under the Apache License 2.0. See `LICENSE` file for details.

## Security

This project deals with sensitive data processing. Please review the security considerations in `SECURITY.md`
before using in any production-like environment.
