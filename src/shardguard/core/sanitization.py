"""Input sanitization utilities for ShardGuard."""

import re

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


class SanitizationResult:
    """Result of input sanitization process."""

    def __init__(
        self, sanitized_input: str, changes_made: list[str], original_length: int
    ):
        self.sanitized_input = sanitized_input
        self.changes_made = changes_made
        self.original_length = original_length
        self.final_length = len(sanitized_input)


class InputSanitizer:
    """Handles sanitization of user input to prevent injection attacks."""

    def __init__(self, console: Console | None = None, max_length: int = 10000):
        self.console = console or Console()
        self.max_length = max_length

        # Patterns for dangerous content removal
        self.dangerous_patterns = [
            (r"<script[^>]*>.*?</script>", "Script tags"),
            (r"javascript:", "JavaScript URLs"),
            (r"data:text/html", "HTML data URLs"),
            (r"(<.*?)(on\w+\s*=\s*['\"][^'\"]+['\"])", "Event handlers (e.g., onmouseover, onclick, etc.)"),
        ]

    def sanitize(
        self, user_input: str, show_progress: bool = True
    ) -> SanitizationResult:
        """
        Sanitize user input to prevent injection attacks while preserving legitimate content.

        Args:
            user_input: The input string to sanitize
            show_progress: Whether to display sanitization progress to console

        Returns:
            SanitizationResult containing sanitized input and metadata

        Raises:
            ValueError: If input is empty or only whitespace
        """
        if show_progress:
            self._show_sanitization_start()
            self._show_original_input(user_input)

        if not user_input or not user_input.strip():
            if show_progress:
                self.console.print(
                    "[bold red]❌ Error: User input cannot be empty[/bold red]"
                )
            raise ValueError("User input cannot be empty")

        changes_made = []
        original_length = len(user_input)
        sanitized = user_input

        # Apply sanitization steps
        sanitized, step_changes = self._normalize_whitespace(sanitized)
        changes_made.extend(step_changes)

        sanitized, step_changes = self._remove_control_characters(sanitized)
        changes_made.extend(step_changes)

        sanitized, step_changes = self._truncate_long_input(sanitized)
        changes_made.extend(step_changes)

        sanitized, step_changes = self._remove_dangerous_patterns(sanitized)
        changes_made.extend(step_changes)

        result = SanitizationResult(sanitized, changes_made, original_length)

        if show_progress:
            self._show_sanitization_results(result, user_input)

        return result

    def _normalize_whitespace(self, text: str) -> tuple[str, list[str]]:
        """Normalize whitespace and line endings."""
        original = text.strip()
        normalized = re.sub(r"\s+", " ", original)

        changes = []
        if normalized != original:
            changes.append("✓ Normalized whitespace and line endings")

        return normalized, changes

    def _remove_control_characters(self, text: str) -> tuple[str, list[str]]:
        """Remove potentially dangerous control characters but preserve important ones."""
        before_removal = text
        # Remove dangerous control characters but keep normal whitespace
        cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

        changes = []
        if cleaned != before_removal:
            changes.append("✓ Removed dangerous control characters")

        return cleaned, changes

    def _truncate_long_input(self, text: str) -> tuple[str, list[str]]:
        """Raise an error for extremely long inputs to prevent DoS."""
        if len(text) > self.max_length:
            raise ValueError(
                f"Input truncated: exceeds maximum length of {self.max_length} characters"
            )

        return text, []

    def _remove_dangerous_patterns(self, text: str) -> tuple[str, list[str]]:
        """Remove obvious injection attempts while preserving legitimate content."""
        changes = []

        for pattern, description in self.dangerous_patterns:
            before_removal = text
            text = re.sub(
                pattern, "", text, flags=re.IGNORECASE | re.DOTALL
            )  # Completely remove matches
            if text != before_removal:
                changes.append(f"✓ Removed {description}")

        return text, changes

    def _show_sanitization_start(self):
        """Display sanitization process header."""
        self.console.print("\n[bold blue]🔍 Input Sanitization Process[/bold blue]")

    def _show_original_input(self, user_input: str):
        """Display the original input."""
        display_input = user_input[:200] + ("..." if len(user_input) > 200 else "")
        original_panel = Panel(
            display_input,
            title="[bold]Original Input[/bold]",
            border_style="dim",
        )
        self.console.print(original_panel)

    def _show_sanitization_results(
        self, result: SanitizationResult, original_input: str
    ):
        """Display sanitization results and changes."""
        # Show sanitization changes
        if result.changes_made:
            changes_text = Text()
            for change in result.changes_made:
                changes_text.append(change + "\n", style="green")

            changes_panel = Panel(
                changes_text,
                title="[bold green]Sanitization Changes[/bold green]",
                border_style="green",
            )
            self.console.print(changes_panel)
        else:
            self.console.print(
                "[green]✓ No sanitization needed - input is clean[/green]"
            )

        # Show final sanitized input if different
        if result.sanitized_input != original_input:
            display_sanitized = result.sanitized_input[:200] + (
                "..." if len(result.sanitized_input) > 200 else ""
            )
            sanitized_panel = Panel(
                display_sanitized,
                title="[bold]Sanitized Input[/bold]",
                border_style="green",
            )
            self.console.print(sanitized_panel)

        # Show length comparison
        if result.final_length != result.original_length:
            self.console.print(
                f"[dim]Length: {result.original_length} → {result.final_length} characters[/dim]"
            )

    def add_dangerous_pattern(self, pattern: str, description: str):
        """Add a new dangerous pattern to check for."""
        self.dangerous_patterns.append((pattern, description))

    def remove_dangerous_pattern(self, pattern: str):
        """Remove a dangerous pattern from the check list."""
        self.dangerous_patterns = [
            (p, d) for p, d in self.dangerous_patterns if p != pattern
        ]
