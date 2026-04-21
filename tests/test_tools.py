"""Tests for handle_connect_datasource in tools.py — non-interactive path."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anton.chat import ChatSession
from anton.core.datasources.data_vault import LocalDataVault
from anton.core.session import ChatSessionConfig
from anton.tools import handle_connect_datasource


@pytest.fixture()
def vault_dir(tmp_path):
    return tmp_path / "vault"


def _make_session(vault_dir):
    """Return a minimal ChatSession wired for tool-handler tests."""
    from anton.core.llm.provider import ProviderConnectionInfo

    mock_llm = AsyncMock()
    mock_llm.coding_provider = MagicMock()
    mock_llm.coding_provider.export_connection_info = MagicMock(
        return_value=ProviderConnectionInfo(provider="anthropic", api_key="test")
    )
    mock_llm.coding_model = "claude-sonnet-4-6"
    session = ChatSession(ChatSessionConfig(llm_client=mock_llm))
    session._console = MagicMock()
    session._scratchpads = AsyncMock()
    session._data_vault = LocalDataVault(vault_dir=vault_dir)
    session._settings = None
    return session


def _fake_interactive(session):
    """Return an AsyncMock for the interactive handler that exits cleanly."""

    async def _side_effect(console, scratchpads, sess, **kwargs):
        sess._pending_connect_redirect = "handled by test"
        return sess

    return AsyncMock(side_effect=_side_effect)


class TestNonInteractiveSave:
    @pytest.mark.asyncio
    async def test_saves_with_name_from_field(self, vault_dir):
        """Full credential set with name_from field → saved under derived name, no prompts."""
        session = _make_session(vault_dir)
        result = await handle_connect_datasource(
            session,
            {
                "engine": "postgres",
                "known_variables": {
                    "host": "db.example.com",
                    "database": "sales",
                    "user": "admin",
                    "password": "x",
                },
            },
        )
        vault = session._data_vault
        conns = vault.list_connections()
        assert len(conns) == 1
        assert conns[0]["engine"] == "postgres"
        assert conns[0]["name"] == "sales"
        saved = vault.load("postgres", "sales")
        assert saved is not None
        assert saved["host"] == "db.example.com"
        assert saved["database"] == "sales"
        assert result.startswith("Saved connection")
        assert "sales" in result

    @pytest.mark.asyncio
    async def test_missing_name_from_falls_back_to_timestamp(self, vault_dir):
        """When the name_from field is absent, a numeric timestamp name is used."""
        session = _make_session(vault_dir)
        result = await handle_connect_datasource(
            session,
            {
                "engine": "postgres",
                "known_variables": {"host": "db.example.com"},
            },
        )
        vault = session._data_vault
        conns = vault.list_connections()
        assert len(conns) == 1
        assert conns[0]["engine"] == "postgres"
        assert conns[0]["name"].isdigit()
        saved = vault.load("postgres", conns[0]["name"])
        assert saved is not None
        assert saved["host"] == "db.example.com"
        assert result.startswith("Saved connection")

    @pytest.mark.asyncio
    async def test_unknown_engine_falls_through_to_interactive(self, vault_dir):
        """Engine not in registry → non-interactive branch is skipped, interactive called."""
        session = _make_session(vault_dir)
        interactive = _fake_interactive(session)
        with patch("anton.commands.datasource.handle_connect_datasource", new=interactive):
            await handle_connect_datasource(
                session,
                {"engine": "NotARealEngine", "known_variables": {"api_key": "x"}},
            )
        interactive.assert_called_once()
        assert session._data_vault.list_connections() == []

    @pytest.mark.asyncio
    async def test_no_field_overlap_falls_through_to_interactive(self, vault_dir):
        """known_variables with no matching engine fields → interactive called, nothing saved."""
        session = _make_session(vault_dir)
        interactive = _fake_interactive(session)
        with patch("anton.commands.datasource.handle_connect_datasource", new=interactive):
            await handle_connect_datasource(
                session,
                {"engine": "postgres", "known_variables": {"wrongfield": "foo"}},
            )
        interactive.assert_called_once()
        assert session._data_vault.list_connections() == []

    @pytest.mark.asyncio
    async def test_no_known_variables_runs_interactive(self, vault_dir):
        """No known_variables → interactive flow is invoked, no non-interactive save."""
        session = _make_session(vault_dir)
        interactive = _fake_interactive(session)
        with patch("anton.commands.datasource.handle_connect_datasource", new=interactive):
            await handle_connect_datasource(session, {"engine": "postgres"})
        interactive.assert_called_once()
        assert session._data_vault.list_connections() == []
