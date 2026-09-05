#!/usr/bin/env python3
"""Small runtime-only Pier adapters shared by all task images."""

from __future__ import annotations

from pier.agents.installed.codex import Codex
from pier.agents.installed.mini_swe_agent import MiniSweAgent
from pier.models.agent.install import AgentInstallSpec, InstallStep


class ScienceBenchCodex(Codex):
    """Use npm's optional platform package when Pier installs Codex."""

    def install_spec(self):
        spec = super().install_spec()
        for step in spec.steps:
            step.run = step.run.replace(
                "npm install -g @openai/codex",
                "npm install -g --include=optional @openai/codex",
            )
        return spec


class ScienceBenchMini(MiniSweAgent):
    """Use a prebuilt mini-swe-agent image in offline task environments."""

    def install_spec(self):
        return AgentInstallSpec(
            agent_name=self.name(),
            version=self._version,
            steps=[
                InstallStep(
                    user="root",
                    run="command -v mini-swe-agent >/dev/null",
                )
            ],
            verification_command=self.get_version_command(),
        )
