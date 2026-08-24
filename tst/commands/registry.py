"""Command lookup and the default TST command set."""

from __future__ import annotations

import json
from collections.abc import Iterable

from tst.service.service import TSTService

from .models import CommandResult, CommandSpec, ParsedCommand
from .parser import parse_command, suggest_commands


class CommandRegistry:
    def __init__(self, specs: Iterable[CommandSpec] = ()) -> None:
        self._specs: dict[str, CommandSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: CommandSpec) -> None:
        name = spec.name.strip().casefold()
        if not name or any(char.isspace() for char in name) or name.startswith("/"):
            raise ValueError("command names must be non-empty single tokens")
        self._specs[name] = CommandSpec(name, spec.description, spec.handler, spec.aliases)
        for alias in spec.aliases:
            self._specs[alias.casefold()] = self._specs[name]

    def get(self, name: str) -> CommandSpec | None:
        return self._specs.get(name.casefold())

    def specs(self) -> list[CommandSpec]:
        return sorted(set(self._specs.values()), key=lambda spec: spec.name)

    def suggestions(self, prefix: str) -> list[CommandSpec]:
        return suggest_commands(prefix, self.specs())

    def execute(self, text: str) -> CommandResult:
        parsed = parse_command(text)
        spec = self.get(parsed.name)
        if spec is None:
            raise ValueError(f"unknown TST command: /{parsed.name}")
        return spec.handler(parsed)


def default_registry(service: TSTService) -> CommandRegistry:
    def result(data, output: str | None = None) -> CommandResult:
        return CommandResult(output or json.dumps(data, indent=2, default=str), data)

    def status(_: ParsedCommand) -> CommandResult:
        return result(service.status())

    def context(command: ParsedCommand) -> CommandResult:
        if not command.args:
            raise ValueError("usage: /context <query>")
        pack = service.retrieve_context(" ".join(command.args))
        return CommandResult(pack.as_prompt(), pack.model_dump(mode="json"))

    def memory(command: ParsedCommand) -> CommandResult:
        if not command.args:
            data = service.list_memories()
        else:
            data = [item.model_dump(mode="json") for item in service.search_memories(" ".join(command.args))]
        return result(data)

    def tree(command: ParsedCommand) -> CommandResult:
        if not command.args:
            raise ValueError("usage: /tree <symbol>")
        data = service.query_symbol(" ".join(command.args))
        return result(data)

    def scope(command: ParsedCommand) -> CommandResult:
        value = command.args[0].casefold() if command.args else "project"
        if value not in {"global", "project", "session"}:
            raise ValueError("scope must be global, project, or session")
        return CommandResult(value, {"scope": value})

    def reindex(_: ParsedCommand) -> CommandResult:
        data = service.index_project()
        return result(data)

    def forget(command: ParsedCommand) -> CommandResult:
        if not command.args:
            raise ValueError("usage: /forget <canonical-key>")
        data = service.forget_memory(command.args[0])
        return result(data)

    def doctor(_: ParsedCommand) -> CommandResult:
        return result(service.doctor())

    def connect(command: ParsedCommand) -> CommandResult:
        if not command.args:
            raise ValueError("usage: /connect claude|codex")
        data = service.install_integration(command.args[0])
        return result(data)

    def init(_: ParsedCommand) -> CommandResult:
        return CommandResult("Project is ready", service.project.to_dict())

    return CommandRegistry(
        [
            CommandSpec("status", "Show project and kernel health", status),
            CommandSpec("context", "Inspect retrieved context", context),
            CommandSpec("memory", "Search or list memories", memory),
            CommandSpec("tree", "Query a project symbol", tree),
            CommandSpec("scope", "Inspect the active scope", scope),
            CommandSpec("reindex", "Index changed project files", reindex),
            CommandSpec("forget", "Forget one canonical memory key", forget),
            CommandSpec("doctor", "Diagnose the local installation", doctor),
            CommandSpec("connect", "Install an agent integration", connect),
            CommandSpec("init", "Show project initialization state", init),
        ]
    )
