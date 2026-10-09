from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from ..config import Config
from ..harness import SkillForgeHarness
from ..registry.store import SkillRegistry
from ..registry.export import export_to_huggingface
from ..llm import LLMError

logger = logging.getLogger("skillforge")


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(format="%(levelname)s %(name)s: %(message)s", level=level)


@click.group()
@click.version_option(package_name="skillforge")
def cli() -> None:
    """SkillForge — synthesize reusable skills from execution traces."""


@cli.command()
@click.argument("trace_file", type=click.Path(exists=True, path_type=Path))
@click.option("--model", envvar="SKILLFORGE_MODEL", help="Model override")
@click.option("--rounds", default=2, show_default=True)
@click.option("--no-save", is_flag=True)
@click.option("--output", "-o", type=click.Path(path_type=Path))
@click.option("--verbose", "-v", is_flag=True)
def generate(
    trace_file: Path,
    model: str | None,
    rounds: int,
    no_save: bool,
    output: Path | None,
    verbose: bool,
) -> None:
    """Synthesize a skill from TRACE_FILE."""
    _configure_logging(verbose)

    try:
        config = Config.from_env()
    except EnvironmentError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)

    if model:
        # Config is frozen; reconstruct with override
        import dataclasses
        config = dataclasses.replace(config, model=model, max_optimization_rounds=rounds)

    trace = SkillForgeHarness.load_trace(trace_file)
    harness = SkillForgeHarness(config)

    try:
        skill = harness.run(trace, save=not no_save)
    except LLMError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)

    click.echo(f"name:   {skill.name}")
    click.echo(f"score:  {skill.validation_score:.2f}")
    click.echo(f"id:     {skill.id}")

    if output:
        output.write_text(json.dumps(skill.to_dict(), indent=2, ensure_ascii=False))
        click.echo(f"saved:  {output}")


@cli.command("list")
@click.option("--min-score", default=0.0, show_default=True)
def list_skills(min_score: float) -> None:
    """List all skills in the local registry."""
    registry = SkillRegistry()
    skills = registry.list_all(min_score=min_score)

    if not skills:
        click.echo("No skills found.")
        return

    rows = [(s.id[:8], f"{s.validation_score:.2f}", s.version, s.name) for s in skills]
    id_w = max(len(r[0]) for r in rows)
    name_w = max(len(r[3]) for r in rows)

    click.echo(f"{'ID':<{id_w}}  {'SCORE':>5}  {'VER':>7}  NAME")
    click.echo("-" * (id_w + name_w + 18))
    for id_, score, ver, name in rows:
        click.echo(f"{id_:<{id_w}}  {score:>5}  {ver:>7}  {name}")

    click.echo(f"\n{len(skills)} skill(s)")


def _resolve_skill(registry: SkillRegistry, skill_id: str):
    """Lookup a skill by exact ID or unique prefix."""
    skill = registry.get(skill_id)
    if skill is not None:
        return skill

    matches = [s for s in registry.list_all() if s.id.startswith(skill_id)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        click.echo(f"Ambiguous prefix — matches: {[s.id[:8] for s in matches]}", err=True)
        sys.exit(1)
    return None


@cli.command()
@click.argument("skill_id")
def show(skill_id: str) -> None:
    """Print a skill's full JSON."""
    registry = SkillRegistry()
    skill = _resolve_skill(registry, skill_id)
    if skill is None:
        click.echo(f"Skill '{skill_id}' not found.", err=True)
        sys.exit(1)

    click.echo(json.dumps(skill.to_dict(), indent=2, ensure_ascii=False))


@cli.command()
@click.argument("skill_id")
@click.option("--verbose", "-v", is_flag=True)
def test(skill_id: str, verbose: bool) -> None:
    """Re-run validation on an existing skill."""
    _configure_logging(verbose)
    registry = SkillRegistry()
    skill = _resolve_skill(registry, skill_id)
    if skill is None:
        click.echo(f"Skill '{skill_id}' not found.", err=True)
        sys.exit(1)

    try:
        config = Config.from_env()
    except EnvironmentError as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)

    from ..core.validator import ValidatorAgent
    report = ValidatorAgent(config).validate(skill)

    click.echo(f"score:  {report.score:.2f}  ({'pass' if report.passed else 'fail'})")
    if report.failure_modes:
        click.echo("failures:")
        for f in report.failure_modes:
            click.echo(f"  - {f}")

    skill.validation_score = report.score
    registry.save(skill)


@cli.command()
@click.argument("output_dir", type=click.Path(path_type=Path))
@click.option("--min-score", default=0.5, show_default=True)
def export(output_dir: Path, min_score: float) -> None:
    """Export qualifying skills to a HuggingFace-compatible dataset."""
    skills = SkillRegistry().list_all(min_score=min_score)
    if not skills:
        click.echo("No qualifying skills to export.")
        sys.exit(1)
    out = export_to_huggingface(skills, output_dir, min_score=min_score)
    click.echo(f"exported {len(skills)} skill(s) to {out}")


@cli.command()
@click.argument("query")
def search(query: str) -> None:
    """Search skills by name or description."""
    results = SkillRegistry().search(query)
    if not results:
        click.echo(f"No results for '{query}'.")
        return
    for s in results:
        click.echo(f"{s.id[:8]}  {s.validation_score:.2f}  {s.name}")


@cli.command()
@click.argument("skill_id")
@click.option("--yes", is_flag=True, help="Skip confirmation")
def delete(skill_id: str, yes: bool) -> None:
    """Delete a skill from the registry."""
    registry = SkillRegistry()
    skill = _resolve_skill(registry, skill_id)
    if skill is None:
        click.echo(f"'{skill_id}' not found.", err=True)
        sys.exit(1)

    if not yes:
        click.confirm(f"Delete '{skill.name}' ({skill.id[:8]})?", abort=True)
    if registry.delete(skill.id):
        click.echo(f"deleted {skill.id}")
    else:
        click.echo(f"'{skill.id}' not found.", err=True)
        sys.exit(1)
