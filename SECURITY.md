# Security Policy

## Supported versions

Only the latest release on `main` receives security fixes.

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Email `rohmishra12@users.noreply.github.com` with:
- A description of the vulnerability
- Steps to reproduce
- Your assessment of impact and severity

You'll receive an acknowledgement within 48 hours. If the issue is confirmed, a fix will be
prioritised and you'll be credited in the release notes unless you prefer otherwise.

## Known security considerations

### Prompt injection via trace files

Trace content is passed verbatim into LLM prompts. A malicious trace file could attempt to
inject instructions that manipulate the synthesis agents — for example, a step whose content
contains "ignore previous instructions."

**Mitigation:** only load traces from sources you trust. `load_trace` validates step roles
and enforces a 512 KB file size limit, but does not sanitize content for prompt injection.
This is an inherent limitation of LLM-based pipelines.

### API key handling

SkillForge reads your API key from the `SKILLFORGE_API_KEY` or `OPENAI_API_KEY` environment
variable. It is never written to disk, logged, or included in skill artifacts.

Never commit `.env` files or hardcode keys in trace files. The `.gitignore` covers `.env`
and `.env.*` by default.

### Local registry

Skills are stored in `~/.skillforge/skills.db` (SQLite). This file may contain sensitive
workflow data depending on the traces you process. Restrict access appropriately if the
machine is shared.

The registry rejects `db_path` values outside the user home directory to prevent path
traversal when the path is user-supplied.

### Template injection

`Skill.render()` uses simple `str.replace()` substitution — not a template engine. This is
intentional. Do not upgrade to Jinja2 or similar without auditing for server-side template
injection, particularly if skills are ever rendered in a web context.
