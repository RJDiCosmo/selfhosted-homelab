"""The UI's frontend authenticates API calls with `Authorization: Bearer <jwt>`.
Behind a reverse proxy's Basic Auth that header REPLACES the browser's cached Basic
credentials on every fetch, so the proxy 401s each API call and the browser keeps
re-prompting. Send the JWT in `X-Auth-Token` instead; nginx (nginx.conf) maps it
back to `Authorization: Bearer` for the backend. Fails the build if too few
call sites are found, so a UI bump can't silently skip it."""
import pathlib, re

root = pathlib.Path("/app")
files = [root / "App.tsx", root / "services/api.ts", *sorted((root / "components").glob("*.tsx"))]
pat_idx = re.compile(r"headers\['Authorization'\]\s*=\s*`Bearer \$\{(\w+)\}`")
pat_obj = re.compile(r"""['"]?Authorization['"]?\s*:\s*`Bearer \$\{([^}]+)\}`""")
total = 0
for f in files:
    s = f.read_text()
    s, n1 = pat_idx.subn(r"headers['X-Auth-Token'] = \1", s)
    s, n2 = pat_obj.subn(r"'X-Auth-Token': \1", s)
    if n1 + n2:
        f.write_text(s)
        total += n1 + n2
print("rewrote", total, "Authorization call sites")
assert total >= 12, f"expected ~13 call sites, found {total}"
