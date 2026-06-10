@echo off
title Claude merge runner
echo [Claude] commit fixes + fetch upstream + merge enhance ...
wsl bash -lc "cd /mnt/d/MAPLE-fix-pbc && { echo '=== branch/remotes'; git rev-parse --abbrev-ref HEAD; git remote -v; echo '=== commit local fixes'; git add -A; git commit -m 'fix: tomllib fallback for py3.10; harden image-flag float cast'; echo '=== fetch upstream'; git fetch upstream; echo '=== merge upstream/enhance'; git merge --no-ff upstream/enhance -m 'merge upstream/enhance into fix/pbc (pre-resolve conflicts)'; echo '=== status --porcelain'; git status --porcelain; echo '=== MERGE_SCRIPT_DONE'; } > /mnt/d/MAPLE-fix-pbc/.claude_merge_log.txt 2>&1"
echo [Claude] done, log: .claude_merge_log.txt
timeout /t 3 >nul
