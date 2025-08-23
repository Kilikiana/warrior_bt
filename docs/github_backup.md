# Automatic GitHub Backups (VS Code + macOS)

This guide sets up automatic backups of `warrior_bt` to GitHub with a simple background job. It plays nicely with VS Code (manual commits still work) and avoids committing large, local-only files.

## 1) Verify Git and Remote

```bash
cd /Users/<you>/Projects/warrior_bt
git status
git remote -v
```

If the remote is missing, add it and set default branch:

```bash
# Choose ONE of these remotes:
# SSH (recommended)
git remote add origin git@github.com:Kilikiana/warrior_bt.git

# or HTTPS
git remote add origin https://github.com/Kilikiana/warrior_bt.git

git branch -M main
```

## 2) Authenticate (choose one)

### Option A — SSH (recommended)
```bash
ssh-keygen -t ed25519 -C "you@example.com"
# Press enter to accept defaults; creates ~/.ssh/id_ed25519 and id_ed25519.pub

# Start SSH agent and add your key
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Copy the public key and add it to GitHub → Settings → SSH and GPG keys
cat ~/.ssh/id_ed25519.pub

# Test connection
ssh -T git@github.com
```

### Option B — HTTPS + Personal Access Token
- Create a PAT (repo scope) at GitHub → Settings → Developer settings → Personal access tokens
- First push will prompt; macOS Keychain will remember it.

## 3) Ignore Large/Local Files

Create or update `.gitignore` to avoid committing big caches and local venv logs:

```
__pycache__/
*.pyc
.DS_Store
venv/
.venv/
shared_cache/
results/logs/
finrobot_test/venv/
```

Tip: If you need to version only certain artifacts, ignore subtrees like `shared_cache/ohlcv_*` but keep `results/hod_momentum_scans/`.

## 4) Initial Commit/Push

```bash
git config user.name "Your Name"
git config user.email "you@example.com"

git add -A
git commit -m "init backup"
git push -u origin main
```

## 5) Automatic Backups (macOS launchd)

Create a small script that commits and pushes only when there are changes:

Create file: `.scripts/auto_backup.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."  # repo root

# Only commit if there are changes
if [[ -n "$(git status --porcelain)" ]]; then
  git add -A
  git commit -m "auto-backup: $(date '+%Y-%m-%d %H:%M:%S')"
  git push origin main
fi
```

Make it executable:

```bash
chmod +x .scripts/auto_backup.sh
```

Create a LaunchAgent to run the script every 10 minutes:

File: `~/Library/LaunchAgents/com.kilikiana.warriorbt.autobackup.plist`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key>
    <string>com.kilikiana.warriorbt.autobackup</string>

    <key>ProgramArguments</key>
    <array>
      <string>/bin/bash</string>
      <string>/Users/<you>/Projects/warrior_bt/.scripts/auto_backup.sh</string>
    </array>

    <key>WorkingDirectory</key>
    <string>/Users/<you>/Projects/warrior_bt</string>

    <key>StartInterval</key>
    <integer>600</integer>

    <key>StandardOutPath</key>
    <string>/tmp/warrior_bt_autobackup.out</string>
    <key>StandardErrorPath</key>
    <string>/tmp/warrior_bt_autobackup.err</string>
  </dict>
</plist>
```

Load and start the agent:

```bash
launchctl load -w ~/Library/LaunchAgents/com.kilikiana.warriorbt.autobackup.plist
# Verify
launchctl list | grep warriorbt
```

Manual test:

```bash
/Users/<you>/Projects/warrior_bt/.scripts/auto_backup.sh
```

Notes:
- Ensure your SSH key is in the agent or your HTTPS PAT is stored, so push won’t prompt.
- You can adjust `StartInterval` (seconds) to your preference.

### Alternative: Cron (simpler, less Mac‑native)

```bash
crontab -e
# Add this line (every 10 minutes):
*/10 * * * * cd /Users/<you>/Projects/warrior_bt && /bin/bash .scripts/auto_backup.sh >> /tmp/warrior_bt_backup.log 2>&1
```

## 6) VS Code Integration (Optional)
- Use the Git panel as usual for manual commits/pushes; the auto‑backup script only runs when there are changes.
- Helpful extensions:
  - GitLens — rich history and blame
  - Run on Save or Git Auto Commit — if you want on-save automation (push still manual)

## 7) Safety & Best Practices
- Keep large data out of Git (use `.gitignore` or Git LFS for huge files).
- Use clear messages for manual commits; auto‑backup adds timestamped messages.
- If you collaborate, protect `main` with branch rules; adjust the auto‑backup to use a feature branch if needed.

## Troubleshooting
- `git push` prompts for credentials: set up SSH agent (`ssh-add`) or store HTTPS PAT in Keychain.
- LaunchAgent not running: check `launchctl list`, inspect `/tmp/warrior_bt_autobackup.err`.
- Script not executable: `chmod +x .scripts/auto_backup.sh`.
- Wrong paths: update `<you>` paths in the plist and commands to your actual username.

---

Once configured, your local changes are committed and pushed automatically on a schedule, while you continue to work in VS Code as usual.
