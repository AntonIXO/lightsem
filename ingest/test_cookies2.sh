#!/usr/bin/env bash
# Does yt-dlp actually DECRYPT the cookies, and are we authenticated?
set -u
export PATH="/root/lightsem/.venv/bin:$PATH"
WORK="/tmp/brave_prof"
OUT="/root/lightsem/ingest/cookies.txt"

echo "=== A. export cookies to netscape file (proves decryption) ==="
yt-dlp --cookies-from-browser "brave:$WORK" --cookies "$OUT" \
       --skip-download --no-warnings --simulate \
       "https://www.youtube.com/watch?v=dQw4w9WgXcQ" 2>&1 | tail -6

echo
if [ -f "$OUT" ]; then
  echo "  cookie file written: $(wc -l < "$OUT") lines"
  echo "  youtube auth cookies present:"
  grep -cE "SID|LOGIN_INFO|SAPISID" "$OUT" 2>/dev/null | sed 's/^/    count=/'
  # Are values real (decrypted) or empty/garbage?
  awk -F'\t' '$6=="SID"{print "    SID value length: " length($7)}' "$OUT" 2>/dev/null | head -1
  awk -F'\t' '$6=="LOGIN_INFO"{print "    LOGIN_INFO length: " length($7)}' "$OUT" 2>/dev/null | head -1
else
  echo "  NO COOKIE FILE -> decryption failed"
fi

echo
echo "=== B. authenticated? (yt-dlp reports account when cookies work) ==="
yt-dlp --cookies "$OUT" --skip-download --no-warnings \
       --print "title=%(title).50s | dur=%(duration)ss | id=%(id)s" \
       "https://www.youtube.com/watch?v=dQw4w9WgXcQ" 2>&1 | tail -4
