#!/bin/sh
set -e
nginx -g 'daemon off;' &
cd /app/server
# exec tsx directly (not via npx/npm) so docker stop's SIGTERM reaches node and the
# container exits cleanly instead of with npm's "command failed" exit 1.
exec ./node_modules/.bin/tsx src/index.ts
