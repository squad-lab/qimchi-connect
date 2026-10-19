#!/usr/bin/env bash
# Update citation versions and the README citation year before the release build.
set -euo pipefail
: "${NEW_VERSION:?NEW_VERSION is set by semantic-release}"
sed -i "s/^version: .*/version: ${NEW_VERSION}/" CITATION.cff
sed -i -E "s/(note += \{Version )[^,]*/\1${NEW_VERSION}/; s/(year += \{)[0-9]{4}/\1$(date +%Y)/" README.md
