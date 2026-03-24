#!/bin/zsh
# Patch RunAnywhere SDK for iOS 26.2+ compression_stream() API compatibility
# This fixes the `compression_stream()` zero-arg initializer issue in Xcode 26.2+

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IOS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ -n "${1:-}" ]]; then
  DERIVED_DATA_ROOT="$1"
elif [[ -n "${BUILD_ROOT:-}" && "$BUILD_ROOT" == *"/Build/"* ]]; then
  DERIVED_DATA_ROOT="${BUILD_ROOT%%/Build/*}"
else
  DERIVED_DATA_ROOT="$HOME/Library/Developer/Xcode/DerivedData"
fi

patched=0

ROOTS=(
  "$DERIVED_DATA_ROOT/SourcePackages/checkouts/runanywhere-sdks"
  "$IOS_ROOT/runanywhere-sdks-latest"
)

for root in "${ROOTS[@]}"; do
  if [[ ! -d "$root" ]]; then
    continue
  fi

  SWIFT_ARCHIVE_UTILITY="$root/sdk/runanywhere-swift/Sources/RunAnywhere/Infrastructure/Download/Utilities/ArchiveUtility.swift"
  RN_ARCHIVE_UTILITY="$root/sdk/runanywhere-react-native/packages/core/ios/ArchiveUtility.swift"

  for file in "$SWIFT_ARCHIVE_UTILITY" "$RN_ARCHIVE_UTILITY"; do
    if [[ ! -f "$file" ]]; then
      continue
    fi

    # Patch original zero-arg initializer.
    perl -0pi -e 's/var stream = compression_stream\(\)/let dummyStreamPointer = UnsafeMutablePointer<UInt8>.allocate(capacity: 1)\n        defer { dummyStreamPointer.deallocate() }\n        var stream = compression_stream(\n            dst_ptr: dummyStreamPointer,\n            dst_size: 0,\n            src_ptr: UnsafePointer(dummyStreamPointer),\n            src_size: 0,\n            state: nil\n        )/g' "$file"

    # Upgrade previous nil-based patch to valid non-optional pointers.
    perl -0pi -e 's/var stream = compression_stream\(\n\s*dst_ptr: nil,\n\s*dst_size: 0,\n\s*src_ptr: nil,\n\s*src_size: 0,\n\s*state: nil\n\s*\)/let dummyStreamPointer = UnsafeMutablePointer<UInt8>.allocate(capacity: 1)\n        defer { dummyStreamPointer.deallocate() }\n        var stream = compression_stream(\n            dst_ptr: dummyStreamPointer,\n            dst_size: 0,\n            src_ptr: UnsafePointer(dummyStreamPointer),\n            src_size: 0,\n            state: nil\n        )/g' "$file"

    # New SDKs require Int32 flags for compression_stream_process.
    perl -0pi -e 's/compression_stream_process\(&stream, COMPRESSION_STREAM_FINALIZE\)/compression_stream_process(&stream, Int32(COMPRESSION_STREAM_FINALIZE.rawValue))/g' "$file"

    if grep -q 'dummyStreamPointer' "$file"; then
      echo "[PATCH] Fixed: $file"
      patched=$((patched + 1))
    fi
  done
done

if [ $patched -gt 0 ]; then
  echo "[PATCH] Successfully patched $patched file(s)"
  exit 0
else
  echo "[PATCH] No files needed patching"
  exit 0
fi
