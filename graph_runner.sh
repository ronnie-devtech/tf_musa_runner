#!/usr/bin/env bash
set -uo pipefail

DEFAULT_SPEC="./meta_graph/meta_graph_3.spec"
DEFAULT_MUSA_PLUGIN="../tensorflow_musa_extension/build/libmusa_plugin.so"
DEFAULT_DEVICE="0"
DEFAULT_RUNNER_SCRIPT="./musa_run_pb_graph.py"
DEFAULT_WORKDIR="."
DEFAULT_LOG_DIR="log"

usage() {
  cat <<'EOF'
Usage:
  bash ./test_pb.sh --all [--repeat N] [--averge] [--no-averge] [--spec PATH] [--musa-plugin PATH] [--device ID]
  bash ./test_pb.sh --single BS [--repeat N] [--averge] [--no-averge] [--spec PATH] [--musa-plugin PATH] [--device ID]
  bash ./test_pb.sh --sigle BS [--repeat N] [--averge] [--no-averge] [--spec PATH] [--musa-plugin PATH] [--device ID]

Options:
  --all                    Run bs=1,2,4,...,4096
  --single, --sigle BS     Run only one batch size
  --repeat N               Repeat count for each bs. Default: 5
  --averge, --average      Print the average over repeat results. Default: enabled
  --no-averge, --no-average
                           Disable repeat average output
  --spec PATH              Default: ./meta_graph/meta_graph_3.spec
  --musa-plugin PATH       Default: ../tensorflow_musa_extension/build/libmusa_plugin.so
  --device ID              Default: 0
  --runner-script PATH     Default: ./musa_run_pb_graph.py
  --workdir PATH           Default: current directory
  --log-dir PATH           Default: ./log
EOF
}

MODE=""
SINGLE_BS=""
SPEC="$DEFAULT_SPEC"
MUSA_PLUGIN="$DEFAULT_MUSA_PLUGIN"
DEVICE="$DEFAULT_DEVICE"
RUNNER_SCRIPT="$DEFAULT_RUNNER_SCRIPT"
WORKDIR="$DEFAULT_WORKDIR"
LOG_DIR="$DEFAULT_LOG_DIR"
REPEAT_COUNT="5"
AVERAGE_ENABLED="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)
      MODE="all"
      shift
      ;;
    --single|--sigle)
      MODE="single"
      SINGLE_BS="${2:-}"
      if [[ -z "$SINGLE_BS" ]]; then
        echo "error: $1 requires a batch size value" >&2
        usage
        exit 1
      fi
      shift 2
      ;;
    --spec)
      SPEC="${2:-}"
      shift 2
      ;;
    --repeat)
      REPEAT_COUNT="${2:-}"
      shift 2
      ;;
    --averge|--average)
      AVERAGE_ENABLED="1"
      shift
      ;;
    --no-averge|--no-average)
      AVERAGE_ENABLED="0"
      shift
      ;;
    --musa-plugin)
      MUSA_PLUGIN="${2:-}"
      shift 2
      ;;
    --device)
      DEVICE="${2:-}"
      shift 2
      ;;
    --runner-script)
      RUNNER_SCRIPT="${2:-}"
      shift 2
      ;;
    --workdir)
      WORKDIR="${2:-}"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$MODE" ]]; then
  echo "error: one of --all or --single/--sigle is required" >&2
  usage
  exit 1
fi

if ! [[ "$REPEAT_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: --repeat must be a positive integer" >&2
  exit 1
fi

if [[ "$MODE" == "single" ]] && ! [[ "$SINGLE_BS" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: batch size must be a positive integer" >&2
  exit 1
fi

if [[ "$MODE" == "all" ]]; then
  BATCH_SIZES=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096)
else
  BATCH_SIZES=("$SINGLE_BS")
fi

WORKDIR="$(cd "$WORKDIR" && pwd)"

if [[ "$LOG_DIR" = /* ]]; then
  ACTUAL_LOG_DIR="$LOG_DIR"
else
  ACTUAL_LOG_DIR="$WORKDIR/$LOG_DIR"
fi

mkdir -p "$ACTUAL_LOG_DIR"

SUMMARY_ROWS=()
FAILED_COUNT=0

extract_average_time() {
  local log_path="$1"
  local value
  value="$(
    grep -oE "average_time_ms['\"]?[[:space:]]*:[[:space:]]*[0-9]+([.][0-9]+)?" "$log_path" \
      | tail -1 \
      | grep -oE "[0-9]+([.][0-9]+)?$" \
      || true
  )"
  printf '%s' "$value"
}

extract_trimmed_avg_time() {
  local log_path="$1"
  local value
  value="$(
    grep -oE "trimmed_avg_ms['\"]?[[:space:]]*:[[:space:]]*[0-9]+([.][0-9]+)?" "$log_path" \
      | tail -1 \
      | grep -oE "[0-9]+([.][0-9]+)?$" \
      || true
  )"
  printf '%s' "$value"
}

join_by_comma() {
  local out=""
  local item
  for item in "$@"; do
    out+="${out:+, }${item}"
  done
  printf '%s' "$out"
}

compute_average() {
  if [[ $# -eq 0 ]]; then
    printf '%s' ""
    return
  fi

  printf '%s\n' "$@" | awk '
    BEGIN {
      sum = 0;
      count = 0;
    }
    {
      sum += $1;
      count += 1;
    }
    END {
      if (count == 0) {
        exit 1;
      }
      printf "%.6f", sum / count;
    }
  '
}

for bs in "${BATCH_SIZES[@]}"; do
  log_path="$ACTUAL_LOG_DIR/bs_${bs}.log"
  : > "$log_path"

  bs_status="ok"
  bs_values=()
  bs_trimmed_values=()
  bs_display_values=()
  bs_trimmed_display_values=()
  bs_average=""
  bs_trimmed_average=""

  for ((repeat_idx = 1; repeat_idx <= REPEAT_COUNT; repeat_idx++)); do
    tmp_log="$(mktemp "${TMPDIR:-/tmp}/test_pb.XXXXXX")"
    status="ok"
    avg_ms=""
    trimmed_avg_ms=""

    if (
      cd "$WORKDIR" && \
      MUSA_PINNED_FEED=1 \
      MUSA_PINNED_H2D_ON_COMPUTE_STREAM=1 \
      MUSA_VISIBLE_DEVICES="$DEVICE" \
      python3 "$RUNNER_SCRIPT" \
        --spec "$SPEC" \
        --musa-plugin "$MUSA_PLUGIN" \
        --bs "$bs" \
        --run_iters 20 \
        >"$tmp_log" 2>&1
    ); then
      avg_ms="$(extract_average_time "$tmp_log")"
      trimmed_avg_ms="$(extract_trimmed_avg_time "$tmp_log")"

      if [[ -z "$avg_ms" ]]; then
        status="failed"
      fi
      if [[ -z "$trimmed_avg_ms" ]]; then
        trimmed_avg_ms="N/A"
      fi
    else
      status="failed"
      trimmed_avg_ms="N/A"
    fi

    {
      printf '===== bs=%s repeat=%s/%s =====\n' "$bs" "$repeat_idx" "$REPEAT_COUNT"
      cat "$tmp_log"
      printf '\n'
    } >> "$log_path"
    rm -f "$tmp_log"

    if [[ "$status" == "ok" ]]; then
      bs_values+=("$avg_ms")
      bs_display_values+=("$avg_ms")

      if [[ "$trimmed_avg_ms" != "N/A" ]]; then
        bs_trimmed_values+=("$trimmed_avg_ms")
        bs_trimmed_display_values+=("$trimmed_avg_ms")
      else
        bs_trimmed_display_values+=("N/A")
      fi
    else
      FAILED_COUNT=$((FAILED_COUNT + 1))
      bs_status="failed"
      bs_display_values+=("FAILED")
      bs_trimmed_display_values+=("FAILED")
    fi

    current_text="${avg_ms:-FAILED}"
    current_trimmed_text="${trimmed_avg_ms:-FAILED}"
    echo "bs=${bs} status=${status} average_time_ms={${current_text}} trimmed_avg_ms={${current_trimmed_text}} log=${log_path}"
  done

  if [[ "$AVERAGE_ENABLED" == "1" ]]; then
    if [[ "${#bs_values[@]}" -gt 0 ]]; then
      bs_average="$(compute_average "${bs_values[@]}")"
      echo "average_repeat={${bs_average}}"
    else
      echo "average_repeat={N/A}"
    fi

    if [[ "${#bs_trimmed_values[@]}" -gt 0 ]]; then
      bs_trimmed_average="$(compute_average "${bs_trimmed_values[@]}")"
      echo "trimmed_avg_repeat={${bs_trimmed_average}}"
    else
      echo "trimmed_avg_repeat={N/A}"
    fi
  fi

  SUMMARY_ROWS+=("${bs}|${bs_status}|$(join_by_comma "${bs_values[@]}")|$(join_by_comma "${bs_display_values[@]}")|$(join_by_comma "${bs_trimmed_display_values[@]}")|${bs_average}|${bs_trimmed_average}|${log_path}")
  echo
done

echo "Latency Summary"
printf '%s\n' "========================================================================================================================"
printf "%8s  %8s  %24s  %24s  %16s  %18s  %s\n" "bs" "status" "average_time_ms" "trimmed_avg_ms" "average_repeat" "trimmed_avg_repeat" "log"
printf '%s\n' "------------------------------------------------------------------------------------------------------------------------"

FINAL_DATA="{"
SEP=""
for row in "${SUMMARY_ROWS[@]}"; do
  IFS='|' read -r bs status avg_values display_values trimmed_display_values avg_repeat trimmed_avg_repeat log_path <<<"$row"

  avg_text="{${display_values}}"
  trimmed_text="{${trimmed_display_values}}"
  avg_repeat_text="${avg_repeat:-N/A}"
  trimmed_avg_repeat_text="${trimmed_avg_repeat:-N/A}"

  printf "%8s  %8s  %24s  %24s  %16s  %18s  %s\n" \
    "$bs" "$status" "$avg_text" "$trimmed_text" "$avg_repeat_text" "$trimmed_avg_repeat_text" "$log_path"

  if [[ -n "$display_values" || -n "$trimmed_display_values" ]]; then
    if [[ "$AVERAGE_ENABLED" == "1" ]]; then
      FINAL_DATA+="${SEP}${bs}: {values=[${display_values}], trimmed_values=[${trimmed_display_values}], average_repeat=${avg_repeat_text}, trimmed_avg_repeat=${trimmed_avg_repeat_text}}"
    else
      FINAL_DATA+="${SEP}${bs}: {values=[${display_values}], trimmed_values=[${trimmed_display_values}]}"
    fi
    SEP=", "
  fi
done
FINAL_DATA+="}"

printf '%s\n' "------------------------------------------------------------------------------------------------------------------------"
if [[ "$FINAL_DATA" != "{}" ]]; then
  echo "Final performance data:"
  echo "$FINAL_DATA"
else
  echo "No successful runs. Please check logs under: $ACTUAL_LOG_DIR"
fi

if [[ "$FAILED_COUNT" -gt 0 ]]; then
  exit 1
fi
