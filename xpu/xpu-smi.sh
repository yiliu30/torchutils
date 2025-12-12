#!/bin/bash
# @xin & chatgpt
# intel-smi - PARALLEL version: query all devices concurrently

# --- Step 1: Get all device info in ONE call ---
discovery_output=$(xpu-smi discovery 2>/dev/null)
if [ -z "$discovery_output" ] || ! echo "$discovery_output" | grep -q "| [0-9]"; then
    echo "No XPU devices found."
    exit 1
fi

# Parse IDs and clean names
declare -A device_names
device_ids=()
while IFS= read -r line; do
    if [[ $line =~ ^[[:space:]]*\|[[:space:]]*([0-9]+)[[:space:]]*\|[[:space:]]*Device[[:space:]]Name:[[:space:]]*(.*)$ ]]; then
        id="${BASH_REMATCH[1]}"
        name_raw="${BASH_REMATCH[2]}"
        name=$(echo "$name_raw" | sed 's/ \[0x[0-9a-fA-F]*\]$//' | xargs)
        [ ${#name} -gt 22 ] && name="${name:0:19}..."
        device_ids+=("$id")
        device_names[$id]="$name"
    fi
done <<< "$discovery_output"

# --- Step 2: Parallel stats collection ---
MEM_TOTAL="16384"
TMPDIR=$(mktemp -d)
declare -A results

# Launch all stats queries in parallel
for id in "${device_ids[@]}"; do
    (
        # Get stats for this device
        stats=$(timeout 3 xpu-smi stats -d "$id" 2>/dev/null)
        
        # Extract fields
        temp="--"; power="--"; util="--"; mem_used="--"; mem_bw="--"; freq="--"
        while IFS= read -r line; do
            if [[ $line == *"|"* ]] && [[ $line != *"Device ID"* ]]; then
                key=$(echo "$line" | awk -F'|' '{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}')
                val=$(echo "$line" | awk -F'|' '{gsub(/^[ \t]+|[ \t]+$/, "", $3); print $3}')
                case "$key" in
                    "GPU Core Temperature (C)") temp="$val" ;;
                    "GPU Power (W)") power="$val" ;;
                    "GPU Utilization (%)") util="$val" ;;
                    "GPU Memory Used (MiB)") mem_used="$val" ;;
                    "GPU Memory Bandwidth (%)") mem_bw="$val" ;;
                    "GPU Frequency (MHz)") freq="$val" ;;
                esac
            fi
        done <<< "$stats"
        
        # Save result to temp file
        echo "$temp|$power|$util|$mem_used|$mem_bw|$freq" > "$TMPDIR/stats.$id"
    ) &
done

# Wait for all background jobs
wait

# --- Step 3: Output ---
printf "+%s+\n" "-------------------------------------------------------------------------------------------------"
tool_ver=$(xpu-smi version 2>/dev/null | awk '/XPU-SMI/ {print $2; exit}' || echo "N/A")
drv_ver=$(xpu-smi version 2>/dev/null | awk '/Driver/ {print $2; exit}' || echo "N/A")
printf "| %-30s %-25s %58s |\n" "XPU-SMI Version: $tool_ver" "Driver Version: $drv_ver" "$(date '+%Y/%m/%d %H:%M:%S')"
printf "|%s|\n" "================================================================================================="
printf "| %-3s %-22s %5s %6s %6s %18s %8s %10s |\n" "ID" "Name" "Temp" "Power" "Util%" "Mem_Used/Total" "Mem_BW%" "Freq(MHz)"
printf "|%s|\n" "================================================================================================="

for id in "${device_ids[@]}"; do
    if [ -f "$TMPDIR/stats.$id" ]; then
        IFS='|' read -r temp power util mem_used mem_bw freq < "$TMPDIR/stats.$id"
    else
        temp=power=util=mem_used=mem_bw=freq="--"
    fi

    mem_str="${mem_used}MiB / ${MEM_TOTAL}MiB"
    [ "$mem_used" = "N/A" ] || [ "$mem_used" = "--" ] && mem_str="-- / --"

    printf "| %-3s %-22s %5s %6s %6s %18s %8s %10s |\n" \
        "$id" "${device_names[$id]}" "${temp}C" "${power}W" "${util}%" "$mem_str" "${mem_bw}%" "$freq"
done

printf "+%s+\n" "-------------------------------------------------------------------------------------------------"

# Cleanup
rm -rf "$TMPDIR"