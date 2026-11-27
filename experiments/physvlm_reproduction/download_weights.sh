#!/usr/bin/env bash
set -euo pipefail

dest="${1:?Pass a model directory}"
root="${HF_ENDPOINT:-https://huggingface.co}/JettZhou/PhysVLM-Qwen2.5-3B/resolve/411280628b973db0c1e58d3e1a22c0dda90729d5"
mkdir -p "$dest"
cd "$dest"

fetch() {
    curl -fsSL --retry 3 --connect-timeout 18 --max-time 600 --speed-time 35 --speed-limit 8192 "$@"
}

complete_range() {
    local name="$1" start="$2" end="$3" size="$4" part="$5"
    local expected=$((end - start + 1)) have first last received misses=0
    while true; do
        have=0
        if [[ -f "$part" ]]; then
            have="$(stat -c%s "$part")"
        fi
        if (( have == expected )); then
            break
        fi
        (( have < expected && misses < 8 )) || return 1
        first=$((start + have))
        last=$((first + 16777215))
        (( last <= end )) || last="$end"
        : > "$part.headers"
        : > "$part.next"
        curl -fsSL --connect-timeout 18 --max-time 240 --speed-time 45 --speed-limit 4096 \
            -D "$part.headers" -r "$first-$last" "$root/$name?paired_range=$first-$last" -o "$part.next" || true
        received="$(stat -c%s "$part.next")"
        if (( received > 0 && received <= last - first + 1 )) &&
            tr -d '\r' < "$part.headers" | grep -Fxiq "content-range: bytes $first-$last/$size"; then
            cat "$part.next" >> "$part"
            misses=0
        else
            misses=$((misses + 1))
            sleep 2
        fi
    done
    printf '%s %s-%s complete\n' "$name" "$start" "$end"
    rm -f -- "$part.headers" "$part.next"
}

download_shard() (
    name="$1" size="$2" sha="$3"
    if [[ -s "$name" ]] && echo "$sha  $name" | sha256sum -c -; then
        return
    fi
    chunk=268435456
    pids=() parts=()
    for ((start=0; start<size; start+=chunk)); do
        end=$((start + chunk - 1))
        ((end < size)) || end=$((size - 1))
        part="$name.part.$start"
        parts+=("$part")
        complete_range "$name" "$start" "$end" "$size" "$part" &
        pids+=("$!")
        if (( ${#pids[@]} >= 8 )); then
            wait "${pids[0]}"
            pids=("${pids[@]:1}")
        fi
    done
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
    cat "${parts[@]}" > "$name.verified"
    echo "$sha  $name.verified" | sha256sum -c -
    mv "$name.verified" "$name"
    rm -- "${parts[@]}"
)

download_shard model-00001-of-00002.safetensors 4957560304 4a12dcfafe82fea30e46a4d111a59b98fd2edf634e9309f9850d590aadffe521 &
big=$!
download_shard model-00002-of-00002.safetensors 3458769488 cb5375e18172d0258f09c56fb72a03de7b9ba8ddab313a8ea688acc680554eb1 &
small=$!
for name in config.json generation_config.json tokenizer_config.json special_tokens_map.json added_tokens.json merges.txt vocab.json model.safetensors.index.json; do
    if [[ ! -s "$name" ]]; then
        fetch "$root/$name" -o "$name.tmp"
        mv "$name.tmp" "$name"
    fi
done
wait "$big"
wait "$small"
