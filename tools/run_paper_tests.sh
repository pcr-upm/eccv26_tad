#!/usr/bin/env bash
#
# Run every reported SVTAD test in order and print one summary table.
#
# The runs and expected numbers come from report_svtad.xlsx (sheets THUMOS /
# ActivityNet / ATTACH). Each entry below is the excel command plus the overrides
# needed to make it correct against *this* checkout -- see NOTES at the bottom.
#
# Usage
#   bash tools/run_paper_tests.sh                 # run all, in order
#   bash tools/run_paper_tests.sh --list          # show the manifest and exit
#   bash tools/run_paper_tests.sh --dry-run       # print commands, run nothing
#   bash tools/run_paper_tests.sh --preflight     # validate configs/paths/weights only
#   bash tools/run_paper_tests.sh --only thumos_internvideo_l --only attach_nokp
#   bash tools/run_paper_tests.sh --skip anet_internvideo_l
#   bash tools/run_paper_tests.sh --stop-on-fail  # stop at the first failure
#                                                 # (default: run every selected test)
#
# Env overrides
#   WEIGHTS_DIR        where the result .pth files live
#   NPROC              GPUs per run (default: detected GPU count)
#   RESULTS_DIR        where logs + summary.tsv go (default: exps/paper_tests/<stamp>)
#   THUMOS_DATA_ROOT   THUMOS mp4 dir      (the committed config points at a server path)
#   ANET_CLS_PATH      CUHK classifier json for ActivityNet
#
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
REPO_ROOT="$PWD"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=16
WEIGHTS_DIR="${WEIGHTS_DIR:-test_weights}"
THUMOS_DATA_ROOT="${THUMOS_DATA_ROOT:-/datasets/THUMOS14/raw_data/video}"
ANET_CLS_PATH="${ANET_CLS_PATH:-/datasets/activitynet-1.3/classifiers/cuhk_val_simp_7.json}"
# Both configs point at debug-subset annotations (14 THUMOS videos / 9 ATTACH videos).
# These are the files whose GT counts match the reported logs: 3325 and 15578.
THUMOS_ANN="${THUMOS_ANN:-/datasets/THUMOS14/annotations/thumos_14_anno.json}"
ATTACH_ANN="${ATTACH_ANN:-/datasets/ATTACH/141.24.24.111:50021/attach_person_split_ann.json}"
RESULTS_DIR="${RESULTS_DIR:-exps/paper_tests/$(date +%Y%m%d_%H%M%S)}"
# Set ALLOW_PARTIAL_DATA=1 to smoke-test on an incomplete dataset copy.
PARTIAL_FLAG=""
[[ -n "${ALLOW_PARTIAL_DATA:-}" ]] && PARTIAL_FLAG="--allow-partial-data"

if [[ -z "${NPROC:-}" ]]; then
    NPROC="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)"
    [[ "$NPROC" -lt 1 ]] && NPROC=1
fi

# ---------------------------------------------------------------------------
# Manifest: NAME | CONFIG | CHECKPOINT | CFG_OPTIONS | EXTRA FLAGS
# Order matches the report: THUMOS, then ActivityNet, then ATTACH. 
# ---------------------------------------------------------------------------
RUNS=(
"thumos_vitb|configs/vitsparse/thumos/e2e_thumos_videomae_b_768x1_160_sparse_adapter.py|thumos_vitb.pth|model.backbone.backbone.n_landmarks=0|--data-root ${THUMOS_DATA_ROOT} --ann-file ${THUMOS_ANN}"
"thumos_vitl|configs/vitsparse/thumos/e2e_thumos_videomae_l_768x1_160_sparse_adapter.py|thumos_vitl.pth|-|--data-root ${THUMOS_DATA_ROOT} --ann-file ${THUMOS_ANN}"
"thumos_internvideo_l|configs/vitsparse/thumos/e2e_thumos_internvideo_l_768x1_224_sparse_adapter.py|thumos_internvid.pth|-|--data-root ${THUMOS_DATA_ROOT} --ann-file ${THUMOS_ANN}"
"anet_vitb|configs/vitsparse/anet/e2e_anet_videomae_b_192x4_160_sparse_adapter.py|anet_vitb.pth|-|--external-cls-path ${ANET_CLS_PATH}"
"anet_vitl|configs/vitsparse/anet/e2e_anet_videomae_l_192x4_160_sparse_adapter.py|anet_vitl.pth|-|--external-cls-path ${ANET_CLS_PATH}"
"anet_internvideo_l|configs/vitsparse/anet/e2e_anet_internvideo_l_192x4_224_sparse_adapter.py|anet_internvid.pth|-|-"
"attach_nokp|configs/vitsparse/attach/e2e_attach_videomae_b_768x1_160_sparse_adapter.py|attach_nokpb.pth|model.backbone.backbone.n_landmarks=0|--ann-file ${ATTACH_ANN}"
"attach_nokp_no_crossattn|configs/vitsparse/attach/e2e_attach_videomae_b_768x1_160_sparse_adapter.py|attach_nocrossattn.pth|model.backbone.backbone.n_landmarks=0 model.backbone.backbone.adapter_use_attn=0|--ann-file ${ATTACH_ANN}"
"attach_kp|configs/vitsparse/attach/e2e_attach_videomae_b_768x1_160_sparse_adapter.py|attach_kpb.pth|model.backbone.backbone.n_landmarks=32|--ann-file ${ATTACH_ANN}"
)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
# A failing run never blocks the others: the whole point of the suite is one
# summary table covering every entry. --stop-on-fail restores the old behaviour.
DRY_RUN=0; PREFLIGHT_ONLY=0; KEEP_GOING=1; ONLY=(); SKIP=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --list)       LIST_ONLY=1; shift ;;
        --dry-run)    DRY_RUN=1; shift ;;
        --preflight)  PREFLIGHT_ONLY=1; shift ;;
        --keep-going)   KEEP_GOING=1; shift ;;   # now the default; kept for compat
        --stop-on-fail) KEEP_GOING=0; shift ;;
        --only)       ONLY+=("$2"); shift 2 ;;
        --skip)       SKIP+=("$2"); shift 2 ;;
        --nproc)      NPROC="$2"; shift 2 ;;
        -h|--help)    sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

selected() {
    local name="$1" s
    for s in "${SKIP[@]:-}"; do [[ "$s" == "$name" ]] && return 1; done
    [[ ${#ONLY[@]} -eq 0 ]] && return 0
    for s in "${ONLY[@]}"; do [[ "$s" == "$name" ]] && return 0; done
    return 1
}

if [[ -n "${LIST_ONLY:-}" ]]; then
    printf '%-20s %s\n' NAME CHECKPOINT
    for entry in "${RUNS[@]}"; do
        IFS='|' read -r name cfg ckpt _ _ <<< "$entry"
        printf '%-20s %s\n' "$name" "$ckpt"
    done
    exit 0
fi

mkdir -p "$RESULTS_DIR"
SUMMARY="$RESULTS_DIR/summary.tsv"
printf 'name\tstatus\tavg_mAP\tepoch\tlog\n' > "$SUMMARY"

echo "=========================================================================="
echo " SVTAD paper test suite"
echo "   repo         : $REPO_ROOT"
echo "   weights      : $WEIGHTS_DIR"
echo "   GPUs per run : $NPROC"
echo "   results      : $RESULTS_DIR"
echo "=========================================================================="

FAILED=0
for entry in "${RUNS[@]}"; do
    IFS='|' read -r name config ckpt_file cfg_opts extra_flags <<< "$entry"
    selected "$name" || { echo "-- skipping $name"; continue; }

    checkpoint="$WEIGHTS_DIR/$ckpt_file"
    log="$RESULTS_DIR/$name.log"
    [[ "$cfg_opts"    == "-" ]] && cfg_opts=""
    [[ "$extra_flags" == "-" ]] && extra_flags=""

    echo
    echo "--------------------------------------------------------------------------"
    echo ">> $name"
    echo "--------------------------------------------------------------------------"

    # ---- preflight: catch config/checkpoint mismatches before burning GPU time
    pf_log="$RESULTS_DIR/$name.preflight.log"
    # shellcheck disable=SC2086
    python tools/paper_test_preflight.py \
        --config "$config" --checkpoint "$checkpoint" \
        ${cfg_opts:+--cfg-options $cfg_opts} \
        $extra_flags $PARTIAL_FLAG 2>&1 | tee "$pf_log"
    pf_status="${PIPESTATUS[0]}"

    if [[ "$pf_status" -ne 0 ]]; then
        echo "!! $name FAILED PREFLIGHT -- not run (see $pf_log)"
        printf '%s\tpreflight_fail\t-\t-\t%s\n' "$name" "$pf_log" >> "$SUMMARY"
        FAILED=$((FAILED + 1))
        [[ "$KEEP_GOING" -eq 1 ]] && continue || break
    fi

    # preflight may ask for extra overrides (e.g. a pretrain file absent on this box)
    pf_extra="$(sed -n 's/^PREFLIGHT_RESULT ok EXTRA_CFG_OPTS=//p' "$pf_log")"
    [[ -n "$pf_extra" ]] && cfg_opts="$cfg_opts $pf_extra"

    if [[ "$PREFLIGHT_ONLY" -eq 1 ]]; then
        printf '%s\tpreflight_ok\t-\t-\t%s\n' "$name" "$pf_log" >> "$SUMMARY"
        continue
    fi

    # ---- the actual test command
    cmd=(torchrun --nnodes=1 --nproc_per_node="$NPROC"
         --rdzv_backend=c10d --rdzv_endpoint=localhost:0
         tools/test.py "$config" --checkpoint "$checkpoint")
    [[ -n "$extra_flags" ]] && read -r -a _extra <<< "$extra_flags" && cmd+=("${_extra[@]}")
    if [[ -n "${cfg_opts// /}" ]]; then
        read -r -a _opts <<< "$cfg_opts"
        cmd+=(--cfg-options "${_opts[@]}")
    fi

    printf '%q ' "${cmd[@]}"; echo
    if [[ "$DRY_RUN" -eq 1 ]]; then
        printf '%s\tdry_run\t-\t-\t-\n' "$name" >> "$SUMMARY"
        continue
    fi

    start=$SECONDS
    "${cmd[@]}" 2>&1 | tee "$log"
    run_status="${PIPESTATUS[0]}"
    elapsed=$((SECONDS - start))

    # ---- parse the result out of the log
    mAP="$(grep -oE 'Average-mAP: +[0-9.]+' "$log" | tail -1 | grep -oE '[0-9.]+$')"
    epoch="$(grep -oE 'Checkpoint is epoch [0-9]+' "$log" | tail -1 | grep -oE '[0-9]+$')"

    if [[ "$run_status" -ne 0 ]]; then
        status=crashed; FAILED=$((FAILED + 1))
    elif [[ -z "$mAP" ]]; then
        status=no_result; FAILED=$((FAILED + 1))
    else
        status=ok
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$status" "${mAP:--}" "${epoch:--}" "$log" >> "$SUMMARY"
    echo ">> $name: $status  avg mAP=${mAP:--}  in ${elapsed}s"

    if [[ "$status" != "ok" && "$KEEP_GOING" -ne 1 ]]; then
        echo "!! stopping (--stop-on-fail)"
        break
    fi
done

echo
echo "=========================================================================="
echo " SUMMARY   ($RESULTS_DIR)"
echo "=========================================================================="
awk -F'\t' '{printf "%-20s %-16s %-9s %-6s %s\n", $1, $2, $3, $4, $5}' "$SUMMARY"
echo
[[ "$FAILED" -gt 0 ]] && echo "$FAILED run(s) did not produce a result." && exit 1
echo "All selected runs produced a result."
exit 0

