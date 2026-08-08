#!/usr/bin/env python3
"""Repeatable end-to-end benchmark for PixelHolo avatar streams.

The runner measures a complete ``/chat`` request as the browser receives it:

* request to first complete PHS1 media packet
* complete-stream time and server-reported inference time
* media packet cadence and rendered-audio duration
* predicted browser playback headroom using the production scheduler

It intentionally does not inspect a user's workspace.  Pass an explicit
workspace id and a benchmark-only profile.  Each run uses a fresh system
prompt, which prevents prior benchmark responses from changing the LLM's
conversation history and timing later repetitions.

Examples:

  python tools/benchmark_avatar_delivery.py \
    --api-base https://pixelholo.com/api \
    --workspace-id <workspace-id> --profile test1 --repeats 3

  # From the VM, compare the FastAPI path without Cloudflare/proxy overhead.
  python tools/benchmark_avatar_delivery.py \
    --api-base http://127.0.0.1:8001 \
    --workspace-id <workspace-id> --profile test1 --repeats 3
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import struct
import sys
import time
import uuid
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

STREAM_MEDIA_TYPE = "application/vnd.pixelholo.stream-v1"
STREAM_MAGIC = b"PHS1"
DEFAULT_PROMPT = "Briefly greet the user and mention that their avatar is ready."
DEFAULT_SYSTEM_PROMPT = (
    "Reply in exactly one natural sentence of 12 to 18 words. "
    "Do not use a greeting label, bullet, or markdown."
)


@dataclass(frozen=True)
class Case:
    name: str
    llm_mode: str
    prompt: str
    description: str


CASES = (
    Case("llama_static", "legacy_fast", DEFAULT_PROMPT, "Llama 3.1 8B Instant static route"),
    Case("gpt_live", "live_search", DEFAULT_PROMPT, "GPT-4o mini live route"),
    Case("gemini_live", "gemini_search", DEFAULT_PROMPT, "Gemini 2.5 Flash Lite live route"),
    Case("auto_static", "auto", DEFAULT_PROMPT, "Auto mode with a static prompt"),
    Case(
        "auto_current",
        "auto",
        "What is the current time in Toronto? Answer in one concise sentence.",
        "Auto mode with a current-information prompt",
    ),
)


def _now_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    """Return linear-interpolated percentiles, including p50 and p95.

    This behaves predictably for the intentionally small repeated benchmark
    samples.  The report records the count beside each percentile.
    """

    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return round(ordered[0], 1)
    position = (len(ordered) - 1) * (percentile / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 1)
    weight = position - lower
    return round(ordered[lower] * (1.0 - weight) + ordered[upper] * weight, 1)


def _read_stream(
    response: Any,
    started: float,
    *,
    browser_start_delay_sec: float,
    browser_chunk_lead_sec: float,
) -> dict[str, Any]:
    """Parse PHS1 packets and model production Web Audio scheduling.

    The browser schedules the first buffer after a fixed cushion from the
    moment *that media arrives*.  Every later packet continues from the prior
    buffer unless it arrives after the queued audio has run out.  This is the
    actual distinction between a long server gap and a visible audio/video
    stall.
    """

    pending = b""
    first_media_ms: float | None = None
    packet_arrivals_ms: list[float] = []
    packet_durations_ms: list[float] = []
    packet_frames: list[int] = []
    events: list[dict[str, Any]] = []
    wire_bytes = 0
    audio_bytes = 0
    frame_bytes = 0
    playback_end_ms: float | None = None
    headroom_before_packet_ms: list[float] = []
    underrun_lateness_ms: list[float] = []
    server_error: str | None = None

    while raw := response.read(16 * 1024):
        if not raw:
            continue
        wire_bytes += len(raw)
        pending += raw
        while len(pending) >= 12:
            if pending[:4] != STREAM_MAGIC:
                raise RuntimeError(f"invalid PHS1 stream header: {pending[:12]!r}")
            metadata_length, payload_length = struct.unpack(">II", pending[4:12])
            total_length = 12 + metadata_length + payload_length
            if len(pending) < total_length:
                break
            try:
                metadata = json.loads(pending[12 : 12 + metadata_length])
            except json.JSONDecodeError as exc:
                raise RuntimeError("invalid PHS1 metadata") from exc
            payload = pending[12 + metadata_length : total_length]
            pending = pending[total_length:]
            if len(payload) != payload_length:
                raise RuntimeError("PHS1 payload length mismatch")
            events.append(metadata)
            event = metadata.get("event")
            if event == "error":
                server_error = str(metadata.get("detail") or "backend stream error")
                continue
            if event != "chunk":
                continue

            received_ms = _now_ms(started)
            duration_ms = max(0.0, float(metadata.get("duration_sec") or 0.0) * 1000.0)
            audio_length = max(0, int(metadata.get("audio_bytes_len") or 0))
            frame_lengths = metadata.get("frame_lengths") or []
            frame_lengths = [max(0, int(length)) for length in frame_lengths]
            if audio_length + sum(frame_lengths) != payload_length:
                raise RuntimeError("PHS1 packet media lengths do not match payload")

            if first_media_ms is None:
                first_media_ms = received_ms
                playback_start_ms = received_ms + browser_start_delay_sec * 1000.0
                playback_end_ms = playback_start_ms + duration_ms
                headroom_before_packet_ms.append(browser_start_delay_sec * 1000.0)
            else:
                assert playback_end_ms is not None
                scheduled_earliest_ms = received_ms + browser_chunk_lead_sec * 1000.0
                headroom = playback_end_ms - received_ms
                headroom_before_packet_ms.append(headroom)
                if scheduled_earliest_ms > playback_end_ms:
                    underrun_lateness_ms.append(scheduled_earliest_ms - playback_end_ms)
                playback_end_ms = max(playback_end_ms, scheduled_earliest_ms) + duration_ms

            packet_arrivals_ms.append(round(received_ms, 1))
            packet_durations_ms.append(round(duration_ms, 1))
            packet_frames.append(len(frame_lengths))
            audio_bytes += audio_length
            frame_bytes += sum(frame_lengths)

    if pending:
        raise RuntimeError(f"stream ended with {len(pending)} undecoded trailing bytes")

    done = next((event for event in reversed(events) if event.get("event") == "done"), None)
    completion_ms = _now_ms(started)
    interarrival_ms = [
        round(packet_arrivals_ms[index] - packet_arrivals_ms[index - 1], 1)
        for index in range(1, len(packet_arrivals_ms))
    ]
    return {
        "first_media_ms": round(first_media_ms, 1) if first_media_ms is not None else None,
        "completion_ms": round(completion_ms, 1),
        "server_inference_ms": done.get("inference_ms") if done else None,
        "chunks": len(packet_arrivals_ms),
        "frames": sum(packet_frames),
        "audio_duration_ms": round(sum(packet_durations_ms), 1),
        "wire_bytes": wire_bytes,
        "audio_bytes": audio_bytes,
        "frame_bytes": frame_bytes,
        "packet_arrivals_ms": packet_arrivals_ms,
        "packet_durations_ms": packet_durations_ms,
        "packet_frames": packet_frames,
        "interarrival_ms": interarrival_ms,
        "interarrival_p50_ms": _percentile(interarrival_ms, 50),
        "interarrival_p95_ms": _percentile(interarrival_ms, 95),
        "interarrival_max_ms": round(max(interarrival_ms), 1) if interarrival_ms else 0.0,
        "min_browser_headroom_ms": round(min(headroom_before_packet_ms), 1)
        if headroom_before_packet_ms
        else None,
        "browser_underruns": len(underrun_lateness_ms),
        "max_browser_underrun_ms": round(max(underrun_lateness_ms), 1)
        if underrun_lateness_ms
        else 0.0,
        "server_error": server_error,
    }


def _open_request(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"User-Agent": "PixelHolo-Benchmark/1.0", **headers},
        method="POST",
    )
    return urllib.request.urlopen(request, timeout=timeout)


def _post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    with _open_request(url, payload, headers, timeout) as response:
        body = response.read()
    elapsed_ms = _now_ms(started)
    return json.loads(body), round(elapsed_ms, 1)


def _warmup(
    *,
    api_base: str,
    profile: str,
    workspace_id: str,
    llm_mode: str,
    timeout: float,
) -> dict[str, Any]:
    headers = {
        "Content-Type": "application/json",
        "X-PixelHolo-Workspace": workspace_id,
    }
    data, wall_ms = _post_json(
        f"{api_base.rstrip('/')}/warmup",
        {
            "profile": profile,
            "profile_type": "avatar",
            "tts_backend": "chatterbox",
            "lipsync_backend": "musetalk",
            "avatar_max_frame_edge": 768,
            "musetalk_preset": "realistic",
            "include_llm": True,
            "llm_mode": llm_mode,
        },
        headers,
        timeout,
    )
    return {"wall_ms": wall_ms, **data}


def _run_case(
    *,
    api_base: str,
    profile: str,
    workspace_id: str,
    case: Case,
    run_number: int,
    timeout: float,
    browser_start_delay_sec: float,
    browser_chunk_lead_sec: float,
    endpoint: str,
) -> dict[str, Any]:
    headers = {
        "Content-Type": "application/json",
        "Accept": STREAM_MEDIA_TYPE,
        "X-PixelHolo-Transport": "binary",
        "X-PixelHolo-Workspace": workspace_id,
        "X-PixelHolo-Client": "benchmark",
    }
    # Changing the prompt forces a clean isolated conversation state without
    # touching the profile's persisted user-facing persona.
    system_prompt = f"{DEFAULT_SYSTEM_PROMPT} Benchmark isolation token: {case.name}-{run_number}-{uuid.uuid4().hex[:8]}."
    # /speak bypasses the selected assistant and therefore isolates the
    # Chatterbox + MuseTalk streaming path under a fixed long response.  This
    # catches cadence problems that short assistant replies can hide.
    text = case.prompt if endpoint == "chat" else (
        "This is a fixed long streaming test for the avatar pipeline. "
        "Each sentence is spoken at a calm and even pace so the audio and face "
        "frames can be checked across several consecutive rendering windows. "
        "The system should keep the jaw, lips, and chin stable between words, "
        "and natural pauses should preserve a neutral resting expression. "
        "This test also verifies that the browser receives continuous media "
        "without losing its playback buffer or freezing on an old frame. "
        "The final sentence confirms that the same profile stays active for the "
        "entire request from the first packet through the final rendered frame."
    )
    payload = {
        "text": text,
        "speaker": profile,
        "avatar_profile": profile,
        "profile_type": "avatar",
        "tts_backend": "chatterbox",
        "lipsync_backend": "musetalk",
        "llm_mode": case.llm_mode,
        "system_prompt": system_prompt,
        "seed": 1234,
        "avatar_fps": 25,
        "avatar_max_frame_edge": 768,
        "musetalk_preset": "realistic",
        "musetalk_jpeg_quality": 90,
    }
    started = time.perf_counter()
    try:
        with _open_request(
            f"{api_base.rstrip('/')}/{endpoint}", payload, headers, timeout
        ) as response:
            content_type = response.headers.get("content-type", "")
            if STREAM_MEDIA_TYPE not in content_type:
                body = response.read(500).decode("utf-8", errors="replace")
                raise RuntimeError(f"expected PHS1 response, got {content_type!r}: {body}")
            metrics = _read_stream(
                response,
                started,
                browser_start_delay_sec=browser_start_delay_sec,
                browser_chunk_lead_sec=browser_chunk_lead_sec,
            )
        return {
            "case": case.name,
            "llm_mode": case.llm_mode,
            "description": case.description,
            "endpoint": endpoint,
            "run": run_number,
            "ok": metrics["server_error"] is None and metrics["chunks"] > 0,
            **metrics,
        }
    except Exception as exc:
        return {
            "case": case.name,
            "llm_mode": case.llm_mode,
            "description": case.description,
            "endpoint": endpoint,
            "run": run_number,
            "ok": False,
            "error": repr(exc),
            "elapsed_ms": round(_now_ms(started), 1),
        }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["case"])].append(row)
    result: dict[str, Any] = {}
    numeric_metrics = (
        "first_media_ms",
        "completion_ms",
        "server_inference_ms",
        "interarrival_p50_ms",
        "interarrival_p95_ms",
        "interarrival_max_ms",
        "min_browser_headroom_ms",
        "browser_underruns",
        "max_browser_underrun_ms",
        "audio_duration_ms",
    )
    for case_name, case_rows in grouped.items():
        successful = [row for row in case_rows if row.get("ok")]
        entry: dict[str, Any] = {
            "runs": len(case_rows),
            "successful_runs": len(successful),
            "failures": [row.get("error") or row.get("server_error") for row in case_rows if not row.get("ok")],
            "browser_underrun_runs": sum(1 for row in successful if row.get("browser_underruns", 0) > 0),
        }
        for metric in numeric_metrics:
            values = [float(row[metric]) for row in successful if row.get(metric) is not None]
            if values:
                entry[f"{metric}_p50"] = _percentile(values, 50)
                entry[f"{metric}_p95"] = _percentile(values, 95)
                entry[f"{metric}_max"] = round(max(values), 1)
        result[case_name] = entry
    return result


def _write_markdown(report: dict[str, Any], destination: Path) -> None:
    lines = [
        "# PixelHolo avatar delivery benchmark",
        "",
        f"Generated: {report['created_at']}",
        "",
        f"API base: `{report['api_base']}`  ",
        f"Endpoint: `/{report['endpoint']}`  ",
        f"Profile: `{report['profile']}`  ",
        f"Repetitions per case: {report['repeats']}  ",
        f"Browser schedule model: first packet + {report['browser_start_delay_sec']:.2f}s, then {report['browser_chunk_lead_sec']:.2f}s minimum lead.",
        "",
        "This models the current browser audio scheduler. A large packet-arrival gap is not a playback freeze if the queued-audio headroom remains positive.",
        "",
        "## Warm-up",
        "",
        "| Mode | Warm-up wall time | TTS ready | Lip sync ready | LLM ready |",
        "|---|---:|---|---|---|",
    ]
    for mode, data in report["warmups"].items():
        lines.append(
            f"| {mode} | {data.get('wall_ms', '—')} ms | {data.get('tts_ready', '—')} | "
            f"{data.get('lipsync_ready', '—')} | {data.get('llm_ready', '—')} |"
        )
    lines.extend(
        [
            "",
            "## Repeated request results",
            "",
            "p50 and p95 are the 50th and 95th percentiles computed by linear interpolation over the listed repetitions.",
            "",
            "| Case | Runs | First media p50 / p95 | Completion p50 / p95 | Packet gap p95 | Minimum headroom p50 | Browser underrun runs |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for case in CASES:
        item = report["summary"].get(case.name, {})
        def cell(key: str) -> str:
            value = item.get(key)
            return "—" if value is None else f"{value:.1f} ms"
        lines.append(
            f"| {case.name} | {item.get('successful_runs', 0)}/{item.get('runs', 0)} | "
            f"{cell('first_media_ms_p50')} / {cell('first_media_ms_p95')} | "
            f"{cell('completion_ms_p50')} / {cell('completion_ms_p95')} | "
            f"{cell('interarrival_p95_ms_p50')} | {cell('min_browser_headroom_ms_p50')} | "
            f"{item.get('browser_underrun_runs', 0)} |"
        )
    lines.extend(["", "## Per-run details", ""])
    for row in report["runs"]:
        if row.get("ok"):
            lines.append(
                f"- {row['case']} #{row['run']}: first media {row['first_media_ms']} ms, "
                f"completion {row['completion_ms']} ms, {row['chunks']} packets, "
                f"minimum headroom {row['min_browser_headroom_ms']} ms, "
                f"underruns {row['browser_underruns']}."
            )
        else:
            lines.append(f"- {row['case']} #{row['run']}: failed: {row.get('error') or row.get('server_error')}.")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--workspace-id", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--browser-start-delay-sec", type=float, default=1.35)
    parser.add_argument("--browser-chunk-lead-sec", type=float, default=0.08)
    parser.add_argument(
        "--endpoint",
        choices=("chat", "speak"),
        default="chat",
        help="chat measures the assistant-plus-avatar path. speak isolates TTS and lip sync.",
    )
    parser.add_argument("--cases", default=",".join(case.name for case in CASES))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    selected_names = {name.strip() for name in args.cases.split(",") if name.strip()}
    unknown = sorted(selected_names - {case.name for case in CASES})
    if unknown:
        parser.error(f"unknown case names: {', '.join(unknown)}")
    selected_cases = [case for case in CASES if case.name in selected_names]

    warmups: dict[str, dict[str, Any]] = {}
    for llm_mode in dict.fromkeys(case.llm_mode for case in selected_cases):
        try:
            warmups[llm_mode] = _warmup(
                api_base=args.api_base,
                profile=args.profile,
                workspace_id=args.workspace_id,
                llm_mode=llm_mode,
                timeout=args.timeout,
            )
            print(
                f"warmup mode={llm_mode} wall_ms={warmups[llm_mode].get('wall_ms')} "
                f"tts={warmups[llm_mode].get('tts_ready')} lipsync={warmups[llm_mode].get('lipsync_ready')} "
                f"llm={warmups[llm_mode].get('llm_ready')}",
                flush=True,
            )
        except Exception as exc:
            warmups[llm_mode] = {"error": repr(exc)}
            print(f"warmup mode={llm_mode} FAILED error={exc!r}", flush=True)

    rows: list[dict[str, Any]] = []
    for repetition in range(1, args.repeats + 1):
        for case in selected_cases:
            row = _run_case(
                api_base=args.api_base,
                profile=args.profile,
                workspace_id=args.workspace_id,
                case=case,
                run_number=repetition,
                timeout=args.timeout,
                browser_start_delay_sec=args.browser_start_delay_sec,
                browser_chunk_lead_sec=args.browser_chunk_lead_sec,
                endpoint=args.endpoint,
            )
            rows.append(row)
            if row.get("ok"):
                print(
                    f"case={case.name} run={repetition} first_media={row['first_media_ms']}ms "
                    f"completion={row['completion_ms']}ms packets={row['chunks']} "
                    f"gap_p95={row['interarrival_p95_ms']}ms "
                    f"headroom_min={row['min_browser_headroom_ms']}ms "
                    f"underruns={row['browser_underruns']}",
                    flush=True,
                )
            else:
                print(f"case={case.name} run={repetition} FAILED {row.get('error') or row.get('server_error')}", flush=True)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "api_base": args.api_base,
        "profile": args.profile,
        "repeats": args.repeats,
        "browser_start_delay_sec": args.browser_start_delay_sec,
        "browser_chunk_lead_sec": args.browser_chunk_lead_sec,
        "endpoint": args.endpoint,
        "warmups": warmups,
        "runs": rows,
        "summary": _summary(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(report, args.output.with_suffix(".md"))
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"wrote {args.output} and {args.output.with_suffix('.md')}", flush=True)
    return 0 if all(row.get("ok") for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
