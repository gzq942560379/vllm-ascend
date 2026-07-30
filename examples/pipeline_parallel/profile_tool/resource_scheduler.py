#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Ascend die discovery, topology-aware selection, and cooperative leases."""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

try:
    import fcntl
except ImportError:  # pragma: no cover - leases run on the Linux server.
    fcntl = None  # type: ignore[assignment]


@dataclass(frozen=True)
class DeviceInfo:
    logic_id: int
    card_id: int
    chip_id: int
    aicore: int = 0
    hbm_pct: int = 0

    @classmethod
    def from_mapping(cls, value: dict[str, object]) -> "DeviceInfo":
        return cls(
            logic_id=int(value["logic_id"]),
            card_id=int(value["card_id"]),
            chip_id=int(value["chip_id"]),
            aicore=int(value.get("aicore", 0)),
            hbm_pct=int(value.get("hbm_pct", 0)),
        )


def select_devices(
    devices: Sequence[DeviceInfo],
    count: int,
    topology_aware: bool = True,
) -> list[DeviceInfo]:
    if count < 1:
        raise ValueError("count must be positive")
    unique = {device.logic_id: device for device in devices}
    if len(unique) < count:
        return []
    ordered = sorted(unique.values(), key=lambda item: item.logic_id)
    if not topology_aware:
        return ordered[:count]
    by_card: dict[int, list[DeviceInfo]] = {}
    for device in ordered:
        by_card.setdefault(device.card_id, []).append(device)
    selected: list[DeviceInfo] = []
    # Full cards first keeps the two dies on a 910B card together.
    for card_id in sorted(
        by_card, key=lambda key: (-len(by_card[key]), key)
    ):
        for device in sorted(
            by_card[card_id], key=lambda item: item.chip_id
        ):
            if len(selected) == count:
                break
            selected.append(device)
        if len(selected) == count:
            break
    return sorted(selected, key=lambda item: item.logic_id)


def discover_idle_devices(
    selector_script: str | Path,
    *,
    max_aicore: int,
    max_hbm_pct: int,
) -> list[DeviceInfo]:
    command = [
        "bash",
        str(selector_script),
        "--list",
        "--quiet",
        str(max_aicore),
        str(max_hbm_pct),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode == 5:
        return []
    if completed.returncode:
        raise RuntimeError(
            "idle NPU discovery failed: "
            + (completed.stderr or completed.stdout).strip()
        )
    devices: list[DeviceInfo] = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            devices.append(DeviceInfo.from_mapping(json.loads(line)))
    if devices:
        return devices
    # Compatibility with older selector output: IDS=0 1 2.
    for line in completed.stdout.splitlines():
        if line.startswith("IDS="):
            return [
                DeviceInfo(logic_id=int(value), card_id=int(value) // 2,
                           chip_id=int(value) % 2)
                for value in line[4:].split()
            ]
        values = line.replace(",", " ").split()
        if values and all(value.isdigit() for value in values):
            return [
                DeviceInfo(
                    logic_id=int(value),
                    card_id=int(value) // 2,
                    chip_id=int(value) % 2,
                )
                for value in values
            ]
    return []


class DeviceLease:
    """Cooperative per-die flock lease; it never touches foreign processes."""

    def __init__(self, devices: Sequence[DeviceInfo], lock_dir: str | Path):
        self.devices = tuple(devices)
        self.lock_dir = Path(lock_dir)
        self._files: list[object] = []

    def acquire(self) -> bool:
        if fcntl is None:
            raise RuntimeError("DeviceLease requires a Linux host with fcntl")
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        try:
            for device in self.devices:
                handle = (self.lock_dir / f"die-{device.logic_id}.lock").open(
                    "a+", encoding="utf-8"
                )
                self._files.append(handle)
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                handle.seek(0)
                handle.truncate()
                handle.write(
                    json.dumps({"pid": os.getpid(), "time": time.time()})
                )
                handle.flush()
            return True
        except (BlockingIOError, OSError):
            self.release()
            return False

    def release(self) -> None:
        if fcntl is None:
            self._files.clear()
            return
        for handle in reversed(self._files):
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()
            except OSError:
                pass
        self._files.clear()

    def __enter__(self) -> "DeviceLease":
        if not self.acquire():
            raise RuntimeError("one or more NPU dies are already leased")
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


def wait_for_lease(
    *,
    count: int,
    discover: Callable[[], Sequence[DeviceInfo]],
    lock_dir: str | Path,
    poll_seconds: int,
    max_wait_seconds: int,
    heartbeat: Callable[[str], None] | None = None,
) -> DeviceLease | None:
    deadline = time.monotonic() + max_wait_seconds
    while time.monotonic() < deadline:
        try:
            discovered = discover()
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            if heartbeat:
                heartbeat(f"WAIT_NPU: discovery error: {exc}")
            time.sleep(poll_seconds)
            continue
        candidates = select_devices(discovered, count, topology_aware=True)
        if len(candidates) == count:
            lease = DeviceLease(candidates, lock_dir)
            if lease.acquire():
                # Recheck after the cooperative lock to reduce the npu-smi race.
                try:
                    visible = {device.logic_id for device in discover()}
                except (OSError, RuntimeError, subprocess.SubprocessError):
                    visible = set()
                if all(device.logic_id in visible for device in candidates):
                    return lease
                lease.release()
        if heartbeat:
            heartbeat(f"WAIT_NPU: need {count} idle dies")
        time.sleep(poll_seconds)
    return None
