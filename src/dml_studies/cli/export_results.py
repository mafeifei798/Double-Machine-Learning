#!/usr/bin/env python3
"""Export selected saved jobs without rewriting the historical source records."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import zipfile

REPO = Path(__file__).resolve().parents[3]
ARCHIVES = ("completion_results_v7.tgz", "server_source_v7.tgz", "v6_selected_records.zip")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read_archives(root: Path):
    """Validate original archive/member hashes without extracting or running code."""
    manifest = json.loads((root / "completed_simulation_manifest.json").read_text())
    archives, members = {}, {}
    for name in ARCHIVES:
        data = (root / name).read_bytes()
        expected = manifest["archives"][name]
        assert len(data) == expected["bytes"] and digest(data) == expected["sha256"], name
        archives[name] = {"bytes": len(data), "sha256": digest(data)}
        if name.endswith(".tgz"):
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
                entries = [x for x in archive.getmembers() if x.isfile()]
                records = {x.name: archive.extractfile(x).read() for x in entries}
        else:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                entries = [x for x in archive.infolist() if not x.is_dir()]
                records = {x.filename: archive.read(x) for x in entries}
        assert len(records) == len(entries), "Duplicate archive member"
        members[name] = records
    for label, name in (("bridge", ARCHIVES[0]), ("v6_selected", ARCHIVES[2])):
        expected = manifest["record_files"][label]
        assert set(expected) == set(members[name])
        for member, data in members[name].items():
            assert len(data) == expected[member]["bytes"] and digest(data) == expected[member]["sha256"]
    return archives, members


def source_jobs(root: Path):
    archives, members = read_archives(root)
    exports, sources, durations = [], {}, []
    for label, archive_name, freeze_name, ledger_name, prefix in (
        ("finite_sample", ARCHIVES[0], "BRIDGE_FROZEN.json", "output/legacy_bridge_v7/RESULT_LEDGER.json", "output/legacy_bridge_v7/figure"),
        ("sensitivity", ARCHIVES[2], "output/formal/FROZEN_MANIFEST.json", "output/formal/RESULT_LEDGER.json", "output/formal/figure"),
    ):
        content = members[archive_name]
        frozen = json.loads(content[freeze_name])
        identity = frozen["identity"]
        assert digest(json.dumps(identity, sort_keys=True).encode()) == frozen["identity_sha256"]
        ledger = json.loads(content[ledger_name])
        protocol = dict(identity["protocol"])
        protocol.pop("authorized_execution_root", None)
        sources[label] = {
            "archive": archive_name,
            "original_identity_sha256": frozen["identity_sha256"],
            "original_source_hashes": identity["source_hashes"],
            "environment": identity["environment"],
            "scientific_protocol": protocol,
            "base_protocol": identity["base_protocol"],
        }
        for member in sorted(content):
            data = content[member]
            if member.startswith(prefix) and member.endswith(".json"):
                assert digest(data) == ledger[Path(member).name]
                receipt = json.loads(data)
                assert receipt["identity_sha256"] == frozen["identity_sha256"]
                exports.append({"source": label, "source_member": member,
                    "original_receipt_sha256": digest(data),
                    "canonical_receipt_sha256": digest(canonical(receipt)), "receipt": receipt})
            elif member.endswith(".json"):
                def collect(value, path=()):
                    if isinstance(value, dict):
                        for key, item in value.items():
                            if key in ("seconds", "elapsed_seconds", "duration_seconds") and isinstance(item, (int, float)):
                                durations.append({"source": label, "source_member": member,
                                    "field": ".".join((*path, key)), "seconds": item,
                                    "original_record_sha256": digest(data)})
                            elif isinstance(item, (dict, list)):
                                collect(item, (*path, key))
                    elif isinstance(value, list):
                        for i, item in enumerate(value):
                            collect(item, (*path, str(i)))
                collect(json.loads(data))
    source_content = members[ARCHIVES[1]]
    for name, expected in sources["finite_sample"]["original_source_hashes"].items():
        assert digest(source_content[name]) == expected, name
    assert sum(x["source"] == "finite_sample" for x in exports) == 600
    assert sum(x["source"] == "sensitivity" for x in exports) == 400
    assert sum(len(x["receipt"]["rows"]) for x in exports) == 10800
    return exports, archives, sources, durations


def write_once(path: Path, data: bytes):
    if path.exists():
        assert path.read_bytes() == data, f"Refusing to overwrite different results: {path}"
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True,
                        help="Original archived evidence directory, recovered separately from repository history.")
    parser.add_argument("--output-directory", type=Path, default=REPO / "results")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--primary-receipt-hashes", type=Path,
                        help="Original primary-study receipt hash map; required for the initial export.")
    args = parser.parse_args()
    assert len(args.source_commit) == len(args.source_tree) == 40
    jobs, archives, sources, durations = source_jobs(args.input_directory)
    output = args.output_directory
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0, compresslevel=9) as stream:
        for job in jobs:
            stream.write(canonical(job) + b"\n")
    write_once(output / "simulation_records.jsonl.gz", buffer.getvalue())
    prior = output / "data_manifest.json"
    if args.primary_receipt_hashes:
        main_hashes = json.loads(args.primary_receipt_hashes.read_text())
    else:
        if not prior.is_file():
            parser.error("Initial export requires --primary-receipt-hashes")
        main_hashes = json.loads(prior.read_text())["primary_source_receipt_hashes"]
    assert len(main_hashes) == 800
    result_files = {"simulation_records.jsonl.gz": digest(buffer.getvalue())}
    for prefix in ("primary", "finite_sample", "sensitivity", "nuisance_comparison"):
        for path in sorted((output / prefix).rglob("*")):
            if path.is_file():
                result_files[str(path.relative_to(output))] = digest(path.read_bytes())
    provenance = {
        "schema_version": 1,
        "record_type": "derived_saved_result_export",
        "source_repository": "https://github.com/mafeifei798/Double-Machine-Learning",
        "original_source_commit": args.source_commit,
        "original_source_tree": args.source_tree,
        "original_archives": archives,
        "sources": sources,
        "control_durations": durations,
        "primary_source_receipt_hashes": main_hashes,
        "scope": {"exported_jobs": 1000, "exported_estimates": 10800,
                  "finite_sample_comparison_estimates": 7200, "additional_moderate_estimates": 3600,
                  "primary_csv_estimates": 6000,
                  "full_historical_sensitivity_raw_jobs_bundled": False},
        "result_files": result_files,
        "interpretation": "Derived exports retain saved scientific payloads, warnings, seeds and elapsed times. Original administrative records and frozen sources remain in repository history; this export is not an original execution receipt or a new experiment.",
    }
    write_once(prior, json.dumps(provenance, indent=2, allow_nan=False).encode() + b"\n")
    print(json.dumps({"status": "EXPORTED_VERIFIED_SAVED_RESULTS", "jobs": len(jobs),
        "estimates": 10800, "result_files": len(result_files), "model_fits": 0}))


if __name__ == "__main__":
    main()
