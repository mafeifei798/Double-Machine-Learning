#!/usr/bin/env python3
"""Hash-check and unpack selected saved records. Stdlib only; never fits models."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import zipfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def relative_path(name):
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or '..' in path.parts or '\\' in name:
        raise ValueError(f'Unsafe archive path: {name!r}')
    return Path(*path.parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-root', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'evidence')
    args = parser.parse_args()
    root = args.evidence_root.resolve()
    manifest = json.loads((root / 'completed_simulation_manifest.json').read_text())
    for name, expected in manifest['archives'].items():
        data = (root / relative_path(name)).read_bytes()
        if len(data) != expected['bytes'] or digest(data) != expected['sha256']:
            raise ValueError(f'Archive hash mismatch: {name}')

    # Validate every member before creating anything. Existing identical files
    # are accepted; different files, symbolic links, and special files are not.
    pending = []
    for prefix, archive_name, archive_type in (
        ('bridge', 'completion_results_v7.tgz', 'tar'),
        ('v6_selected', 'v6_selected_records.zip', 'zip'),
    ):
        expected = manifest['record_files'][prefix]
        actual = {}
        if archive_type == 'tar':
            with tarfile.open(root / archive_name, 'r:gz') as archive:
                for member in archive.getmembers():
                    relative_path(member.name)
                    if member.isdir():
                        continue
                    if not member.isfile():
                        raise ValueError(f'Non-regular archive member: {member.name}')
                    if member.name in actual:
                        raise ValueError(f'Duplicate member: {member.name}')
                    actual[member.name] = archive.extractfile(member).read()
        else:
            with zipfile.ZipFile(root / archive_name) as archive:
                for member in archive.infolist():
                    relative_path(member.filename)
                    if member.is_dir():
                        continue
                    mode = (member.external_attr >> 16) & 0o170000
                    if mode not in (0, 0o100000):
                        raise ValueError(f'Non-regular ZIP member: {member.filename}')
                    if member.filename in actual:
                        raise ValueError(f'Duplicate member: {member.filename}')
                    actual[member.filename] = archive.read(member)
        if set(actual) != set(expected):
            raise ValueError(f'Unexpected or missing archive members: {prefix}')
        for name, data in actual.items():
            record = expected[name]
            if len(data) != record['bytes'] or digest(data) != record['sha256']:
                raise ValueError(f'Record hash mismatch: {prefix}/{name}')
            target = root / prefix / relative_path(name)
            if any(p.is_symlink() for p in (target, *target.parents)):
                raise ValueError(f'Symbolic link in extraction path: {target}')
            if target.exists():
                if not target.is_file() or target.read_bytes() != data:
                    raise FileExistsError(f'Refusing to overwrite: {target}')
            else:
                pending.append((target, data))
    for target, data in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(data)
    print(json.dumps({'status': 'HASH_VERIFIED_SAVED_RECORDS_READY',
                      'record_files': sum(len(v) for v in manifest['record_files'].values()),
                      'new_files_extracted': len(pending), 'model_fits': 0}))


if __name__ == '__main__':
    main()
