"""Archive all cross-attribution outputs and verify every archived file."""
import hashlib
import json
from pathlib import Path
import zipfile

root = Path('outputs/cross_attribution_generalization')
dest = Path('C:/CREMI/trdp/research-backup-2026-10-05')
dest.mkdir(exist_ok=True)
records = []
groups = [('cross_attribution_root', sorted(p for p in root.iterdir() if p.is_file()))]
groups += [(f'{d}_{s}', sorted((root/d/s).rglob('*'))) for d in ['refcoco', 'refcocoplus', 'refcocog'] for s in ['CS', 'GE']]
for name, paths in groups:
    target = dest/(name+'.zip')
    assert not target.exists(), f'Archive already exists: {target}'
    manifest = []
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for path in paths:
            if not path.is_file():
                continue
            data = path.read_bytes()
            rel = path.relative_to(root).as_posix()
            archive.writestr(rel, data)
            manifest.append({'path': rel, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    with zipfile.ZipFile(target) as archive:
        assert len(archive.namelist()) == len(manifest)
        for record in manifest:
            assert hashlib.sha256(archive.read(record['path'])).hexdigest() == record['sha256']
    (dest/(name+'.files.json')).write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    digest = hashlib.file_digest(target.open('rb'), 'sha256').hexdigest()
    records.append({'archive': target.name, 'bytes': target.stat().st_size, 'sha256': digest, 'files': len(manifest), 'verified': True})
    print(name, len(manifest), target.stat().st_size, 'verified', flush=True)
(dest/'BACKUP_MANIFEST.json').write_text(json.dumps({'source': str(root.resolve()), 'archives': records}, indent=2)+'\n', encoding='utf-8')
