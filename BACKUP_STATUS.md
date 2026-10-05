# Research backup status

## October 5, 2026 update

The new backup is now COMPLETE: all seven archives passed full per-file SHA256 readback verification. Their manifests cover 247,828 archived files. Archive hashes and sizes are retained in backup_manifests/cross_attribution_2026-10-05.json; the older archive inventory is in backup_manifests/outputs_2026-10-01.json. Raw archives remain local at the two directories described below. A subsequent authenticated-upload attempt still returned GitHub CLI HTTP 401 and missing Git HTTPS credentials; no upload is confirmed.

All six CS/GE RefCOCO, RefCOCO+ and RefCOCOg inference cells and their saved-prediction evaluations are complete. New full-precision metrics, prediction metadata, evaluation audits, scripts and continuity notes are being committed locally for the explicitly requested GitHub backup.

GitHub publication remains blocked: the October 5 push attempt could not obtain HTTPS credentials, and the installed GitHub CLI reports an invalid token for mohsin651. No remote upload is confirmed. Authenticate using C:/CREMI/trdp/.tools/github-cli/bin/gh.exe auth login --hostname github.com --git-protocol https --web before pushing and uploading artifacts.

The new output tree (247,839 files, 9,577,108,677 bytes at backup start) is being archived under C:/CREMI/trdp/research-backup-2026-10-05 by scripts/backup_cross_attribution.py. Each completed archive has a per-file SHA256 manifest and is read back in full for verification. BACKUP_MANIFEST.json is written only once every group completes; absence of that file means the backup job has not yet finished. The earlier October 1 archives remain intact. Raw NPZ arrays and diagnostic logs are preserved in these archives even when Git excludes them. Both archive sets still require off-machine upload.

Public datasets, pretrained checkpoints, environments and external source checkouts are outside the research-output archives. Their restoration sources and hashes are retained in ARTIFACTS.md and experiment provenance. Do not claim the entire original workspace is hosted on GitHub.

Local commit b418a6a4 preserves research through RefCOCOg, CLIP Surgery and Grad-ECLIP, including full-precision numeric tables and prediction metadata. Git push failed because authentication was unavailable; remote publication is NOT confirmed. GitHub CLI also reports an invalid token. Renew login before retrying push and release upload.

All 129,746 files under outputs as of the pre-cross-product snapshot are preserved in 16 verified archives totaling 3,411,527,538 bytes at C:/CREMI/trdp/research-backup-2026-10-01. Every archived file was checked against its SHA256 manifest. These archives are LOCAL ONLY and have not been uploaded. They precede the new cross-attribution generalization experiment and do not contain its new outputs.

Raw maps, all candidate arrays and old galleries are included in those archives, but excluded from Git itself. Datasets, pretrained weights, Python environments, external source checkouts and the parent paper PDF remain outside this output snapshot. See ARTIFACTS.md and retained provenance for restoration.
