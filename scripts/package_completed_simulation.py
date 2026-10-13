#!/usr/bin/env python3
"""Create a portable, allowlisted delivery from already completed records."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

REPO=Path(__file__).resolve().parents[1]
TASK=REPO.parent
OLD=TASK.parent/'dml_original_optimized/output/sensitivity_v6_server_20261002/evidence'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    payload=TASK/'delivery/Simulation_Completion'
    payload.mkdir(parents=True,exist_ok=False)
    def copy(source,target):
        assert source.is_file() and not source.is_symlink(),source
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        assert sha(source)==sha(target)
    # No Git metadata, virtual environments, credentials, or unrelated projects.
    for name in ('README.md','REPRODUCE.txt','pyproject.toml','.gitignore','.gitattributes'):
        copy(REPO/name,payload/name)
    for prefix in ('src','scripts','configs','tests','results','output/pdf'):
        for p in sorted((REPO/prefix).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and not p.name.startswith('._'):
                copy(p,payload/p.relative_to(REPO))
    for name in ('completion_results_v7.tgz','server_source_v7.tgz'):
        copy(TASK/name,payload/'evidence'/name)
    for p in sorted((TASK/'collected_server').rglob('*')):
        if p.is_file():copy(p,payload/'evidence/bridge'/p.relative_to(TASK/'collected_server'))
    frozen=json.loads((OLD/'output/formal/FROZEN_MANIFEST.json').read_text())
    selected=payload/'evidence/v6_selected'
    for name,expected in frozen['identity']['source_hashes'].items():
        assert sha(OLD/name)==expected
        copy(OLD/name,selected/name)
    for name in ('output/formal/FROZEN_MANIFEST.json','output/formal/RESULT_LEDGER.json',
                 'output/formal/analysis/SUMMARY.json','INDEPENDENT_VERIFICATION.json','SERVER_START.json','SERVER_EXIT.json'):
        copy(OLD/name,selected/name)
    for dgp in ('linear','nonlinear'):
        for rep in range(200):
            name=f'output/formal/figure2_{dgp}_n300_k2_base_rep{rep:04d}.json'
            copy(OLD/name,selected/name)
    for name in ('independent_recount.json','pdf_qa.json','remote_final_verification.json'):
        copy(TASK/'qa'/name,payload/'verification'/name)
    files={str(p.relative_to(payload)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(payload.rglob('*')) if p.is_file()}
    manifest={'schema_version':1,'status':'COMPLETED_SAVED_EVIDENCE_PACKAGE',
              'base_commit':'b01dd19f22d44c3ca5dd6771772d1cb8f31fb755','new_server_jobs':600,'reused_existing_v6_jobs':400,
              'primary_figures':6,'tables':8,'extension_figures':32,'local_model_fits':0,
              'historical_full_v6_raw_archive_not_bundled':True,'historical_v6_subset_is_explicit':True,
              'original_records_rewritten':False,'github_changed':False,'files':files}
    (payload/'PACKAGE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    archive=TASK/'Simulation_Completion_Package.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(payload.rglob('*')):
            if p.is_file():z.write(p,str(Path('Simulation_Completion')/p.relative_to(payload)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,record in files.items():
            data=z.read('Simulation_Completion/'+name)
            assert len(data)==record['bytes'] and hashlib.sha256(data).hexdigest()==record['sha256']
    print(json.dumps({'status':'PACKAGED_AND_VERIFIED','archive':str(archive),'archive_sha256':sha(archive),
                      'bytes':archive.stat().st_size,'files':len(files)+1,'portable_root':str(payload)}))


if __name__=='__main__':main()
