import argparse
import hashlib
import json
from pathlib import Path


def checksum(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for part in iter(lambda:stream.read(1048576),b''):
            h.update(part)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project-root',default='.')
    p.add_argument('--manifest',required=True)
    p.add_argument('--config',required=True)
    p.add_argument('--out',default='source_hash_validation.json')
    args=p.parse_args()
    root=Path(args.project_root).resolve()
    manifest=json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    config=json.loads(Path(args.config).read_text(encoding='utf-8'))
    directory=root/'bdd_sample/data/images_100k/100k/train/shard_000'
    rows=[]
    for item in manifest:
        f=directory/item['image']
        actual=checksum(f) if f.is_file() else None
        rows.append({'image':item['image'],'expected_sha256':item['sha256'],'actual_sha256':actual,'status':'MISSING' if actual is None else ('PASS' if actual==item['sha256'] else 'FAIL')})
    weights=root/'yolo11s.pt'
    expected=config['weights_sha256']
    actual=checksum(weights) if weights.is_file() else None
    result={'source_count':len(rows),'source_pass':sum(x['status']=='PASS' for x in rows),'source_fail':sum(x['status']=='FAIL' for x in rows), 'source_missing':sum(x['status']=='MISSING' for x in rows),
            'weights':{'expected_sha256':expected,'actual_sha256':actual,'status':'MISSING' if actual is None else ('PASS' if actual==expected else 'FAIL')},'items':rows}
    result['status']='PASS' if result['source_pass']==len(rows) and result['weights']['status']=='PASS' else 'INCOMPLETE_OR_FAIL'
    Path(args.out).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='items'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
