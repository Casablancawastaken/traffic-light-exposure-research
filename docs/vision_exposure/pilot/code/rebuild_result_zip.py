import argparse
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True)
    p.add_argument('--out',required=True)
    args=p.parse_args()
    data=Path(args.data).resolve()
    if not (data/'manifest.json').is_file():
        raise SystemExit('manifest.json not found')
    target=Path(args.out).resolve()
    target.parent.mkdir(parents=True,exist_ok=True)
    with ZipFile(target,'w',compression=ZIP_DEFLATED,compresslevel=5) as z:
        for file in sorted(data.rglob('*')):
            if file.is_file():
                z.write(file,arcname='led_phase20/'+file.relative_to(data).as_posix())
    print(target)


if __name__=='__main__':
    main()
