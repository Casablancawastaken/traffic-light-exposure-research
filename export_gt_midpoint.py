import argparse
import json
import math
from pathlib import Path

import cv2

import run_effects_pilot as core
import run_led_control_pilot as led

MODES = {'exp5_off':5.0,'exp5_weak':5.0,'exp5_partial':5.0,'exp10_period':10.0}
VARIANTS = ('clean','pwm','motion','combined')


def size_group(box):
    area = max(0, box[2]-box[0])*max(0, box[3]-box[1])
    return 'small' if area < 32**2 else ('medium' if area < 96**2 else 'large')


def area_visible(box, width, height):
    area = max(0.0,box[2]-box[0])*max(0.0,box[3]-box[1])
    if not area:
        return 0.0
    visible = max(0.0,min(width,box[2])-max(0,box[0]))*max(0.0,min(height,box[3])-max(0,box[1]))
    return visible/area


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project-root',default='.')
    p.add_argument('--count',type=int,default=20)
    p.add_argument('--out',default='outputs_bdd_control/gt_midpoint_export')
    args=p.parse_args()
    root=Path(args.project_root).resolve()
    output=root/args.out
    output.mkdir(parents=True,exist_ok=True)
    (output/'images').mkdir(exist_ok=True)
    data=json.loads((root/'outputs_bdd_control/predictions.json').read_text(encoding='utf-8'))
    export=[]
    checks=[]
    for mode, exposure in MODES.items():
        predpath=root/'outputs_bdd_control/led_phase20'/mode/'predictions.json'
        preds=json.loads(predpath.read_text(encoding='utf-8')) if predpath.exists() else None
        for i, record in enumerate(data[:args.count]):
            original_path, original = core.scene_image(record,root)
            scene=led.controlled_scene(record, original,640,led.COLORS[i%3])
            width,height=scene['size']
            for variant in VARIANTS:
                speed=90 if variant in ('motion','combined') else 0
                mid=exposure/2000
                scale=core.projection_scale(mid,speed,15)
                gt=[]
                for j,obj in enumerate(scene['gt']):
                    baseline_box=obj['box']
                    box=core.scale_box(baseline_box,scale,width,height)
                    v=area_visible(box,width,height)
                    gt.append({'object_index':j,'class':obj['class'], 'source':'inserted_control' if j==len(scene['gt'])-1 else 'BDD100K_natural',
                               'color_gt':obj.get('traffic_light_color'), 'box_xyxy_mid_exposure':[float(x) for x in box],
                               'size_group_from_clean_640px':size_group(baseline_box),
                               'visible_fraction_mid_exposure':v, 'evaluation_status':'exclude_geometry' if v<0.5 else 'evaluate'})
                item={'mode':mode,'variant':variant,'scene':record['image'],'scene_index':i+1,'source_image_sha256':core.checksum(original_path), 'source_label_sha256':record.get('sha256_label'),
                      'time_reference':'middle_of_exposure_t_over_2','time_ms':exposure/2,
                      'bbox_is_blur_trail':False,'objects':gt}
                export.append(item)
                if preds:
                    matched=preds[i]
                    assert matched['image']==record['image']
                    for c in ('person','car','traffic light'):
                        n=sum(g['class']==c for g in gt)
                        ct=matched['variants'][variant]['counts'][c]
                        checks.append({'mode':mode,'variant':variant,'scene':record['image'],'class':c,
                                       'gt_count':n,'detector_tp_plus_fn':ct['TP']+ct['FN'], 'matched':n==ct['TP']+ct['FN']})
            if i<5:
                image_path=root/'outputs_bdd_control/led_phase20'/mode/'images'/f'{i+1:02d}_motion.png'
                if image_path.exists():
                    frame=cv2.imread(str(image_path))
                    for g in gt:
                        if g['source']=='inserted_control':
                            rgb=(255,0,0)
                        else:
                            rgb=(0,200,0)
                        coords=[int(round(q)) for q in g['box_xyxy_mid_exposure']]
                        cv2.rectangle(frame,(coords[0],coords[1]),(coords[2],coords[3]),rgb,2)
                    cv2.putText(frame,f'{mode}: GT t/2, green natural / blue control',(10,22),cv2.FONT_HERSHEY_SIMPLEX,0.48,(255,255,255),2)
                    cv2.imwrite(str(output/'images'/f'{mode}_{i+1:02d}_GT_mid.png'),frame)
        print(f'{mode}: GT exported')
    (output/'gt_midpoint.json').write_text(json.dumps(export,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'gt_consistency.json').write_text(json.dumps({'checks':len(checks),'mismatches':[r for r in checks if not r['matched']]},ensure_ascii=False,indent=2),encoding='utf-8')
    if checks and not all(x['matched'] for x in checks):
        raise SystemExit('GT and saved TP+FN mismatch: check gt_consistency.json')
    print(f'GT records: {len(export)}, checks: {len(checks)}')


if __name__=='__main__':
    main()
