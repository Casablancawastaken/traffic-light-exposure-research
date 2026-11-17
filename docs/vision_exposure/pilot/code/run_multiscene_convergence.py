import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

import run_effects_pilot as core
import run_led_control_pilot as led

CASES = (
    ('short_off_stationary', 2.0, 0, 0.35),
    ('weak_threshold_fast', 5.0, 90, 0.175),
    ('long_fast_partial', 10.0, 90, 0.13),
)
STEPS = (16, 32, 64, 128)


def scene_ranking(records, root, width):
    scores=[]
    for i, rec in enumerate(records):
        _, photo=core.scene_image(rec,root)
        prepared=led.controlled_scene(rec,photo,width,led.COLORS[i%3])
        box=prepared['light']['gt_bbox']
        x1,y1,x2,y2=[int(round(q)) for q in box]
        patch=cv2.resize(photo,(prepared['size'][0],prepared['size'][1]))[y1:y2,x1:x2]
        brightness=float(np.mean(cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY))) if patch.size else 0.0
        scores.append((brightness,i,rec,prepared))
    return sorted(scores,key=lambda r:(r[0],r[1]))


def selected_scenes(ranked,count):
    if count>=len(ranked):return ranked
    picks=[]
    for i in range(count):
        idx=round(i*(len(ranked)-1)/(count-1)) if count>1 else len(ranked)//2
        picks.append(ranked[idx])
    return picks


def contrast_value(linear, prepared, bbox, scale):
    height,width=prepared['size'][1],prepared['size'][0]
    x1,y1,x2,y2=[max(0,int(round(x))) for x in bbox]
    x2=min(width,x2);y2=min(height,y2)
    if x2<=x1 or y2<=y1:return None
    patch=linear[y1:y2,x1:x2]
    moved_mask=core.warp(prepared['mask'].astype(np.float32),scale,width,height)
    mask=moved_mask[y1:y2,x1:x2]>128
    light=0.0722*patch[...,0]+0.7152*patch[...,1]+0.2126*patch[...,2]
    if not np.any(mask) or not np.any(~mask):return None
    fg=float(light[mask].mean());bg=float(light[~mask].mean())
    return (fg-bg)/max(fg+bg,1e-9)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project-root',default='.')
    p.add_argument('--count',type=int,default=5)
    p.add_argument('--pool',type=int,default=20)
    p.add_argument('--render-width',type=int,default=384)
    p.add_argument('--yolo',action='store_true')
    p.add_argument('--output',default='outputs_bdd_control/multiscene_convergence')
    args=p.parse_args()
    root=Path(args.project_root).resolve()
    entries=json.loads((root/'outputs_bdd_control/predictions.json').read_text(encoding='utf-8'))
    if args.pool<args.count or args.count<1:
        raise SystemExit('Invalid count/pool')
    pool=entries[:args.pool]
    ranked=scene_ranking(pool,root,args.render_width)
    chosen=selected_scenes(ranked,args.count)
    model=led.load_model(root,args.yolo)
    output=root/args.output
    images=output/'gt_images';images.mkdir(parents=True,exist_ok=True)
    rows=[];scene_manifest=[]
    for index,(brightness,original_index,rec,prepared) in enumerate(chosen,1):
        scene_manifest.append({'index':index,'image':rec['image'],'source_sha256':rec.get('sha256_image'), 'source_background_patch_brightness_0_255':brightness,
                               'source_label_sha256':rec.get('sha256_label'),'original_scene_index':original_index})
        for case_name, exposure, speed, phase in CASES:
            pars=argparse.Namespace(steps=16,exposure_ms=exposure,reference_exposure_ms=5.0,speed_kmh=speed,depth_m=15.0,pwm_hz=100.0,duty=0.2,phase=phase,imgsz=960,conf=0.1)
            test={}
            for n in STEPS:
                image,linear,gt=core.simulate(prepared,'combined',pars,steps=n)
                color=core.classify_color(image,gt[-1]['box'])
                predicted=core.infer_hybrid(model,image,pars) if model is not None else []
                counts=core.evaluate(gt,predicted) if model is not None else None
                end=core.light_evaluation(image,gt,predicted,prepared['light'])['status'] if model is not None else None
                midpoint_scale=core.projection_scale(exposure/2000.0,speed,pars.depth_m)
                score=contrast_value(linear,prepared,gt[-1]['box'],midpoint_scale)
                test[n]={'linear':linear,'gt':gt,'img':image,'color':color,'contrast':score,'counts':counts,'end':end}
            for low,high in zip(STEPS,STEPS[1:]):
                aa,bb=test[low],test[high]
                delta=np.abs(aa['linear']-bb['linear'])
                bbox_shift=max(abs(x-y) for ga,gb in zip(aa['gt'],bb['gt']) for x,y in zip(ga['box'],gb['box']))
                rows.append({'scene':rec['image'],'case':case_name,'exposure_ms':exposure,'speed_kmh':speed,'phase':phase,
                             'steps':f'{low}-{high}','mean_abs_linear_diff':float(delta.mean()),'max_abs_linear_diff':float(delta.max()),
                             'contrast_low':aa['contrast'],'contrast_high':bb['contrast'],
                             'abs_contrast_diff':abs(aa['contrast']-bb['contrast']) if aa['contrast'] is not None and bb['contrast'] is not None else None,
                             'gt_midbox_max_coord_shift':bbox_shift,'oracle_color_change':int(aa['color']!=bb['color']),
                             'detector_counts_change':int(aa['counts']!=bb['counts']) if args.yolo else None,
                             'detector_color_change':int(aa['end']!=bb['end']) if args.yolo else None,
                             'detector_64_128_status':bb['end'] if args.yolo and high==128 else None})
            if case_name=='weak_threshold_fast':
                marked=test[128]['img'].copy()
                for j,g in enumerate(test[128]['gt']):
                    box=[int(round(x)) for x in g['box']]
                    cv2.rectangle(marked,(box[0],box[1]),(box[2],box[3]),(255,0,0) if j==len(test[128]['gt'])-1 else (0,210,0),1)
                cv2.putText(marked,'GT at t/2: natural green, inserted blue',(5,18),cv2.FONT_HERSHEY_SIMPLEX,0.42,(255,255,255),1)
                cv2.imwrite(str(images/f'{index:02d}_GT_mid.png'),marked)
        print(f'{index}/{len(chosen)}: {rec["image"]} (background={brightness:.1f})',flush=True)
    with (output/'comparisons.csv').open('w',newline='',encoding='utf-8') as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    last=[x for x in rows if x['steps']=='64-128']
    failures=[r for r in last if r['mean_abs_linear_diff']>0.01 or r['abs_contrast_diff'] is None or r['abs_contrast_diff']>0.02 or r['gt_midbox_max_coord_shift']>1e-8 or r['oracle_color_change'] or (args.yolo and (r['detector_counts_change'] or r['detector_color_change']))]
    report={'scenes':len(chosen),'cases_per_scene':len(CASES),'total_step_pair_comparisons':len(rows), 'higher_res_step_comparisons':len(last),
            'detector_compared':args.yolo,'near_threshold_case_present':True,
            'max_mean_linear_diff_64_128':max(x['mean_abs_linear_diff'] for x in last),
            'max_abs_contrast_delta_64_128':max((x['abs_contrast_diff'] or 0) for x in last),
            'failures_64_128':failures, 'status':'PASS_NUMERICAL_ONLY' if not failures and args.yolo else 'INCOMPLETE_OR_FAILED',
            'scope':'Numerical convergence of 2D overlay on JPEG; not physical sensor validation'}
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'selected_scenes.json').write_text(json.dumps(scene_manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='failures_64_128'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
