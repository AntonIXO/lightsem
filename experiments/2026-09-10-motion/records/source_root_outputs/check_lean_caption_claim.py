import json,hashlib
from pathlib import Path
import cv2,numpy as np
BASE=Path('/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/stage_lean_visual_audit_01')
labels=['STAGE02 SIGNED LEAN 0','STAGE03 SIGNED LEAN 2'];masks=[]
for text in labels:
 im=np.zeros((360,640,3),np.uint8);cv2.putText(im,text,(12,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
 masks.append(im[:30,:,0]>128)
rows=[]
for item in json.loads((BASE/'native_manifest.json').read_text()):
 path=BASE/item['pair'];im=cv2.imread(str(path));assert im.shape==(360,1280,3)
 recalls=[]
 for i,mask in enumerate(masks):
  patch=im[:30,i*640:(i+1)*640];white=patch.min(2)>80;recalls.append(float(white[mask].mean()))
 rows.append({'frame':item['frame'],'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'caption_expected_white_recall':recalls})
r={'scope':'Only45 already-decoded native sample pairs; expected caption glyph pixels threshold128 versus decoded all-channel>80. Not an all-frame or OCR proof.','minimum_recall_each_side':np.min([x['caption_expected_white_recall'] for x in rows],axis=0).tolist(),'rows':rows}
path=BASE/'parent_caption_check.json';assert not path.exists();path.write_text(json.dumps(r,indent=2));print(r['minimum_recall_each_side'])
