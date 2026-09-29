from pathlib import Path
from PIL import Image
ROOT=Path('/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/music_physical_revision_01/visual_review_support_revision_01/renders/holds_083_1200')
for frame in (0,300,599):
 src=Image.open(ROOT/f'sample_{frame:06d}.jpg');out=Image.new('RGB',(1920,1080))
 for j,c in enumerate((0,3,4)):out.paste(src.crop((640*c,0,640*(c+1),1080)),(640*j,0))
 out.save(ROOT.parent.parent/f'parent_preview_{frame:06d}.png')
