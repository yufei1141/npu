"""Refresh public metadata only; never downloads the multi-GB tensor payload."""
import urllib.request,json,hashlib
from pathlib import Path
root=Path(__file__).parents[1]/'sources'
mid='openbmb/MiniCPM-V-4.7-1B-GPTQ'
# Stable revision avoids fetching a mixed set of changing main files.
with urllib.request.urlopen('https://huggingface.co/api/models/'+mid,timeout=20) as r:d=json.load(r)
sha=d['sha']; records=[]
for name,out in [('config.json','model_config.json'),('quantize_config.json','quantize_config.json'),('model.safetensors.index.json','weight_index.json')]:
 url=f'https://huggingface.co/{mid}/resolve/{sha}/{name}'
 with urllib.request.urlopen(url,timeout=20) as r:raw=r.read()
 json.loads(raw)
 (root/out).write_bytes(raw)
 records.append(dict(file=out,url=url,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)))
(root/'model_revision.json').write_text(json.dumps({k:d.get(k) for k in ['id','sha','createdAt','lastModified']},indent=2))
(root/'metadata_refresh.json').write_text(json.dumps(records,indent=2))
print('Refreshed 3 metadata files at revision',sha,'; full model not downloaded.')
