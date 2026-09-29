import asyncio,json
from pathlib import Path
import local_mcp
async def main():
    p=Path('/root/lightsem/outputs')
    req=json.loads((p/'dance_investigation_reason_request.json').read_text())
    result=await local_mcp.perplexity.perplexity_reason(**req)
    (p/'dance_investigation_reason_response.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str))
    print('SAVED response')
asyncio.run(main())
