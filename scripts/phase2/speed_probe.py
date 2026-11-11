import json
import sys
import time

from bazireason.mlx_backend import MLXChatModel
from bazireason.p2_prompts import P2_SYSTEM

path, label = sys.argv[1], sys.argv[2]
chat = MLXChatModel(path, label=label)
qs = [f"请用中文分步骤说明：公历 {1950 + 3 * i} 年（立春后）的年柱干支怎么算？已知 (年份−4) 除以 60 的余数对应六十甲子的序数，甲子为 0。" for i in range(16)]
convs = [[{"role": "system", "content": P2_SYSTEM}, {"role": "user", "content": q}] for q in qs]
out = {}
for batch in (8, 16):
    t = time.perf_counter()
    texts, stats = chat.generate_batch(convs, max_tokens=300, batch=batch)
    dt = time.perf_counter() - t
    out[f"batch{batch}"] = {"gen_tokens": stats["completion_tokens"], "prompt_tokens": stats["prompt_tokens"], "seconds": round(dt, 1), "gen_tps_wall": round(stats["completion_tokens"] / dt, 1), "peak_gb": round(stats["peak_memory_gb"], 2)}
text, s = chat.generate(convs[0], max_tokens=300)
out["single"] = {"gen_tps": round(s["generation_tps"], 1), "prompt_tps": round(s["prompt_tps"], 1)}
print(label, json.dumps(out, ensure_ascii=False))
print("SAMPLE:", texts[0][:500].replace("\n", " | "))
