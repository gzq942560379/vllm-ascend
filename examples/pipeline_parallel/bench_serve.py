#!/usr/bin/env python3
"""轻量在线吞吐压测：打 vllm /v1/completions，固定输入/输出长度+并发。
比 vllm bench 简单、依赖少（纯 stdlib），适合横向对照两个 endpoint。"""
import json, time, threading, urllib.request, sys, argparse

def gen_prompt(n_tokens_approx):
    # 用重复句子凑长度，token 数不精确但两个 endpoint 一致即可
    sent = "The quick brown fox jumps over the lazy dog. "
    return sent * (n_tokens_approx // 7 + 1)

def one_request(base_url, model, prompt, max_tokens, idx, results, sem):
    url = base_url.rstrip("/") + "/v1/completions"
    body = json.dumps({
        "model": model, "prompt": prompt,
        "max_tokens": max_tokens, "temperature": 0.0,
        "ignore_eos": True, "stream": False,
    }).encode()
    req = urllib.request.Request(url, data=body,
        headers={"Content-Type":"application/json"}, method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            d = json.loads(r.read())
        t1 = time.time()
        out_tok = d.get("usage", {}).get("completion_tokens", 0)
        results[idx] = dict(ok=True, t0=t0, t1=t1, out_tok=out_tok, err=None)
    except Exception as e:
        t1 = time.time()
        results[idx] = dict(ok=False, t0=t0, t1=t1, out_tok=0, err=str(e)[:200])
    finally:
        sem.release()

def run(base_url, model, num_prompts, concurrency, in_len, out_len):
    prompt = gen_prompt(in_len)
    results = [None]*num_prompts
    sem = threading.Semaphore(concurrency)
    threads = []
    t_start = time.time()
    for i in range(num_prompts):
        sem.acquire()
        th = threading.Thread(target=one_request,
            args=(base_url, model, prompt, out_len, i, results, sem))
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    t_end = time.time()
    ok = [r for r in results if r and r["ok"]]
    fail = [r for r in results if r and not r["ok"]]
    total_out = sum(r["out_tok"] for r in ok)
    wall = t_end - t_start
    # goodput: 总输出 token / 从首个请求开始到末个完成
    first_start = min(r["t0"] for r in ok) if ok else t_start
    last_end = max(r["t1"] for r in ok) if ok else t_end
    goodput_wall = last_end - first_start
    print(f"\n=== {base_url} ===")
    print(f"  prompts={num_prompts} concurrency={concurrency} in_len~{in_len} out_len={out_len}")
    print(f"  ok={len(ok)} fail={len(fail)} wall={wall:.1f}s goodput_wall={goodput_wall:.1f}s")
    print(f"  total_output_tokens={total_out}")
    print(f"  THROUGHPUT (output tok/s, goodput) = {total_out/goodput_wall:.1f}")
    if len(ok) > 1:
        lats = sorted(r["t1"]-r["t0"] for r in ok)
        n=len(lats)
        print(f"  per-req latency: min={lats[0]:.1f}s p50={lats[n//2]:.1f}s p95={lats[int(n*0.95)]:.1f}s max={lats[-1]:.1f}s")
        # 单请求吞吐 = out_len / latency
        solo = out_len / (sum(lats)/n)
        print(f"  per-req output tok/s (mean out_len/latency) = {solo:.1f}")
    if fail:
        print(f"  first fail: {fail[0]['err']}")
    return total_out/goodput_wall if ok else 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--num-prompts", type=int, default=64)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--in-len", type=int, default=256)
    ap.add_argument("--out-len", type=int, default=256)
    a = ap.parse_args()
    run(a.url, a.model, a.num_prompts, a.concurrency, a.in_len, a.out_len)
