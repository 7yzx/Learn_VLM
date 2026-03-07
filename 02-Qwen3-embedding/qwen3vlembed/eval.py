import argparse
import glob
import json
import math
import os
import random
import re

import torch
from swift.infer_engine import InferRequest, TransformersEngine

def _find_latest_checkpoint(output_dir: str) -> str | None:
    checkpoints = glob.glob(os.path.join(output_dir, "v*/checkpoint-*"))
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)


def _load_eval_samples(jsonl_path: str, n: int) -> list[dict]:
    samples: list[dict] = []
    if jsonl_path.endswith(".json"):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            arr = json.load(f)
        for obj in arr[:n]:
            image_path = obj["images"][0]
            pos_content = obj["positive_messages"][0][0]["content"]
            caption = pos_content.replace("<image>", "").strip()
            samples.append({"image_path": image_path, "caption": caption})
    else:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if len(samples) >= n:
                    break
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                image_path = obj["images"][0]
                pos_content = obj["positive_messages"][0][0]["content"]
                caption = pos_content.replace("<image>", "").strip()
                samples.append({"image_path": image_path, "caption": caption})
    if len(samples) < n:
        raise ValueError(f"Not enough samples in {jsonl_path}: got {len(samples)}, need {n}")
    return samples


def _build_eval_requests(samples: list[dict]) -> tuple[list[InferRequest], list[InferRequest]]:
    queries: list[InferRequest] = []
    candidates: list[InferRequest] = []
    for s in samples:
        img_path = os.path.abspath(s["image_path"])
        caption = s["caption"]
        queries.append(
            InferRequest(
                messages=[{"role": "user", "content": "<image> Find a description for this image."}],
                images=[img_path],
            )
        )
        candidates.append(InferRequest(messages=[{"role": "user", "content": caption}]))
    return queries, candidates


def _embed(engine: TransformersEngine, reqs: list[InferRequest]) -> torch.Tensor:
    resps = engine.infer(reqs)
    embs = [torch.tensor(r.data[0].embedding, dtype=torch.float32) for r in resps]
    return torch.stack(embs, dim=0)


def _validate_embeddings(name: str, x: torch.Tensor) -> None:
    if x.ndim != 2:
        raise ValueError(f"{name}: expected 2D tensor, got shape={tuple(x.shape)}")
    finite_mask = torch.isfinite(x)
    non_finite = int((~finite_mask).sum().item())
    norms = torch.linalg.vector_norm(x, ord=2, dim=1)
    zero_norm = int((norms == 0).sum().item())
    if non_finite or zero_norm:
        n = x.shape[0]
        msg = (
            f"{name}: invalid embeddings detected (n={n}). "
            f"non_finite_values={non_finite}, zero_norm_rows={zero_norm}. "
            "This usually means the finetuned checkpoint has diverged (NaN/Inf weights) "
            "or the model is producing all-zero vectors."
        )
        raise ValueError(msg)


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, p=2, dim=1)


def _scores(q: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return _normalize(q) @ _normalize(c).T


def _top1_accuracy(scores: torch.Tensor) -> tuple[float, list[int]]:
    preds = scores.argmax(dim=1).tolist()
    correct = sum(int(i == p) for i, p in enumerate(preds))
    return correct / scores.shape[0], preds


def _summarize(scores: torch.Tensor) -> dict:
    n = scores.shape[0]
    diag = scores.diag()
    diag_mean = diag.mean().item()
    offdiag_sum = (scores.sum() - diag.sum()).item()
    offdiag_mean = offdiag_sum / (n * (n - 1)) if n > 1 else float("nan")
    return {"diag_mean": diag_mean, "offdiag_mean": offdiag_mean, "gap": diag_mean - offdiag_mean}


def _parse_int_list(s: str) -> list[int]:
    items = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    if not items:
        raise ValueError("Empty list")
    return items


def _full_metrics(scores: torch.Tensor, recall_ks: list[int]) -> dict:
    if not torch.isfinite(scores).all():
        raise ValueError("scores contains NaN/Inf; metrics are invalid. Check finetuned embeddings/checkpoint.")
    n = scores.shape[0]
    gt = torch.arange(n, device=scores.device)

    metrics: dict = {"recall": {}, "mrr": None, "n": n}
    max_k = max(recall_ks)
    topk = scores.topk(k=min(max_k, n), dim=1).indices
    for k in recall_ks:
        k = min(k, n)
        hit = (topk[:, :k] == gt[:, None]).any(dim=1).float().mean().item()
        metrics["recall"][k] = hit

    gt_scores = scores.diag()
    ranks = (scores > gt_scores[:, None]).sum(dim=1).float() + 1.0
    metrics["mrr"] = (1.0 / ranks).mean().item()
    return metrics


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    return re.findall(r"[a-z0-9]+", text)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _bm25_prepare(docs: list[list[str]]):
    n = len(docs)
    df: dict[str, int] = {}
    doc_lens = [len(d) for d in docs]
    avgdl = sum(doc_lens) / n if n else 0.0
    for d in docs:
        for t in set(d):
            df[t] = df.get(t, 0) + 1
    idf = {t: math.log((n - c + 0.5) / (c + 0.5) + 1.0) for t, c in df.items()}
    tfs: list[dict[str, int]] = []
    for d in docs:
        m: dict[str, int] = {}
        for t in d:
            m[t] = m.get(t, 0) + 1
        tfs.append(m)
    return {"idf": idf, "tfs": tfs, "doc_lens": doc_lens, "avgdl": avgdl}


def _bm25_score(query: list[str], doc_tf: dict[str, int], idf: dict[str, float], doc_len: int, avgdl: float) -> float:
    k1 = 1.2
    b = 0.75
    score = 0.0
    for t in query:
        if t not in doc_tf:
            continue
        tf = doc_tf[t]
        w = idf.get(t, 0.0)
        denom = tf + k1 * (1.0 - b + b * (doc_len / avgdl if avgdl else 0.0))
        score += w * (tf * (k1 + 1.0) / denom if denom else 0.0)
    return score


def _hard_pools(captions: list[str], *, hard_k: int, random_k: int, sim: str, seed: int) -> list[list[int]]:
    n = len(captions)
    if n <= 1:
        return [[0]]

    token_lists = [_tokenize(c) for c in captions]
    token_sets = [set(toks) for toks in token_lists]
    bm25_state = _bm25_prepare(token_lists) if sim == "bm25" else None

    pools: list[list[int]] = []
    for i in range(n):
        sims: list[tuple[float, int]] = []
        for j in range(n):
            if i == j:
                continue
            if sim == "bm25":
                s = _bm25_score(
                    token_lists[i],
                    bm25_state["tfs"][j],
                    bm25_state["idf"],
                    bm25_state["doc_lens"][j],
                    bm25_state["avgdl"],
                )
            else:
                s = _jaccard(token_sets[i], token_sets[j])
            sims.append((s, j))
        sims.sort(reverse=True, key=lambda x: x[0])
        hard = [j for _, j in sims[: min(hard_k, n - 1)]]

        rng = random.Random(seed + i)
        banned = set(hard)
        banned.add(i)
        remaining = [j for j in range(n) if j not in banned]
        rand = rng.sample(remaining, k=min(random_k, len(remaining))) if random_k > 0 else []

        pool = [i] + hard + rand
        pools.append(pool)
    return pools


def _hard_metrics(q: torch.Tensor, c: torch.Tensor, pools: list[list[int]], recall_ks: list[int]) -> dict:
    qn = _normalize(q)
    cn = _normalize(c)
    if not torch.isfinite(qn).all() or not torch.isfinite(cn).all():
        raise ValueError("normalized embeddings contains NaN/Inf; hard metrics are invalid.")
    n = qn.shape[0]
    hits = {k: 0 for k in recall_ks}
    rr_sum = 0.0
    for i in range(n):
        pool = pools[i]
        csub = cn[pool]
        s = torch.mv(csub, qn[i])
        pos = 0
        pos_score = s[pos].item()
        rank = int((s > pos_score).sum().item()) + 1
        rr_sum += 1.0 / rank
        for k in recall_ks:
            if rank <= k:
                hits[k] += 1
    return {"recall": {k: hits[k] / n for k in recall_ks}, "mrr": rr_sum / n, "n": n, "pool_size": len(pools[0])}


def _print_metrics(prefix: str, metrics: dict, recall_ks: list[int]):
    parts = [f"MRR={metrics['mrr']:.4f}"]
    for k in recall_ks:
        parts.append(f"R@{k}={metrics['recall'][min(k, metrics['n'])]:.4f}")
    if "pool_size" in metrics:
        parts.append(f"pool={metrics['pool_size']}")
    print(f"{prefix} " + " ".join(parts))


def run_multimodal_eval(
    *,
    num_images: int,
    batch_size: int,
    device_map: str,
    jsonl_path: str,
    eval_mode: str,
    recall_ks: list[int],
    hard_k: int,
    random_k: int,
    hard_sim: str,
    seed: int,
    ckpt: str | None,
):
    output_dir = "output/qwen3-vl-emb-lora"
    latest_checkpoint: str | None = None
    if ckpt is not None:
        if ckpt.strip().lower() == "none":
            latest_checkpoint = None
            print("Adapter disabled (--ckpt none). Evaluating base model only.")
        else:
            ckpt_path = os.path.abspath(ckpt)
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            latest_checkpoint = ckpt_path
            print(f"Loading adapter from: {latest_checkpoint}")
    else:
        latest_checkpoint = _find_latest_checkpoint(output_dir)
        if latest_checkpoint is None:
            print("No checkpoints found. Please run train.sh first, or pass --ckpt <path>.")
        else:
            print(f"Loading adapter from: {latest_checkpoint}")

    model_path = "./models/Qwen3-VL-Embedding-2B"
    base_engine = TransformersEngine(
        model_path,
        task_type="embedding",
        attn_impl="eager",
        device_map=device_map,
        max_batch_size=batch_size,
    )
    tuned_engine = None
    if latest_checkpoint is not None:
        tuned_engine = TransformersEngine(
            model_path,
            task_type="embedding",
            attn_impl="eager",
            adapters=[latest_checkpoint],
            device_map=device_map,
            max_batch_size=batch_size,
        )

    samples = _load_eval_samples(jsonl_path, n=num_images)
    queries, candidates = _build_eval_requests(samples)
    captions = [s["caption"] for s in samples]
    pools = None
    if eval_mode in {"hard", "both"}:
        pools = _hard_pools(captions, hard_k=hard_k, random_k=random_k, sim=hard_sim, seed=seed)

    print(f"Evaluating {num_images} images (query=image+instruction, candidate=text caption)...")
    base_q = _embed(base_engine, queries)
    base_c = _embed(base_engine, candidates)
    _validate_embeddings("base_queries", base_q)
    _validate_embeddings("base_candidates", base_c)

    print("\nBase model:")
    if eval_mode in {"full", "both"}:
        base_scores = _scores(base_q, base_c)
        _print_metrics("full:", _full_metrics(base_scores, recall_ks), recall_ks)
        base_acc, base_preds = _top1_accuracy(base_scores)
        base_sum = _summarize(base_scores)
        print(
            f"top1_acc={base_acc:.2%} diag_mean={base_sum['diag_mean']:.4f} "
            f"offdiag_mean={base_sum['offdiag_mean']:.4f} gap={base_sum['gap']:.4f}"
        )
        bad = [(i, p) for i, p in enumerate(base_preds) if i != p]
        if bad:
            print(f"mismatches={bad[:10]}")
        else:
            print("mismatches=none")
    if eval_mode in {"hard", "both"}:
        _print_metrics("hard:", _hard_metrics(base_q, base_c, pools, recall_ks), recall_ks)

    if tuned_engine is None:
        return

    tuned_q = _embed(tuned_engine, queries)
    tuned_c = _embed(tuned_engine, candidates)
    _validate_embeddings("tuned_queries", tuned_q)
    _validate_embeddings("tuned_candidates", tuned_c)

    print("\nFinetuned model:")
    if eval_mode in {"full", "both"}:
        tuned_scores = _scores(tuned_q, tuned_c)
        _print_metrics("full:", _full_metrics(tuned_scores, recall_ks), recall_ks)
        tuned_acc, tuned_preds = _top1_accuracy(tuned_scores)
        tuned_sum = _summarize(tuned_scores)
        print(
            f"top1_acc={tuned_acc:.2%} diag_mean={tuned_sum['diag_mean']:.4f} "
            f"offdiag_mean={tuned_sum['offdiag_mean']:.4f} gap={tuned_sum['gap']:.4f}"
        )
        bad = [(i, p) for i, p in enumerate(tuned_preds) if i != p]
        if bad:
            print(f"mismatches={bad[:10]}")
        else:
            print("mismatches=none")
    if eval_mode in {"hard", "both"}:
        _print_metrics("hard:", _hard_metrics(tuned_q, tuned_c, pools, recall_ks), recall_ks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--jsonl", type=str, default="dataset/eval.json")
    parser.add_argument("--eval_mode", type=str, default="both", choices=["full", "hard", "both"])
    parser.add_argument("--recall_ks", type=str, default="1,5,10")
    parser.add_argument("--hard_k", type=int, default=15)
    parser.add_argument("--random_k", type=int, default=15)
    parser.add_argument("--hard_sim", type=str, default="jaccard", choices=["jaccard", "bm25"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()

    run_multimodal_eval(
        num_images=args.num_images,
        batch_size=args.batch_size,
        device_map=args.device,
        jsonl_path=args.jsonl,
        eval_mode=args.eval_mode,
        recall_ks=_parse_int_list(args.recall_ks),
        hard_k=args.hard_k,
        random_k=args.random_k,
        hard_sim=args.hard_sim,
        seed=args.seed,
        ckpt=args.ckpt,
    )


if __name__ == "__main__":
    main()
