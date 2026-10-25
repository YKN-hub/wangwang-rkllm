"""RKLLM 模型转换: HuggingFace 权重 -> .rkllm (RK3588 NPU 用)

照抄 airockchip/rknn-llm 官方 examples/rkllm_api_demo/export/export_rkllm.py 的调用,
只改了三点: device 走 cpu(无 GPU)、load 用 bfloat16(省一半内存, 3B float32 要 12GB)、
按芯片挡掉非法的量化档。
"""
import argparse
import os

from rkllm.api import RKLLM

# 官方 SDK 1.3.1 文档 p.19 的按芯片支持表。抄错一行的代价是在 CI 上白等一小时,
# 所以这里挡在调用之前, 并直接把合法值打出来。
SUPPORTED = {
    "RK3588": ("W8A8", "W8A8_G128", "W8A8_G256", "W8A8_G512"),
    "RK3576": ("W4A16", "W4A16_G32", "W4A16_G64", "W4A16_G128", "W8A8"),
    "RV1126B": ("W4A16", "W4A16_G32", "W4A16_G64", "W4A16_G128", "W8A8"),
    "RK3562": ("W8A8", "W4A16_G32", "W4A16_G64", "W4A16_G128", "W4A8_G32"),
}

# 官方: w8a8 用 normal, w4a16 用 grq。grq 只支持 w4a16 系且必须有 GPU, CPU runner 上别想。
ALGO = {"W8A8": "normal", "W8A8_G128": "normal", "W8A8_G256": "normal", "W8A8_G512": "normal",
        "W4A16": "grq", "W4A16_G32": "grq", "W4A16_G64": "grq", "W4A16_G128": "grq"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="已下载好的 HF 权重目录")
    ap.add_argument("--dtype", default="W8A8", choices=sorted(ALGO))
    ap.add_argument("--platform", default="RK3588")
    ap.add_argument("--npu-core", type=int, default=3)
    # 汪汪只需要 84 字人设 + 4 条历史 + 400 token 输出, 4096 会在 8G 盒子上白占 NPU 内存
    ap.add_argument("--max-context", type=int, default=2048)
    ap.add_argument("--calib", default="./data_quant.json")
    ap.add_argument("--out-dir", default="./out")
    # 权重目录在 CI 里统一叫 ./src, 拿它当产物名会得到 src_W8A8_RK3588.rkllm ——
    # 换模型也是同一个名字, 两个 Release 会撞车, 所以必须能显式指定。
    ap.add_argument("--name", default=None, help="产物文件名前缀")
    args = ap.parse_args()

    legal = SUPPORTED.get(args.platform.upper())
    if legal is None:
        raise SystemExit(f"未知平台 {args.platform}, 已知: {sorted(SUPPORTED)}")
    if args.dtype not in legal:
        raise SystemExit(f"{args.platform} 不支持 {args.dtype}; 合法值只有 {list(legal)}")

    os.makedirs(args.out_dir, exist_ok=True)
    name = args.name or os.path.basename(os.path.abspath(args.model).rstrip("/"))
    out = os.path.join(args.out_dir, f"{name}_{args.dtype}_{args.platform}.rkllm")
    if os.path.exists(out):
        print(f"[skip] 已存在 {out}")
        return

    llm = RKLLM()
    print(f"[1/3] load_huggingface {args.model} (device=cpu, dtype=bfloat16)", flush=True)
    ret = llm.load_huggingface(model=args.model, device="cpu", dtype="bfloat16")
    if ret != 0:
        raise SystemExit(f"load 失败 ret={ret}")

    print(f"[2/3] build dtype={args.dtype} algo={ALGO[args.dtype]} "
          f"platform={args.platform} max_context={args.max_context}", flush=True)
    ret = llm.build(do_quantization=True, optimization_level=1,
                    quantized_dtype=args.dtype.lower(), quantized_algorithm=ALGO[args.dtype],
                    target_platform=args.platform, num_npu_core=args.npu_core,
                    extra_qparams=None, dataset=args.calib, hybrid_rate=0,
                    max_context=args.max_context)
    if ret != 0:
        raise SystemExit(f"build 失败 ret={ret}")

    print(f"[3/3] export -> {out}", flush=True)
    ret = llm.export_rkllm(out)
    if ret != 0:
        raise SystemExit(f"export 失败 ret={ret}")
    size = os.path.getsize(out) / 1048576.0
    print(f"[done] {out}  {size:.1f} MB", flush=True)
    with open(os.path.join(args.out_dir, "md5.txt"), "a", encoding="utf-8") as f:
        os.system(f"md5sum '{out}' >> '{os.path.join(args.out_dir, 'md5.txt')}'")


if __name__ == "__main__":
    main()
