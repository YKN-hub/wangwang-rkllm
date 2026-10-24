# RKLLM 转换

把 HuggingFace 的 Qwen2.5 权重转成 RK3588 NPU 能跑的 `.rkllm`。
不需要你自己的电脑装 WSL、不需要 CUDA，只需要浏览器点几下。

## 一次性准备（只做一遍，约 10 分钟）

1. 从瑞芯微官方 README 里的网盘下载 RKLLM SDK 压缩包（**GitHub 上没有**，releases 里也没有附件）：
   - `https://github.com/airockchip/rknn-llm` 页面底部两个网盘链接
   - `console.box.lenovo.com/l/l0tXb8`（提取码 `rkllm`）或 `console.zbox.filez.com/l/RJJDmB`
   - 找 `rknn-llm-release-v1.3.x.zip`
2. 解压，从里面取 **两个** 东西：
   - `rkllm-toolkit/packages/rkllm_toolkit-1.3.1-cp310-cp310-linux_x86_64.whl` ← **传到本仓库根目录**（7.2 MB，网页上传上限 25 MB，装得下）
   - `rkllm-runtime/Linux/librkllm_api/aarch64/librkllmrt.so` ← 盒子要的运行时，**留着别传**
   > wheel 文件名里的 `cp310` 必须对应 Python 3.10，`linux_x86_64` 必须是 x86 版；
   > 别用 `pip install git+https://github.com/airockchip/rknn-llm.git`，那样会缺预编译的 C++ 扩展，
   > 后面报 `No module named 'rkllm'`。
   > 已确认 PyPI 上查不到 `rkllm_toolkit`，这个 wheel 没有第二条来路。
   > 不想让 wheel 进 git 也行：CI 会退回去找 tag=`rkllm-wheel` 的 Release 并从里面下载。

## 仓库里必须有这几个文件

```
.github/workflows/export.yml     ← 注意 .github 这个带点的目录，少一层 Actions 就看不到
export_rkllm.py
requirements.txt                 ← 从 SDK 的 rkllm-toolkit/packages/ 里拷来的死-pin，别改
data_quant.json                  ← 量化校准语料，已从 SDK 里拷进仓库，CI 不再依赖外网
rkllm_toolkit-1.3.1-cp310-cp310-linux_x86_64.whl  ← 见上一节
```

> GitHub 网页版 "Add file → Upload files" 会把 `.github` 这种带点开头的目录名吃掉，
> 传上去变成根目录的 `workflows/`，Actions 就永远看不到这个 workflow。
> 要么用 git 命令行推，要么在网页上 "Add file → Create new file"，
> 在文件名框里**直接敲完整路径** `.github/workflows/export.yml`，GitHub 会自动补出目录层级。

## 怎么触发

**改 `.github/workflows/export.yml` 并推送 —— 推一次就跑一次**，不用谁去网页上点 Run workflow。
或者 Actions 页 → 左侧 "export-rkllm" → Run workflow：
- `model` 选 `Qwen/Qwen2.5-3B-Instruct`
- `dtypes` 是 JSON 数组，默认 `["W4A16","W8A8"]`（并行开两个 job）；只转一档就填 `["W4A16"]`

单次 3B 在 CI 的 4 核 CPU 上量化约 1~3 小时。日志里会打 `[1/3] load → [2/3] build → [3/3] export`。

## 拿回产物

跑完后 Release 列表里会出现两个新 Release，每个里面是产物（超过 1.9GB 的已自动分卷）+ `md5.txt`。

```bash
# 盒子上的落地路径（eMMC，47G 空闲，冷读 211MB/s；别放 SD 卡那边的 /opt/wangwang/models）
sudo mkdir -p /var/lib/wangwang-llm

# 分卷产物拼回去 + 校验（part 文件按名字顺序 cat）
cat Qwen2.5-3B-Instruct_W8A8_RK3588.rkllm.part_* > Qwen2.5-3B-Instruct_W8A8_RK3588.rkllm
md5sum Qwen2.5-3B-Instruct_W8A8_RK3588.rkllm      # 和 md5.txt 里那行一致才算完整
sudo mv Qwen2.5-3B-Instruct_W8A8_RK3588.rkllm /var/lib/wangwang-llm/
sudo chmod 644 /var/lib/wangwang-llm/*.rkllm
```

## 转换参数为什么这么定

| 参数 | 值 | 原因 |
|---|---|---|
| `device` | `cpu` | CI 没有 GPU |
| `load_huggingface(dtype=)` | `bfloat16` | 3B 用 float32 要 12GB 内存，runner 只有 16GB，会 OOM |
| `quantized_algorithm` | W8A8→`normal`，W4A16→`grq` | 官方注释明确配对，搞错会掉精度 |
| `target_platform` | `RK3588`（大写） | 产物按芯片编译，不能跨芯片用 |
| `max_context` | 2048 | 默认 4096 会在 8G 盒子上白占 NPU 内存；汪汪只需要 人设84字 + 4条历史 + 400字输出 |
| `num_npu_core` | 3 | RK3588 三核 NPU |
