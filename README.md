# tf_musa_runner

## 基本使用

### 运行 Graph / MetaGraph

```bash
python musa_run_pb_graph.py \
  --spec ./meta_graph/meta_graph_1.spec \
  --bs 32 \
  --musa-plugin /path/to/your/libmusa_plugin.so
```
---

### 参数总览

| 参数              | 类型      | 默认值        | 说明                                     |
| --------------- | ------- | ---------- | -------------------------------------- |
| `--spec`        | str     | None       | 单个 `.spec` 文件路径（MetaGraphDef）          |
| `--spec_dir`    | str     | None       | 批量模式：递归扫描目录下所有 `.spec`                 |
| `--pb`          | str     | None       | 指定 frozen graph `.pb` 文件路径（仅单 spec 可用） |
| `--bs`          | str/int | 1024       | batch size（支持单值或逗号分隔，如 `1,8,16,32`）    |
| `--unknown_dim` | int     | 1          | 非 batch 维度的填充值                         |
| `--seed`        | int     | 2026       | 随机数种子（输入数据生成）                          |
| `--warmup`      | int     | 3          | warmup 轮数（不计时）                         |
| `--run_iters`   | int     | 10         | 正式 benchmark 迭代次数                      |
| `--out_root`    | str     | runner_out | 输出结果目录                                 |
| `--strict`      | bool    | True       | 是否在失败时退出                               |

---

### 设备相关参数

| 参数                       | 类型   | 默认值            | 说明                          |
| ------------------------ | ---- | -------------- | --------------------------- |
| `--device`               | str  | /device:MUSA:0 | 运行设备                        |
| `--allow_soft_placement` | bool | True           | 是否允许 TensorFlow 自动 fallback |
| `--log_device_placement` | bool | False          | 是否打印算子设备分配日志                |

### 支持设备类型

```bash
/device:MUSA:0
/device:CPU:0
```

---

### MUSA 插件参数

| 参数              | 类型  | 默认值  | 说明                           |
| --------------- | --- | ---- | ---------------------------- |
| `--musa-plugin` | str | auto | MUSA 插件路径（libmusa_plugin.so） |

默认路径：

```
../tensorflow_musa_extension/build/libmusa_plugin.so
```

---

### PB / 转换相关参数

| 参数                 | 类型  | 默认值                   | 说明             |
| ------------------ | --- | --------------------- | -------------- |
| `--convert_script` | str | convert_spec_to_pb.py | spec → pb 转换脚本 |

### 自动逻辑

当未指定 `--pb` 时：

1. 自动检测 frozen_graph_xxx.pb
2. 未找到则调用 convert_script 生成
3. 仍失败则报错退出

---

## 推理输入传输优化

当图里存在大量 `Placeholder` / `feed_dict` 输入时，wall time 往往不只包含 kernel compute，还会包含：
- CPU pageable 内存到 pinned bounce buffer 的拷贝
- H2D 专用 stream 与 compute stream 之间的大量 `event_record + stream_wait`
- TensorFlow feed 调度本身的固定开销

当前 runner / plugin 组合支持以下优化开关：

| 环境变量 | 默认值 | 位置 | 作用 | 推荐程度 |
| --- | --- | --- | --- | --- |
| `MUSA_PINNED_FEED` | `0` | `musa_run_pb_graph.py` | runner 侧把 feed 输入分配为 pinned host memory | **推荐优先开启** |
| `MUSA_PINNED_H2D_ON_COMPUTE_STREAM` | `0` | `tensorflow_musa_extension/musa_ext/mu/device/musa_device.cc` | 对 pinned host memory，H2D 不再走 `h2d_stream_ + event/wait`，直接排到 compute stream | **建议配合 pinned feed 开启** |
| `MUSA_PAGEABLE_H2D_ON_COMPUTE_STREAM` | `0` | `tensorflow_musa_extension/musa_ext/mu/device/musa_device.cc` | 对普通 pageable host memory，先拷到 pinned bounce buffer，再把 H2D 直接排到 compute stream | **过渡方案** |

### 语义说明

- `MUSA_PINNED_H2D_ON_COMPUTE_STREAM`
  - 适用于源输入已经是 pinned host memory 的情况
  - 主要收益是减少每个输入一次跨 stream 的 `event_record + stream_wait`
  - 不会消除 H2D 本身的数据传输成本

- `MUSA_PAGEABLE_H2D_ON_COMPUTE_STREAM`
  - 适用于源输入仍是普通 pageable host memory 的情况
  - 只能减少跨 stream 同步，不能消除 pageable -> pinned bounce copy
  - 一般收益低于 pinned feed 路径

### 推荐开启顺序

1. 先开 `MUSA_PINNED_FEED=1`
2. 再开 `MUSA_PINNED_H2D_ON_COMPUTE_STREAM=1`
3. 如果暂时不能改 feed buffer 分配方式，再尝试 `MUSA_PAGEABLE_H2D_ON_COMPUTE_STREAM=1`

### 推荐场景

- 推理场景
- 单次 `sess.run` 有大量 feed tensor
- profile 显示 kernel compute 明显低于 wall time，且 H2D / feed 调度占比高

### 不推荐默认开启的场景

- 输入路数很少
- H2D 和 compute 本来能有效重叠
- 非推理场景或 host tensor 生命周期不稳定

### 推荐命令

```bash
#MUSA_VISIBLE_DEVICES=2 \
MUSA_PINNED_FEED=1 \
MUSA_PINNED_H2D_ON_COMPUTE_STREAM=1 \
python musa_run_pb_graph.py \
  --spec ./meta_graph/meta_graph_3.spec \
  --bs 1024 \
  --warmup 3 \
  --run_iters 20
```

### 过渡方案

```bash
MUSA_PAGEABLE_H2D_ON_COMPUTE_STREAM=1 \
MUSA_VISIBLE_DEVICES=2 \
MUSA_ENABLE_TF32=1 \
python musa_run_pb_graph.py \
  --spec ./meta_graph/meta_graph_3.spec \
  --bs 1024 \
  --warmup 3 \
  --run_iters 20
```

对于 `meta_graph_3` 这类 feed 路数很多的推理图，优先级应为：
- `MUSA_PINNED_FEED=1`：主收益来源
- `MUSA_PINNED_H2D_ON_COMPUTE_STREAM=1`：进一步压缩 event/wait 调度开销
- `MUSA_PAGEABLE_H2D_ON_COMPUTE_STREAM=1`：仅作为未接入 pinned feed 时的替代方案

---

## 输入参数示例

### 单模型运行

```bash
python musa_run_pb_graph.py \
  --spec model.spec \
  --bs 32 \
  --warmup 5 \
  --run_iters 10
```

---

### 多 batch 测试

```bash
python musa_run_pb_graph.py \
  --spec model.spec \
  --bs 1,8,16,32,64
```

---

### 批量目录运行

```bash
python musa_run_pb_graph.py \
  --spec_dir ./meta_graph/
```

---

### 指定 PB 跳过转换

```bash
python musa_run_pb_graph.py \
  --spec model.spec \
  --pb frozen_graph.pb
```

---

### CPU 调试模式

```bash
python musa_run_pb_graph.py \
  --spec model.spec \
  --device /device:CPU:0
```

---
