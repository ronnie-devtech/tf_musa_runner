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
