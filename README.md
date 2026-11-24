# pairwise-trigger-volume-enter

## 领域

游戏、图形与多媒体

## 主题

触发体积进出事件

## 说明

扫掠补交、帧内配对与卸载引用清理。

初始状态下 `gameplay/trigger_volume_enter.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
