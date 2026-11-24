# 电商视频批量修整 Agent

把一条或一批视频、自然语言改单和参考素材整理成可审查的修整任务。每条视频记录目标、保护项、源文件哈希、时间轴、候选版本和同版检查。浏览器工作台可以并排检查原片和候选，并查看保护项和返修记录。

## 安装与试跑

需要 Python 3.10+ 和 FFmpeg。执行 `python3 -m pip install -r requirements.txt`，并确保 `ffmpeg` 在 PATH 中。可选的原生候选播放检查需要 Chrome 和 Node.js。

先将自己的视频和原始改单放到仓库内的 `cases/demo/`。例如准备 `source.mp4` 和 `request.txt`，再保存 `cases/demo/request.json`。

```json
{
  "batch_id": "demo",
  "user_input": "cases/demo/request.txt",
  "files": [{
    "id": "clip",
    "source": "cases/demo/source.mp4",
    "operations": [{
      "id": "remove-tag",
      "kind": "remove",
      "target": "商品旁的小贴纸",
      "protect": ["商品外观", "画面其他文字", "原声"]
    }]
  }]
}
```

```sh
python3 scripts/project.py intake --request cases/demo/request.json --case cases/demo/case.json
python3 scripts/project.py review-init --case cases/demo/case.json
python3 scripts/project.py serve --case cases/demo/case.json --port 8766
```

启动后访问 <http://127.0.0.1:8766/>。首次建档后可核对原片、保护项和待审状态。任务状态保存在 `runs/<batch_id>/workflow.json`。对有候选的批次，`project.py check --case <case.json>` 核对实际证据绑定，缺少成片或审查项时会报告待完成。

## 工作流与边界

- `scripts/project.py` 提供建档、定位、处理、审查和交付入口。运行 `--help` 查看各入口。
- `web/review.html` 展示批次、时间轴和同版审查材料。需要已有合法来源的回片时，可由宿主按任务导入。
- `prompts/` 定义宿主如何保存原始改单、选择处理路线以及检查每条视频。
- 公开配置只启用本地工作。远程上传、模型提交和历史批次授权均关闭。
- 项目仍需要宿主参与通用语义定位和方法选择。能建档、预览和审查不代表任意自然语言改单可以独立完成。使用自己的样本验证对应编辑方法与结果。

```sh
(cd scripts && python3 -m unittest test_local_raw_intake test_delivery_package)
node --test scripts/test_*.mjs
```

完整开发回归还包含历史素材、内部批次配置和旧版比较样本。公开仓库附带适合从新任务开始的本地检查。

代码采用 MIT。
