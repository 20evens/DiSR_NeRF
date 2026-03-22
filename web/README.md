# NeRF SR Studio - Web App

## 快速启动（无需 Node.js）

### 1. 安装后端依赖
```bash
pip install -r web/backend/requirements.txt
```

### 2. 启动后端
```bash
cd web/backend
python main.py
```

### 3. 访问
浏览器打开 **http://localhost:8000**

> 前端以单一 HTML 文件托管于后端，React / Tailwind CSS 均通过 CDN 加载，**无需安装 Node.js**。

---

## 功能说明

| 功能 | 说明 |
|------|------|
| 注册/登录 | 用户数据存储于 `web/backend/users.db` |
| 图像超分辨率 | 调用 `test_full.py`，800×800 → 1600×1600 |
| 三维重建 | 调用 `run_nerf.py`，输出渲染结果 |
| 超分辨率+三维重建 | 先 SR 预处理，再 NeRF 训练 |
| 实时日志 | 通过 SSE 推送运行日志到前端 |
| 视频播放 | 自动检测输出目录中的 .mp4 文件 |

## 项目结构

```
web/
├── backend/
│   ├── main.py          # FastAPI 后端（认证 + 进程管理 + SSE）
│   ├── requirements.txt
│   └── users.db         # SQLite 用户数据库（自动创建）
└── frontend/
    ├── src/
    │   ├── App.jsx
    │   ├── api.js        # API 客户端
    │   ├── pages/
    │   │   ├── AuthPage.jsx    # 登录/注册页
    │   │   └── Dashboard.jsx   # 主界面
    │   └── index.css
    └── package.json
```
