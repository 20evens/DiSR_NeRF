"""
NeRF SR Web App - FastAPI Backend
"""
import os, sys, glob, uuid, json, threading, base64, io, re, hashlib, secrets
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional, List

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import create_engine, Column, String, Integer
from sqlalchemy.orm import declarative_base, sessionmaker, Session
import jwt as pyjwt
from pydantic import BaseModel
import subprocess
import asyncio

# ─── Paths ────────────────────────────────────────────────────────────────────
BACKEND_DIR  = Path(__file__).parent
WORKSPACE    = BACKEND_DIR.parent.parent          # nerf-pytorch-master root
DB_PATH      = BACKEND_DIR / "users.db"
PYTHON_EXE   = sys.executable

# ─── Auth Config ──────────────────────────────────────────────────────────────
SECRET_KEY   = "nerf-sr-webapp-secret-2024-xK9p"
ALGORITHM    = "HS256"
TOKEN_EXPIRE = 24  # hours

# ─── Database ─────────────────────────────────────────────────────────────────
Base    = declarative_base()
engine  = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
Session_ = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class UserDB(Base):
    __tablename__ = "users"
    id              = Column(Integer, primary_key=True, index=True)
    username        = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)

Base.metadata.create_all(bind=engine)

def get_db():
    db = Session_()
    try:
        yield db
    finally:
        db.close()

# ─── Auth Helpers ─────────────────────────────────────────────────────────────
bearer = HTTPBearer(auto_error=False)

PBKDF2_ITER = 260_000

def hash_pw(pw: str) -> str:
    salt = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), PBKDF2_ITER)
    return f"{salt}${h.hex()}"

def verify_pw(plain: str, stored: str) -> bool:
    parts = stored.split("$")
    if len(parts) != 2:
        return False
    salt, digest = parts
    h = hashlib.pbkdf2_hmac("sha256", plain.encode(), salt.encode(), PBKDF2_ITER)
    return secrets.compare_digest(h.hex(), digest)

def make_token(username: str) -> str:
    exp = datetime.now(timezone.utc) + timedelta(hours=TOKEN_EXPIRE)
    return pyjwt.encode({"sub": username, "exp": exp}, SECRET_KEY, algorithm=ALGORITHM)

def current_user(
    cred: HTTPAuthorizationCredentials = Depends(bearer),
    db: Session = Depends(get_db)
):
    if not cred:
        raise HTTPException(status_code=401, detail="未登录")
    try:
        payload  = pyjwt.decode(cred.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="无效令牌")
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="令牌已过期，请重新登录")
    except pyjwt.PyJWTError:
        raise HTTPException(status_code=401, detail="无效令牌")
    user = db.query(UserDB).filter(UserDB.username == username).first()
    if not user:
        raise HTTPException(status_code=401, detail="用户不存在")
    return user

# ─── Pydantic Schemas ─────────────────────────────────────────────────────────
class RegisterReq(BaseModel):
    username: str
    password: str

class LoginReq(BaseModel):
    username: str
    password: str

class RunReq(BaseModel):
    mode: str                    # "sr" | "nerf" | "sr_nerf"
    input_dir: str
    output_dir: str
    ddim_steps: int   = 10
    nerf_iters: int   = 200000
    sr_scale: str     = "2x"    # informational only
    config_file: str  = ""       # optional custom NeRF config path

# ─── Job Manager ──────────────────────────────────────────────────────────────
class Job:
    def __init__(self, job_id: str):
        self.job_id    = job_id
        self.lines: List[str] = []
        self.done      = False
        self.success   = False
        self.progress  = 0.0
        self.video_path: Optional[str] = None
        self._lock     = threading.Lock()
        self._cursor   = 0

    def add(self, line: str):
        with self._lock:
            self.lines.append(line)

    def pop_new(self) -> List[str]:
        with self._lock:
            new          = self.lines[self._cursor:]
            self._cursor = len(self.lines)
            return new

jobs: dict[str, Job] = {}

# ─── Process Runner ───────────────────────────────────────────────────────────
def _run_job(job: Job, cmd: List[str], cwd: str, mode: str, nerf_iters: int):
    try:
        job.add(f"[启动] {' '.join(cmd)}\n")
        proc = subprocess.Popen(
            cmd, cwd=cwd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1
        )

        sr_weight   = 20.0
        nerf_weight = 70.0
        sr_done     = False

        for raw in proc.stdout:
            line = raw.rstrip()
            job.add(line)

            # ── 进度估算 ──────────────────────────────────────────────────────
            if mode == "sr":
                if "阶段 1/3" in line:   job.progress = 5.0
                elif "阶段 2/3" in line: job.progress = 15.0
                elif "阶段 3/3" in line: job.progress = 80.0
                elif "完成" in line and "SR" in line: job.progress = 99.0

            elif mode == "nerf":
                m = re.search(r'(?:Iter|iter|step|Step)[:\s]+(\d+)', line)
                if m:
                    step = int(m.group(1))
                    job.progress = min(10.0 + step / nerf_iters * 80.0, 90.0)
                if "Saved test set" in line:
                    job.progress = 95.0

            elif mode == "sr_nerf":
                if not sr_done:
                    if "阶段 1/3" in line:   job.progress = 2.0
                    elif "阶段 2/3" in line: job.progress = 8.0
                    elif "阶段 3/3" in line: job.progress = 15.0
                    elif "超分辨率完成" in line or ("完成" in line and "split" in line.lower()):
                        sr_done = True
                        job.progress = sr_weight
                else:
                    m = re.search(r'(?:Iter|iter|step|Step)[:\s]+(\d+)', line)
                    if m:
                        step = int(m.group(1))
                        job.progress = min(sr_weight + step / nerf_iters * nerf_weight, 90.0)
                    if "Saved test set" in line:
                        job.progress = 95.0

        proc.wait()
        job.success  = (proc.returncode == 0)
        job.progress = 100.0
        status_txt   = "成功" if job.success else f"失败（返回码 {proc.returncode}）"
        job.add(f"\n[完成] 处理{status_txt}")

        # 寻找生成的视频文件
        if job.success:
            output_root = Path(jobs[job.job_id].video_path or "") or Path(cwd)
            for mp4 in Path(cwd).rglob("*.mp4"):
                job.video_path = str(mp4)
                break

    except Exception as e:
        job.add(f"[异常] {e}")
        job.success = False
    finally:
        job.done = True

# ─── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(title="NeRF SR Studio")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ─── Auth Endpoints ───────────────────────────────────────────────────────────
@app.post("/auth/register")
def register(req: RegisterReq, db: Session = Depends(get_db)):
    if not req.username.strip():
        raise HTTPException(400, "用户名不能为空")
    if not req.password:
        raise HTTPException(400, "密码不能为空")
    if db.query(UserDB).filter(UserDB.username == req.username).first():
        raise HTTPException(400, "用户名已存在")
    db.add(UserDB(username=req.username, hashed_password=hash_pw(req.password)))
    db.commit()
    return {"message": "注册成功，请登录"}

@app.post("/auth/login")
def login(req: LoginReq, db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.username == req.username).first()
    if not user or not verify_pw(req.password, user.hashed_password):
        raise HTTPException(401, "用户名或密码错误")
    return {"access_token": make_token(user.username), "username": user.username}

# ─── File Browser ─────────────────────────────────────────────────────────────
@app.get("/files/browse")
def browse(path: str = "", _u=Depends(current_user)):
    base = Path(path) if path else WORKSPACE
    if not base.exists() or not base.is_dir():
        raise HTTPException(404, "目录不存在")
    try:
        entries = sorted(
            [{"name": p.name, "path": str(p).replace("\\", "/"), "is_dir": p.is_dir()}
             for p in base.iterdir()],
            key=lambda x: (not x["is_dir"], x["name"].lower())
        )
    except PermissionError:
        raise HTTPException(403, "无访问权限")
    return {
        "current": str(base).replace("\\", "/"),
        "parent":  str(base.parent).replace("\\", "/") if base.parent != base else None,
        "entries": entries,
    }

@app.get("/files/preview")
def preview(path: str, _u=Depends(current_user)):
    folder = Path(path)
    if not folder.exists() or not folder.is_dir():
        raise HTTPException(404, "目录不存在")
    imgs = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG"]:
        imgs.extend(folder.glob(ext))
    imgs.sort()
    results = []
    for p in imgs[:6]:
        try:
            from PIL import Image as PILImage
            with PILImage.open(p) as im:
                im.thumbnail((200, 200))
                buf = io.BytesIO()
                im.convert("RGB").save(buf, "JPEG", quality=75)
                b64 = base64.b64encode(buf.getvalue()).decode()
                results.append({"name": p.name, "data": f"data:image/jpeg;base64,{b64}"})
        except Exception:
            pass
    return {"images": results}

@app.get("/files/video")
def serve_video(path: str, _u=Depends(current_user)):
    vp = Path(path)
    if not vp.exists():
        raise HTTPException(404, "视频文件不存在")
    return FileResponse(str(vp), media_type="video/mp4")

@app.get("/files/find_video")
def find_video(path: str, _u=Depends(current_user)):
    base = Path(path)
    for mp4 in base.rglob("*.mp4"):
        return {"path": str(mp4).replace("\\", "/")}
    return {"path": None}

# ─── Process Endpoints ────────────────────────────────────────────────────────
@app.post("/process/run")
async def run_process(req: RunReq, background_tasks: BackgroundTasks, _u=Depends(current_user)):
    job_id = uuid.uuid4().hex[:8]
    job    = Job(job_id)
    jobs[job_id] = job

    cwd  = str(WORKSPACE)
    idir = req.input_dir.replace("\\", "/")
    odir = req.output_dir.replace("\\", "/")

    # Determine config file
    cfg = req.config_file if req.config_file else _pick_config(idir)

    if req.mode == "sr":
        cmd = [PYTHON_EXE, "test_full.py",
               "--input_dir",  idir,
               "--output_dir", odir]

    elif req.mode == "nerf":
        cmd = [PYTHON_EXE, "run_nerf.py",
               "--config",   cfg,
               "--datadir",  idir,
               "--basedir",  odir,
               "--N_iters",  str(req.nerf_iters)]

    elif req.mode == "sr_nerf":
        cmd = [PYTHON_EXE, "run_nerf.py",
               "--config",       cfg,
               "--datadir",      idir,
               "--basedir",      odir,
               "--sr_preprocess",
               "--N_iters",      str(req.nerf_iters)]
    else:
        raise HTTPException(400, "无效的运行模式")

    background_tasks.add_task(
        _run_job, job, cmd, cwd, req.mode, req.nerf_iters
    )
    return {"job_id": job_id}

def _pick_config(datadir: str) -> str:
    """Detect blender vs llff and pick an appropriate config."""
    p = Path(datadir)
    if (p / "transforms_train.json").exists() or (p / "transforms_test.json").exists():
        cfg = WORKSPACE / "configs" / "lego_sr.txt"
        return str(cfg).replace("\\", "/") if cfg.exists() else "configs/lego_sr.txt"
    return "configs/lego.txt"

def _verify_token_param(token: str = "", db: Session = Depends(get_db)):
    """SSE专用：从query param验证token（EventSource不支持自定义header）"""
    try:
        payload  = pyjwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(401, "无效令牌")
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(401, "令牌已过期")
    except pyjwt.PyJWTError:
        raise HTTPException(401, "无效令牌")
    user = db.query(UserDB).filter(UserDB.username == username).first()
    if not user:
        raise HTTPException(401, "用户不存在")
    return user

@app.get("/process/logs/{job_id}")
async def stream_logs(job_id: str, _u=Depends(_verify_token_param)):
    if job_id not in jobs:
        raise HTTPException(404, "任务不存在")
    job = jobs[job_id]

    async def generate():
        while True:
            lines = job.pop_new()
            for line in lines:
                payload = json.dumps({"type": "log", "data": line, "progress": job.progress},
                                     ensure_ascii=False)
                yield f"data: {payload}\n\n"
            if job.done and not job.pop_new():
                payload = json.dumps({
                    "type": "done",
                    "success": job.success,
                    "progress": job.progress,
                    "video_path": job.video_path,
                }, ensure_ascii=False)
                yield f"data: {payload}\n\n"
                break
            await asyncio.sleep(0.15)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.get("/process/status/{job_id}")
def job_status(job_id: str, _u=Depends(current_user)):
    if job_id not in jobs:
        raise HTTPException(404, "任务不存在")
    j = jobs[job_id]
    return {"done": j.done, "success": j.success, "progress": j.progress,
            "video_path": j.video_path}

@app.get("/health")
def health():
    return {"status": "ok", "workspace": str(WORKSPACE)}

@app.get("/")
def index():
    html = BACKEND_DIR.parent / "index.html"
    if not html.exists():
        raise HTTPException(404, "index.html not found")
    return FileResponse(str(html), media_type="text/html")

# ─── Entry ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
