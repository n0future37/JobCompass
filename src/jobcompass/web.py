"""本机单体API；同步I/O在线程池运行，导入/向量化由独立worker处理。"""

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select, update

from .career import Application, Background, Career, ConflictError, Preference, applications, preferences
from .recommendation_agent import RecommendationAgent, RecommendationRequest
from .explanations import ExplanationConflict
from .resumes import ConfirmResume, ResumeError, ResumeStore
from .conditions import SearchRequest
from .identity import Identity
from .importing import TextImportRequest, prepare_text_import
from .preprocessing import digest
from .service import RagService
from .errors import QueryCapacityError
from .settings import ROOT
from .storage import jobs
from .tasks import TaskQueue

logger = logging.getLogger("jobcompass.api")


@asynccontextmanager
async def lifespan(app):
    app.state.service = RagService()
    yield
    app.state.service.close()


app = FastAPI(title="JobCompass", version="0.2.0", lifespan=lifespan)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request.state.request_id = str(uuid.uuid4())
    started = time.perf_counter()
    origin = request.headers.get("origin")
    if (
        request.method not in {"GET", "HEAD", "OPTIONS"}
        and origin
        and origin.rstrip("/") != str(request.base_url).rstrip("/")
    ):
        from fastapi.responses import JSONResponse

        response = JSONResponse({"detail": "请求来源不受信任"}, status_code=403)
    else:
        response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = "no-store"
    logger.info(
        json.dumps(
            dict(
                request_id=request.state.request_id,
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                total_ms=round((time.perf_counter() - started) * 1000, 2),
            )
        )
    )
    return response


def service(request: Request):
    return request.app.state.service


def current_user(request: Request, svc=Depends(service)):
    user = Identity(svc.store).current(request.cookies.get("jc_auth", ""))
    if not user:
        raise HTTPException(401, "请先登录")
    return dict(user)


def rate_limit(request, svc, operation, limit):
    import redis

    key = (
        svc.settings.prefix
        + "limit:"
        + operation
        + ":"
        + digest(request.client.host if request.client else "local")
    )
    try:
        count = svc.redis.eval(
            "local n=redis.call('INCR',KEYS[1]); if n==1 then redis.call('EXPIRE',KEYS[1],60) end; return n",
            1,
            key,
        )
    except redis.RedisError:
        raise HTTPException(503, "限流服务暂不可用") from None
    if count > limit:
        raise HTTPException(429, "操作过于频繁，请稍后再试", headers={"Retry-After": "60"})


class Credentials(BaseModel):
    username: str = Field(pattern=r"^[a-zA-Z0-9_]{3,32}$")
    password: str = Field(min_length=8, max_length=128)


@app.post("/api/auth/register", status_code=201)
def register(value: Credentials, request: Request, svc=Depends(service)):
    rate_limit(request, svc, "register", 10)
    try:
        uid = Identity(svc.store).register(value.username, value.password)
    except ValueError:
        raise HTTPException(409, "用户名已存在") from None
    return {"user_id": uid}


@app.post("/api/auth/login")
def login(value: Credentials, request: Request, response: Response, svc=Depends(service)):
    rate_limit(request, svc, "login", 20)
    try:
        token = Identity(svc.store).login(value.username, value.password)
    except ValueError:
        raise HTTPException(401, "用户名或密码错误") from None
    response.set_cookie(
        "jc_auth",
        token,
        httponly=True,
        samesite="strict",
        max_age=86400 * 7,
        secure=request.url.scheme == "https",
    )
    return {"authenticated": True}


@app.post("/api/auth/logout")
def logout(request: Request, response: Response, svc=Depends(service)):
    Identity(svc.store).logout(request.cookies.get("jc_auth", ""))
    response.delete_cookie("jc_auth")
    return {"authenticated": False}


@app.get("/api/me")
def me(user=Depends(current_user)):
    return user


class Profile(BaseModel):
    text: str = Field(max_length=4000)


@app.put("/api/profile")
def profile(value: Profile, user=Depends(current_user), svc=Depends(service)):
    Identity(svc.store).profile(user["id"], value.text)
    return {"text": value.text}


@app.get("/")
def home():
    return FileResponse(ROOT / "web/index.html")


@app.get("/app.js")
def javascript():
    return FileResponse(ROOT / "web/app.js", media_type="application/javascript")


@app.get("/account.js")
def account_javascript():
    return FileResponse(ROOT / "web/account.js", media_type="application/javascript")


@app.get("/api/health")
def health(svc=Depends(service)):
    from sqlalchemy import text

    result = {
        "api": "ok",
        "mysql": "unavailable",
        "milvus": "unavailable",
        "redis": "unavailable",
    }
    try:
        with svc.store.engine.connect() as c:
            c.execute(text("SELECT 1"))
        result["mysql"] = "ok"
    except Exception:
        pass
    try:
        from .retrieval import milvus

        client = milvus(svc.settings)
        try:
            client.list_collections()
            result["milvus"] = "ok"
        finally:
            client.close()
    except Exception:
        pass
    try:
        svc.redis.ping()
        result["redis"] = "ok"
    except Exception:
        pass
    return result


@app.post("/api/search")
def search(
    value: SearchRequest,
    request: Request,
    user=Depends(current_user),
    svc=Depends(service),
):
    rate_limit(request, svc, "search", 60)
    if "profile" not in value.model_fields_set:
        value.profile = user["profile"].get("text", "")
    try:
        return svc.query(value, owner=user["id"], request_id=request.state.request_id)
    except QueryCapacityError:
        raise HTTPException(503, "查询容量暂满，请稍后重试", headers={"Retry-After": "2"}) from None
    except Exception as error:
        logger.warning(
            json.dumps(
                dict(
                    request_id=request.state.request_id,
                    stage="search",
                    error_type=type(error).__name__,
                )
            )
        )
        raise HTTPException(503, detail=f"查询暂未完成（{type(error).__name__}），请查看服务状态") from None


@app.get("/api/jobs")
def list_jobs(
    offset: int = 0,
    limit: int = 20,
    city: str | None = None,
    keyword: str | None = None,
    svc=Depends(service),
):
    if offset < 0 or not 1 <= limit <= 100 or (city and len(city) > 50) or (keyword and len(keyword) > 100):
        raise HTTPException(422, "筛选或分页范围无效")
    return svc.store.page(offset, limit, city, keyword)


@app.get("/api/jobs/{job_key}")
def detail(job_key: str, svc=Depends(service)):
    row = svc.store.current([job_key]).get(job_key)
    if not row:
        raise HTTPException(404, "岗位不存在或已下架")
    return row


class JobStatus(BaseModel):
    active: bool


@app.patch("/api/jobs/{job_key}/status")
def job_status(job_key: str, value: JobStatus, user=Depends(current_user), svc=Depends(service)):
    from sqlalchemy.dialects.mysql import insert

    from .storage import indexes

    with svc.store.engine.begin() as c:
        row = (
            c.execute(
                select(jobs).where(jobs.c.job_key == job_key, jobs.c.owner == user["id"]).with_for_update()
            )
            .mappings()
            .first()
        )
        if not row:
            raise HTTPException(404, "岗位不存在或不属于当前账号")
        c.execute(update(jobs).where(jobs.c.job_key == job_key).values(active=value.active))
        pairs = c.execute(
            select(jobs.c.job_key, jobs.c.content_hash, jobs.c.version)
            .where(jobs.c.active.is_(True))
            .order_by(jobs.c.job_key)
        ).all()
        snapshot = digest([list(p) for p in pairs])
        c.execute(
            insert(indexes)
            .values(snapshot=snapshot, status="pending", payload={})
            .on_duplicate_key_update(snapshot=snapshot)
        )
    return {"job_key": job_key, "active": value.active, "index_status": "pending"}


@app.get("/api/history")
def history(user=Depends(current_user), svc=Depends(service)):
    return svc.store.list_history(user["id"])


class Feedback(BaseModel):
    query_id: uuid.UUID
    useful: bool
    note: str = Field(default="", max_length=1000)


@app.post("/api/feedback")
def feedback(value: Feedback, user=Depends(current_user), svc=Depends(service)):
    try:
        fid = svc.store.add_feedback(user["id"], str(value.query_id), value.model_dump(mode="json"))
    except ValueError:
        raise HTTPException(404, "查询不存在") from None
    return {"feedback_id": fid}


class ImportRequest(BaseModel):
    filename: str = Field(pattern=r"^[a-zA-Z0-9_.-]{1,100}\.(json|csv)$")
    content: str = Field(min_length=1, max_length=5_000_000)
    idempotency_key: str = Field(min_length=1, max_length=100)


def public_task(row):
    return {key: row[key] for key in ("id", "status", "attempts", "result")}


@app.post("/api/imports", status_code=202)
def submit_import(
    value: ImportRequest,
    request: Request,
    user=Depends(current_user),
    svc=Depends(service),
):
    rate_limit(request, svc, "import", 10)
    content = value.content.encode("utf-8")
    if len(content) > 5_000_000:
        raise HTTPException(413, "导入文件不能超过5MB")
    file_hash = digest(content)
    directory = ROOT / "data/imports" / svc.settings.stage / user["id"] / file_hash
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ("original" + (".csv" if value.filename.endswith(".csv") else ".json"))
    try:
        with path.open("xb") as f:
            f.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise HTTPException(409, "原始文件指纹冲突") from None
    try:
        row = TaskQueue(svc.store).submit(
            user["id"],
            value.idempotency_key,
            dict(
                path=path.relative_to(ROOT).as_posix(),
                file_sha256=file_hash,
                filename=value.filename,
            ),
        )
    except ValueError:
        raise HTTPException(409, "该幂等键已用于其他文件") from None
    return public_task(row)


@app.post("/api/imports/text", status_code=202)
def submit_text_import(
    value: TextImportRequest, request: Request, user=Depends(current_user), svc=Depends(service)
):
    packaged = ImportRequest.model_validate(prepare_text_import(value, user["id"]))
    return submit_import(packaged, request, user, svc)


@app.get("/api/imports/{task_id}")
def task_status(task_id: uuid.UUID, user=Depends(current_user), svc=Depends(service)):
    row = TaskQueue(svc.store).get(user["id"], str(task_id))
    if not row:
        raise HTTPException(404, "任务不存在")
    return public_task(row)


@app.post("/api/imports/{task_id}/retry")
def retry_task(task_id: uuid.UUID, user=Depends(current_user), svc=Depends(service)):
    if not TaskQueue(svc.store).retry(user["id"], str(task_id)):
        raise HTTPException(409, "任务不存在或不处于失败状态")
    return {"status": "queued"}


@app.get("/api/raw")
def raw(path: str, request: Request, svc=Depends(service)):
    target = (ROOT / path).resolve()
    allowed = target.is_relative_to((ROOT / "data/raw").resolve()) and target.suffix == ".body"
    if target.is_relative_to((ROOT / "data/imports").resolve()):
        user = Identity(svc.store).current(request.cookies.get("jc_auth", ""))
        allowed = (
            bool(user)
            and target.is_relative_to((ROOT / "data/imports" / svc.settings.stage / user["id"]).resolve())
            and target.name in {"original.json", "original.csv"}
        )
    if not allowed or not target.is_file():
        raise HTTPException(404, "原始响应不存在")
    return FileResponse(target, media_type="text/plain; charset=utf-8")


@app.get("/api/background")
def get_background(user=Depends(current_user), svc=Depends(service)):
    return Career(svc.store).background(user["id"])


@app.put("/api/background")
def put_background(value: Background, user=Depends(current_user), svc=Depends(service)):
    return Career(svc.store).background(user["id"], value)


@app.get("/api/job-preferences")
def get_preferences(user=Depends(current_user), svc=Depends(service)):
    return Career(svc.store).records(user["id"], preferences)


@app.put("/api/job-preferences/{job_key}")
def put_preference(job_key: str, value: Preference, user=Depends(current_user), svc=Depends(service)):
    try:
        return Career(svc.store).put(user["id"], job_key, value)
    except LookupError:
        raise HTTPException(404, "岗位不存在") from None


@app.get("/api/applications")
def get_applications(user=Depends(current_user), svc=Depends(service)):
    return Career(svc.store).records(user["id"], applications)


@app.put("/api/applications/{job_key}")
def put_application(job_key: str, value: Application, user=Depends(current_user), svc=Depends(service)):
    try:
        return Career(svc.store).put(user["id"], job_key, value)
    except ConflictError:
        raise HTTPException(409, "记录已变化，请刷新后重试") from None
    except LookupError:
        raise HTTPException(404, "岗位不存在") from None


@app.post("/api/recommendations")
def recommend(
    value: RecommendationRequest, request: Request, user=Depends(current_user), svc=Depends(service)
):
    rate_limit(request, svc, "recommend", 30)
    try:
        return RecommendationAgent(svc).run(value, user["id"], request.state.request_id)
    except Exception as error:
        logger.warning(
            json.dumps(
                dict(request_id=request.state.request_id, stage="recommend", error_type=type(error).__name__)
            )
        )
        raise HTTPException(503, "推荐暂未完成，请稍后重试或查看服务状态") from None


class ExplanationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_key: str = Field(min_length=1, max_length=192)


@app.get("/api/recommendations/{query_id}/explanation")
def get_explanation(
    query_id: uuid.UUID,
    request: Request,
    job_key: str = Query(min_length=1, max_length=192),
    user=Depends(current_user),
    svc=Depends(service),
):
    try:
        result = svc.explanations.get(user["id"], str(query_id), job_key)
        return dict(result, request_id=request.state.request_id)
    except LookupError:
        raise HTTPException(404, "推荐或岗位不存在") from None
    except ExplanationConflict as error:
        raise HTTPException(409, str(error)) from None
    except Exception as error:
        logger.warning(
            json.dumps(
                dict(
                    request_id=request.state.request_id,
                    stage="read_explanation",
                    error_type=type(error).__name__,
                )
            )
        )
        raise HTTPException(503, "理由状态暂不可用，请稍后重试") from None


@app.post("/api/recommendations/{query_id}/explanation")
def generate_explanation(
    query_id: uuid.UUID,
    value: ExplanationRequest,
    request: Request,
    response: Response,
    user=Depends(current_user),
    svc=Depends(service),
):
    rate_limit(request, svc, "recommendation_explanation", 20)
    try:
        result = svc.explanations.generate(user["id"], str(query_id), value.job_key, request.state.request_id)
        if result["status"] == "running":
            response.status_code = 202
            response.headers["Retry-After"] = "2"
        return dict(result, request_id=request.state.request_id)
    except LookupError:
        raise HTTPException(404, "推荐或岗位不存在") from None
    except ExplanationConflict as error:
        raise HTTPException(409, str(error)) from None
    except Exception as error:
        logger.warning(
            json.dumps(
                dict(
                    request_id=request.state.request_id,
                    stage="recommendation_explanation",
                    error_type=type(error).__name__,
                )
            )
        )
        raise HTTPException(503, "理由暂未生成，请稍后重试") from None


@app.get("/explanations.js")
def explanation_javascript():
    return FileResponse(ROOT / "web/explanations.js", media_type="application/javascript")


@app.get("/career.js")
def career_javascript():
    return FileResponse(ROOT / "web/career.js", media_type="application/javascript")


@app.post("/api/resumes")
async def upload_resume(request: Request, user=Depends(current_user), svc=Depends(service)):
    from starlette.concurrency import run_in_threadpool
    from urllib.parse import unquote
    from .resume_extract import MAX_BYTES

    await run_in_threadpool(rate_limit, request, svc, "resume", 6)
    filename = unquote(request.headers.get("x-resume-filename", ""))
    declared = request.headers.get("content-length")
    if declared:
        try:
            length = int(declared)
        except ValueError:
            raise HTTPException(400, "文件长度格式无效") from None
        if length < 0:
            raise HTTPException(400, "文件长度无效")
        if length > MAX_BYTES:
            raise HTTPException(413, "文件不能超过5MB")
    parts, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_BYTES:
            raise HTTPException(413, "文件不能超过5MB")
        parts.append(chunk)
    try:
        return await run_in_threadpool(ResumeStore(svc.store).upload, user["id"], filename, b"".join(parts))
    except ResumeError as error:
        raise HTTPException(422, str(error)) from None
    except RuntimeError:
        raise HTTPException(503, "简历解析繁忙，请稍后重试") from None


@app.get("/api/resumes")
def resume_list(user=Depends(current_user), svc=Depends(service)):
    return ResumeStore(svc.store).list(user["id"])


@app.get("/api/resumes/{resume_id}")
def resume_preview(resume_id: uuid.UUID, user=Depends(current_user), svc=Depends(service)):
    try:
        return ResumeStore(svc.store).preview(user["id"], str(resume_id))
    except LookupError:
        raise HTTPException(404, "简历不存在") from None


@app.post("/api/resumes/{resume_id}/confirm")
def confirm_resume(
    resume_id: uuid.UUID, value: ConfirmResume, user=Depends(current_user), svc=Depends(service)
):
    try:
        return ResumeStore(svc.store).confirm(user["id"], str(resume_id), value)
    except ConflictError as error:
        raise HTTPException(409, str(error)) from None
    except LookupError:
        raise HTTPException(404, "简历不存在") from None


@app.get("/api/resumes/{resume_id}/original")
def resume_original(resume_id: uuid.UUID, user=Depends(current_user), svc=Depends(service)):
    from urllib.parse import quote

    try:
        row = ResumeStore(svc.store).get(user["id"], str(resume_id), include_original=True)
    except LookupError:
        raise HTTPException(404, "简历不存在") from None
    return Response(
        row["original"],
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": "attachment; filename*=UTF-8''" + quote(row["filename"], safe=""),
            "Content-Security-Policy": "sandbox",
        },
    )


@app.delete("/api/resumes/{resume_id}", status_code=204)
def delete_resume(resume_id: uuid.UUID, user=Depends(current_user), svc=Depends(service)):
    try:
        ResumeStore(svc.store).delete(user["id"], str(resume_id))
    except LookupError:
        raise HTTPException(404, "简历不存在") from None
    return Response(status_code=204)


@app.get("/resume.js")
def resume_javascript():
    return FileResponse(ROOT / "web/resume.js", media_type="application/javascript")
