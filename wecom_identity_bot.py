#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
企业微信智能机器人身份解析示例（脱敏公开版）

功能：
1. 使用 Bot ID + Bot Secret 建立企业微信智能机器人长连接。
2. 收到消息后读取 body.from.userid。
3. 使用企业自建应用 access_token 调用官方 userid 转换接口：
   /cgi-bin/batch/openuserid_to_userid
4. 将机器人侧加密 userid 转换为企业内部明文 userid。
5. 使用 /cgi-bin/user/get 获取姓名、部门 ID。
6. 使用 /cgi-bin/department/list 补全部门名称。
7. 将解析后的身份放入统一 context，供工单、告警、PVC 等业务使用。

公开版本不包含任何真实：
- CorpID
- Bot ID
- Secret
- userid
- 姓名
- 部门 ID
- IP / 主机信息

依赖：
    Python >= 3.8
    pip install wecom-aibot-python-sdk aiohttp

环境变量：
    WECHAT_BOT_ID
    WECHAT_BOT_SECRET
    WECOM_CORP_ID
    WECOM_APP_SECRET

可选：
    WECOM_REPLY_IDENTITY=false
    WECOM_DEBUG_IDENTITY=false
    WECOM_LOG_PII=false
"""

import asyncio
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import aiohttp
from aibot import WSClient, WSClientOptions, generate_req_id


BASE_URL = "https://qyapi.weixin.qq.com/cgi-bin"
HTTP_TIMEOUT = aiohttp.ClientTimeout(total=10)
TOKEN_REFRESH_MARGIN = 60
IDENTITY_CACHE_TTL = 3600
DEPARTMENT_CACHE_TTL = 600


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


BOT_ID = (os.getenv("WECHAT_BOT_ID") or "").strip()
BOT_SECRET = (os.getenv("WECHAT_BOT_SECRET") or "").strip()
CORP_ID = (os.getenv("WECOM_CORP_ID") or "").strip()
APP_SECRET = (os.getenv("WECOM_APP_SECRET") or "").strip()

REPLY_IDENTITY = env_bool("WECOM_REPLY_IDENTITY", False)
DEBUG_IDENTITY = env_bool("WECOM_DEBUG_IDENTITY", False)
LOG_PII = env_bool("WECOM_LOG_PII", False)


def validate_config() -> None:
    required = {
        "WECHAT_BOT_ID": BOT_ID,
        "WECHAT_BOT_SECRET": BOT_SECRET,
        "WECOM_CORP_ID": CORP_ID,
        "WECOM_APP_SECRET": APP_SECRET,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError("缺少必要环境变量: " + ", ".join(missing))


def mask_identifier(value: str, keep: int = 3) -> str:
    """日志脱敏。"""
    if not value:
        return ""
    if len(value) <= keep * 2:
        return "***"
    return f"{value[:keep]}***{value[-keep:]}"


@dataclass
class UserIdentity:
    raw_userid: str
    userid: str
    name: str
    department_ids: list[int]
    department_names: list[str]
    resolve_method: str


class WeComIdentityResolver:
    def __init__(self, corp_id: str, secret: str):
        self.corp_id = corp_id
        self.secret = secret

        self._token: Optional[str] = None
        self._token_expire_at = 0.0
        self._token_lock = asyncio.Lock()

        self._identity_cache: dict[str, tuple[float, UserIdentity]] = {}
        self._department_map: dict[int, str] = {}
        self._department_expire_at = 0.0

        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=HTTP_TIMEOUT)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def _get_json(self, path: str, params: dict) -> dict:
        session = await self._get_session()
        async with session.get(BASE_URL + path, params=params) as response:
            response.raise_for_status()
            return await response.json(content_type=None)

    async def _post_json(self, path: str, params: dict, body: dict) -> dict:
        session = await self._get_session()
        async with session.post(
            BASE_URL + path,
            params=params,
            json=body,
        ) as response:
            response.raise_for_status()
            return await response.json(content_type=None)

    async def get_access_token(self, force_refresh: bool = False) -> str:
        async with self._token_lock:
            now = time.time()

            if (
                not force_refresh
                and self._token
                and now < self._token_expire_at - TOKEN_REFRESH_MARGIN
            ):
                return self._token

            data = await self._get_json(
                "/gettoken",
                {
                    "corpid": self.corp_id,
                    "corpsecret": self.secret,
                },
            )

            if data.get("errcode", 0) != 0 or not data.get("access_token"):
                raise RuntimeError(
                    "获取 access_token 失败: "
                    f"errcode={data.get('errcode')}, "
                    f"errmsg={data.get('errmsg')}"
                )

            self._token = data["access_token"]
            self._token_expire_at = now + int(data.get("expires_in", 7200))
            return self._token

    async def _with_token_retry(self, method):
        """
        access_token 失效时只刷新并重试一次。
        method(token) -> dict
        """
        token = await self.get_access_token()
        data = await method(token)

        if data.get("errcode") in {40014, 42001}:
            token = await self.get_access_token(force_refresh=True)
            data = await method(token)

        return data

    async def convert_open_userid(self, open_userid: str) -> Optional[str]:
        async def request(token: str) -> dict:
            return await self._post_json(
                "/batch/openuserid_to_userid",
                {"access_token": token},
                {"open_userid_list": [open_userid]},
            )

        data = await self._with_token_retry(request)

        if data.get("errcode", 0) != 0:
            raise RuntimeError(
                "userid 转换失败: "
                f"errcode={data.get('errcode')}, "
                f"errmsg={data.get('errmsg')}"
            )

        for item in data.get("userid_list", []):
            if item.get("open_userid") == open_userid and item.get("userid"):
                return str(item["userid"]).strip()

        # 明文 userid 或无效 ID 可能不会产生映射。
        return None

    async def get_user(self, userid: str) -> Optional[dict]:
        async def request(token: str) -> dict:
            return await self._get_json(
                "/user/get",
                {
                    "access_token": token,
                    "userid": userid,
                },
            )

        data = await self._with_token_retry(request)

        if data.get("errcode", 0) != 0:
            return None

        return data

    async def get_department_map(self) -> dict[int, str]:
        now = time.time()

        if (
            self._department_map
            and now < self._department_expire_at
        ):
            return self._department_map

        async def request(token: str) -> dict:
            return await self._get_json(
                "/department/list",
                {"access_token": token},
            )

        data = await self._with_token_retry(request)

        if data.get("errcode", 0) != 0:
            raise RuntimeError(
                "获取部门列表失败: "
                f"errcode={data.get('errcode')}, "
                f"errmsg={data.get('errmsg')}"
            )

        department_map: dict[int, str] = {}

        for department in data.get("department", []):
            dept_id = department.get("id")
            if dept_id is None:
                continue
            try:
                dept_id = int(dept_id)
            except (TypeError, ValueError):
                continue

            department_map[dept_id] = str(
                department.get("name") or ""
            ).strip()

        self._department_map = department_map
        self._department_expire_at = now + DEPARTMENT_CACHE_TTL
        return department_map

    async def resolve(self, raw_userid: str) -> UserIdentity:
        if not raw_userid:
            raise RuntimeError("Bot 消息中不存在 from.userid")

        cached = self._identity_cache.get(raw_userid)
        if cached:
            expire_at, identity = cached
            if time.time() < expire_at:
                return identity
            self._identity_cache.pop(raw_userid, None)

        # 先按官方 Bot -> 自建应用转换流程尝试转换。
        corp_userid = await self.convert_open_userid(raw_userid)

        if corp_userid:
            resolve_method = "openuserid_to_userid"
        else:
            # 如果没有转换结果，则按明文 userid 处理。
            corp_userid = raw_userid
            resolve_method = "plain_userid"

        user = await self.get_user(corp_userid)
        if not user:
            raise RuntimeError(
                "身份已解析，但无法通过 user/get 获取成员信息"
            )

        department_ids: list[int] = []
        for dept_id in user.get("department", []):
            try:
                department_ids.append(int(dept_id))
            except (TypeError, ValueError):
                continue

        department_names: list[str] = []

        try:
            department_map = await self.get_department_map()
            department_names = [
                department_map.get(dept_id, f"UNKNOWN({dept_id})")
                for dept_id in department_ids
            ]
        except Exception as exc:
            print(f"[WARN] 部门名称解析失败: {exc}", file=sys.stderr)

        identity = UserIdentity(
            raw_userid=raw_userid,
            userid=str(user.get("userid") or corp_userid).strip(),
            name=str(user.get("name") or "").strip(),
            department_ids=department_ids,
            department_names=department_names,
            resolve_method=resolve_method,
        )

        self._identity_cache[raw_userid] = (
            time.time() + IDENTITY_CACHE_TTL,
            identity,
        )
        return identity


validate_config()

resolver = WeComIdentityResolver(
    corp_id=CORP_ID,
    secret=APP_SECRET,
)

ws_client = WSClient(
    WSClientOptions(
        bot_id=BOT_ID,
        secret=BOT_SECRET,
        max_reconnect_attempts=-1,
    )
)


@ws_client.on("connected")
def on_connected():
    print("[BOT] WebSocket 已连接")


@ws_client.on("authenticated")
def on_authenticated():
    print("[BOT] 机器人认证成功")


@ws_client.on("disconnected")
def on_disconnected(reason):
    print(f"[BOT] 连接断开: {reason}")


@ws_client.on("reconnecting")
def on_reconnecting(attempt):
    print(f"[BOT] 正在重连: attempt={attempt}")


@ws_client.on("error")
def on_error(error):
    print(f"[BOT ERROR] {error}", file=sys.stderr)


async def handle_business_message(context: dict) -> None:
    """
    业务接入点。

    后续工单、PVC、告警等逻辑统一使用：
        context["userid"]
        context["name"]
        context["department_names"]

    不再直接使用：
        frame["body"]["from"]["userid"]
    """
    if LOG_PII:
        print(
            "[MESSAGE] "
            f"name={context['name']}, "
            f"userid={mask_identifier(context['userid'])}, "
            f"method={context['resolve_method']}, "
            f"msgtype={context['msgtype']}"
        )
    else:
        print(
            "[MESSAGE] "
            f"identity_resolved={bool(context['userid'])}, "
            f"method={context['resolve_method']}, "
            f"msgtype={context['msgtype']}"
        )

    # 示例：
    #
    # await create_ticket(
    #     userid=context["userid"],
    #     username=context["name"],
    #     department_names=context["department_names"],
    #     message=context["text"],
    # )


@ws_client.on("message")
async def on_message(frame):
    body = frame.get("body", {}) or {}
    sender = body.get("from", {}) or {}

    raw_userid = str(sender.get("userid") or "").strip()
    msgtype = str(body.get("msgtype") or "").strip()

    text = ""
    if msgtype == "text":
        text = str((body.get("text") or {}).get("content") or "")

    try:
        identity = await resolver.resolve(raw_userid)
    except Exception as exc:
        print(
            "[IDENTITY ERROR] "
            f"userid={mask_identifier(raw_userid)} "
            f"error={exc}",
            file=sys.stderr,
        )
        identity = UserIdentity(
            raw_userid=raw_userid,
            userid=raw_userid,
            name="",
            department_ids=[],
            department_names=[],
            resolve_method="failed",
        )

    context = {
        "raw_userid": identity.raw_userid,
        "userid": identity.userid,
        "name": identity.name,
        "department_ids": identity.department_ids,
        "department_names": identity.department_names,
        "resolve_method": identity.resolve_method,
        "msgtype": msgtype,
        "text": text,
        "frame": frame,
    }

    if DEBUG_IDENTITY:
        print(
            "[DEBUG] "
            f"raw_userid={mask_identifier(identity.raw_userid)}, "
            f"userid={mask_identifier(identity.userid)}, "
            f"resolve_method={identity.resolve_method}"
        )

    await handle_business_message(context)

    # 仅用于测试。正式环境请保持 false。
    if REPLY_IDENTITY and msgtype == "text":
        stream_id = generate_req_id("identity-test")

        name = identity.name or "未知用户"
        department = ", ".join(identity.department_names) or "未知"

        content = (
            "身份解析成功\n\n"
            f"姓名：{name}\n"
            f"部门：{department}"
        )

        try:
            await ws_client.reply_stream(
                frame,
                stream_id,
                content,
                True,
            )
        except Exception as exc:
            print(f"[WARN] 测试回复失败: {exc}", file=sys.stderr)


def main() -> int:
    print("=== WeCom Bot Identity Resolver ===")
    print("Bot Secret      : loaded")
    print("App Secret      : loaded")
    print(f"Reply identity  : {REPLY_IDENTITY}")
    print(f"Debug identity  : {DEBUG_IDENTITY}")
    print(f"Log PII         : {LOG_PII}")

    try:
        ws_client.run()
    except KeyboardInterrupt:
        print("\n服务已停止")
    except Exception as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
