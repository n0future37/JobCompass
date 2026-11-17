"""结构化模型请求：兼容Responses和Chat Completions，不保存或输出密钥。"""

import math
import os


def api_key(config, values=None):
    name = config.get("API_KEY_ENV", "")
    if not name:
        return ""
    key = os.environ.get(name) or (values or {}).get(name)
    if key:
        return key
    # Windows桌面/IDE可能早于用户环境变量启动；仅读取当前用户既有变量。
    if os.name == "nt":
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as environment:
                value, _ = winreg.QueryValueEx(environment, name)
                return value if isinstance(value, str) else ""
        except OSError:
            pass
    return ""


def build_request(config, system, content, max_tokens):
    kind = config.get("API_TYPE") or "openai_responses"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
    if kind == "openai_chat_completions":
        endpoint = "/chat/completions"
        payload = {
            "model": config["MODEL"],
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        thinking = config.get("ENABLE_THINKING", "")
        if thinking != "":
            if str(thinking).lower() not in {"true", "false"}:
                raise ValueError("ENABLE_THINKING只接受true或false")
            payload["enable_thinking"] = str(thinking).lower() == "true"
        temperature = config.get("TEMPERATURE", "")
        if temperature != "":
            temperature = float(temperature)
            if not math.isfinite(temperature) or not 0 <= temperature <= 2:
                raise ValueError("TEMPERATURE需在0至2之间")
            payload["temperature"] = temperature
    elif kind == "openai_responses":
        endpoint = "/responses"
        payload = {
            "model": config["MODEL"],
            "input": messages,
            "max_output_tokens": max_tokens,
            "store": False,
        }
        if config.get("REASONING_EFFORT"):
            payload["reasoning"] = {"effort": config["REASONING_EFFORT"]}
    else:
        raise ValueError("不支持的模型API_TYPE")
    return config["API_BASE"].rstrip("/") + endpoint, payload


def extract_output(response):
    if response.get("status") in {"incomplete", "failed", "cancelled"}:
        raise ValueError("模型响应未完整完成")
    if "choices" in response:
        choices = response["choices"]
        if not choices or choices[0].get("finish_reason") not in {None, "stop"}:
            raise ValueError("模型输出被截断、过滤或未形成文本答案")
        content = choices[0].get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("模型文本输出为空")
        return content
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    return "".join(
        content.get("text", "")
        for item in response.get("output", [])
        for content in item.get("content", [])
        if content.get("type") == "output_text"
    )


def usage_details(response):
    raw = response.get("usage")
    if not isinstance(raw, dict):
        return {}
    usage = dict(raw)
    for target, source in (("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens")):
        if target not in usage and source in usage:
            usage[target] = usage[source]
    if "total_tokens" not in usage and all(
        isinstance(usage.get(k), int) for k in ("input_tokens", "output_tokens")
    ):
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    return usage
