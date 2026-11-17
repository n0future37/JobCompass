"""真实本机关闭端口：确保Redis隐式退避不会拖延业务降级。"""

import socket
from time import perf_counter

import pytest
import redis

from jobcompass.service import cache_client


@pytest.mark.integration
def test_redis_connection_failure_has_no_hidden_retry(real_store):
    with socket.socket() as closed:
        closed.bind(("127.0.0.1", 0))
        client = cache_client(real_store.settings, port=closed.getsockname()[1])
        started = perf_counter()
        try:
            with pytest.raises((redis.ConnectionError, redis.TimeoutError)):
                client.ping()
            # 本机拒绝连接常见约2秒；预留系统调度余量，旧SDK默认重试实测26秒。
            assert perf_counter() - started < 8
        finally:
            client.close()
