# Milvus 启动、停机与恢复

项目提供独立 Milvus 与 etcd 的部署模板：`deploy/milvus/compose.yaml`。它用于新建 JobCompass 专属实例，默认仅向宿主机回环地址开放19540端口，数据卷与其他Compose项目隔离。应用侧沿用现有MySQL、Redis和独立worker。

## 选择独立etcd的原因

本项目使用的Milvus3.0.3镜像在嵌入etcd模式下，收到SIGTERM后出现`kv.CloseEtcdClient`的`context canceled` panic。嵌入客户端与普通gRPC客户端的关闭路径不同；Milvus对关闭返回的错误直接panic。延长停机宽限时间不能消除该错误。

配置使用独立etcd3.5.25，关闭嵌入模式；`depends_on`要求etcd健康后启动Milvus，也让Compose停止整个项目时先停Milvus、后停etcd。Milvus明确使用SIGTERM和60秒宽限。没有把退出码134包装成0，也没有屏蔽panic。

依据：[Milvus关闭客户端实现](https://github.com/milvus-io/milvus/blob/v3.0.3/internal/util/dependency/kv/kv_client_handler.go)、[嵌入客户端创建](https://github.com/milvus-io/milvus/blob/v3.0.3/pkg/util/etcd/etcd_server.go)、[etcd客户端Close](https://github.com/etcd-io/etcd/blob/v3.5.33/client/v3/client.go)。此客户端版本来自Milvus源码依赖；独立服务镜像使用etcd3.5.25。

## 新实例运行

已有服务可继续使用，不需要启动第二套。使用新模板时，在`deploy/milvus`目录复制`.env.example`为不入Git的`.env.local`，通过本机安全配置填写新实例root密码。不要复用公共服务或其他项目密码。

```powershell
docker compose --env-file .env.local up -d
docker compose --env-file .env.local ps
docker compose --env-file .env.local stop
```

镜像源不可达时，可在同一私有配置中指定`MILVUS_IMAGE`与`ETCD_IMAGE`，保留版本和镜像digest记录。通过sudo运行时也使用`--env-file`，不要依赖可能被sudo过滤的进程环境变量。

初始化项目自己的dev/test账号和数据库后，业务进程使用受限账号。宿主机应用连接`http://127.0.0.1:19540`；Docker中的API连接`host.docker.internal:19540`。root只用于管理，不放进业务账号配置。

关闭API和worker、等待在途操作结束后，再停止此向量服务项目。不要先单独停止etcd。短暂的HTTP健康检查通过只表示组件启动，集合恢复仍可能暂时返回503；恢复探针应在有界等待内执行实际Strong计数、元数据和向量查询，不把零条结果当作就绪。

## 数据与验证边界

模板默认创建新卷。不能将原嵌入式Milvus数据卷直接挂到一个空的独立etcd上并当作迁移完成；元数据和向量文件需要配套恢复。旧实验卷及日志作为对照保留。已有本机共用服务配置不由本模板修改。

停机回归同时检查：收到SIGTERM后退出0、没有panic或OOM、重启后有效条数和块身份/版本/指纹不变、向量查询有效，以及Compose先停Milvus后停etcd。服务版本、镜像digest、数据规模、停机宽限及每轮结果需随独立验收记录保存。这些条件不证明无限次重启或生产级高可用。
