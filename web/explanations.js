(() => {
  "use strict";
  const jc = window.JC;
  const { $, state, node, action, api, jsonOptions } = jc;
  const strategy = $("ranking-strategy");
  if (strategy) strategy.addEventListener("change", () => { $("ranking-egress-note").hidden = strategy.value === "rules"; });

  jc.explanationControls = (job, recommendation, target, queryId) => {
    if (!recommendation.explanation_status) return;
    if (recommendation.explanation_status === "fallback") target.append(node("p", "模型解释未完成，以下为本地原文依据。", "notice"));
    if (recommendation.citations?.length) {
      target.append(node("p", "依据这次推荐时保存的查询与画像。", "small muted"));
      return;
    }
    if (!queryId) { target.append(node("p", "请重新推荐并保存记录后生成理由。", "small muted")); return; }
    const status = node("p", "点击后生成理由和原文引用。", "small muted"); status.setAttribute("aria-live", "polite");
    const owner = state.user?.id;
    const endpoint = `/api/recommendations/${encodeURIComponent(queryId)}/explanation`;
    const sameOwner = () => state.user?.id === owner;
    const button = action("生成推荐理由", async () => {
      button.disabled = true; button.textContent = "正在生成…"; status.textContent = "正在核对岗位原文…";
      try {
        let value = await api(endpoint, jsonOptions({ job_key: job.job_key }));
        for (let i = 0; value.status === "running" && i < 45; i++) {
          if (!sameOwner() || !$("job-dialog").open) return;
          await new Promise(resolve => setTimeout(resolve, 2000));
          value = await api(endpoint + "?job_key=" + encodeURIComponent(job.job_key));
        }
        if (!sameOwner()) return;
        if (value.status !== "completed") throw new Error("理由仍在处理，请稍后重新读取。");
        if (!value.recommendation) { status.textContent = value.message || "当前证据不足，请调整方向重新推荐。"; return; }
        Object.assign(recommendation, value.recommendation);
        // 不重绘整页，保留反馈输入；旧查询/旧账号的异步结果不能覆盖当前页面。
        if (state.latest?.query_id === queryId) {
          const row = state.latest.results.find(r => r.job.job_key === job.job_key);
          for (const card of document.querySelectorAll(".job-card")) {
            if (row && card.dataset.jobKey === job.job_key && card.dataset.queryId === queryId) {
              card.replaceWith(jc.jobCard(row.job, row, recommendation, queryId));
            }
          }
        }
        const dialog = $("job-dialog");
        if (dialog.open && dialog.dataset.jobKey === job.job_key && dialog.dataset.queryId === queryId) jc.showJob(job, recommendation, queryId);
      } catch (error) {
        if (sameOwner()) { status.textContent = error.message; status.className = "notice"; }
      } finally {
        button.disabled = false; button.textContent = "生成 / 读取理由";
      }
    }, "button secondary");
    target.append(status, button);
  };
})();
