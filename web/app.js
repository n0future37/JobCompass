(() => {
  "use strict";
  const $ = id => document.getElementById(id);
  const state = { user: null, latest: null, offset: 0, pageSize: 6, total: 0, displayVersion: 0, view: "explore" };
  const paths = {
    compass: "M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20ZM16 8l-2.5 5.5L8 16l2.5-5.5L16 8Z",
    search: "M10.5 3a7.5 7.5 0 1 0 0 15 7.5 7.5 0 0 0 0-15ZM16 16l5 5",
    clock: "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18Zm0 4v5l3 2",
    user: "M12 3a4 4 0 1 0 0 8 4 4 0 0 0 0-8ZM4 21v-2a7 7 0 0 1 7-5h2a7 7 0 0 1 7 5v2",
    upload: "M12 16V3m-5 5 5-5 5 5M4 15v5a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-5",
    quote: "M4 5h6v8H5c-1 3-1 4 3 5M14 5h6v8h-5c-1 3-1 4 3 5",
    chevron: "m7 10 5 5 5-5", filter: "M4 6h16M7 12h10M10 18h4",
    refresh: "M20 7v5h-5M4 17v-5h5M5.5 7a7.5 7.5 0 0 1 12.8-2L20 7M4 17l1.7 2A7.5 7.5 0 0 0 18.5 17",
    check: "m5 12 4 4L19 6", close: "m6 6 12 12M18 6 6 18", arrow: "M4 12h15m-6-6 6 6-6 6",
    info: "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18Zm0 8v6m0-10v1",
    shield: "M12 2 3 6v6c0 5 9 10 9 10s9-5 9-10V6l-9-4Zm-4 10 3 3 5-6",
    external: "M14 3h7v7M21 3l-11 11M10 4H4v16h16v-6"
  };
  function icon(name) {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    for (const [key, value] of Object.entries({ viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", "stroke-width": "1.7", "stroke-linecap": "round", "stroke-linejoin": "round", "aria-hidden": "true" })) svg.setAttribute(key, value);
    const path = document.createElementNS(svg.namespaceURI, "path");
    path.setAttribute("d", paths[name] || paths.info); svg.append(path); return svg;
  }
  function node(tag, text, cls) {
    const element = document.createElement(tag);
    if (text !== undefined) element.textContent = text;
    if (cls) element.className = cls;
    return element;
  }
  function action(text, callback, cls = "text-button", symbol) {
    const button = node("button", text, cls); button.type = "button";
    if (symbol) button.append(icon(symbol)); button.addEventListener("click", callback); return button;
  }
  function empty(target, title, description, symbol = "search") {
    const box = node("div", undefined, "empty-state"); box.append(icon(symbol), node("h3", title), node("p", description));
    target.replaceChildren(box);
  }
  let toastTimer;
  function notify(message) {
    clearTimeout(toastTimer); $("toast").textContent = message; $("toast").hidden = false;
    toastTimer = setTimeout(() => { $("toast").hidden = true; }, 4500);
  }
  async function api(url, options = {}) {
    let response;
    try { response = await fetch(url, options); } catch { throw new Error("暂时无法连接服务，请检查服务状态后重试。"); }
    if (response.status === 204) return null;
    let data;
    try { data = await response.json(); } catch { throw new Error("服务返回内容异常，请稍后重试。"); }
    if (!response.ok) {
      const message = response.status === 401 ? "请先登录，再保存或检索你的岗位。" :
        response.status === 429 ? "操作比较频繁，请稍后再试。" :
        typeof data.detail === "string" ? data.detail : "输入格式不正确，请检查必填字段和长度。";
      const error = new Error(message); error.status = response.status; throw error;
    }
    return data;
  }
  const jsonOptions = value => ({ method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(value) });
  const split = value => value.split(/[,，、;；]/).map(s => s.trim()).filter(Boolean);
  const dates = new Intl.DateTimeFormat("zh-CN", { timeZone: "Asia/Shanghai", year: "numeric", month: "2-digit", day: "2-digit" });
  function date(value) { const parsed = new Date(value); return Number.isNaN(parsed.getTime()) ? "日期未知" : dates.format(parsed); }
  function title(text, count, suffix) {
    const badge = node("span", `${new Intl.NumberFormat("zh-CN").format(count)} ${suffix}`, "count-label"); badge.id = "corpus-count";
    $("results-title").replaceChildren(document.createTextNode(text + " "), badge);
  }
  function sourceName(job) { return ({ zhilian: "智联招聘", shixiseng: "实习僧" })[job.source] || "导入岗位"; }
  function sourceLink(job) {
    if (!/^https?:\/\//i.test(job.url || "")) return null;
    const link = node("a", "查看来源网站"); link.href = job.url; link.target = "_blank"; link.rel = "noopener noreferrer"; return link;
  }
  function showJob(job, recommendation, queryId = state.latest?.query_id) {
    $("job-dialog").dataset.jobKey = job.job_key; $("job-dialog").dataset.queryId = queryId || "";
    $("detail-title").textContent = job.title; const body = $("job-detail"); body.replaceChildren();
    body.append(node("p", [job.company || "公司未知", (job.cities || []).join(" / ") || "城市未知", job.salary_text || "薪资未知"].join("　"), "detail-meta"));
    if (recommendation) {
      const evidence = node("section", undefined, "detail-section"); evidence.append(node("h3", "工作内容依据"));
      if (recommendation.match_reason) evidence.append(node("p", recommendation.match_reason, "small muted"));
      for (const citation of recommendation.citations || []) {
        const quote = node("blockquote", undefined, "detail-quote"); quote.append(node("span", `JD 第 ${citation.line} 行`, "quote-number"), document.createTextNode(citation.quote)); evidence.append(quote);
      }
      window.JC.explanationControls?.(job, recommendation, evidence, queryId);
      body.append(evidence);
      const analysis = node("details", undefined, "detail-analysis"); analysis.append(node("summary", "匹配分析"));
      for (const check of recommendation.background_checks || []) {
        const field = check.field === "education" ? "学历" : "经验";
        const status = { pass: "下限可匹配", unknown: "待确认", fail: "不满足下限" }[check.status];
        analysis.append(node("p", `${field}：岗位 ${check.requirement ?? "未知"} / 自述 ${check.user_value ?? "未填写"} / ${status}`, "small muted"));
      }
      const checks = recommendation.hard_condition_checks || [];
      if (checks.length) analysis.append(node("p", "本轮条件：" + checks.map(c => c.field === "education" ? (c.unknown ? "学历待确认" : "学历下限符合") : c.term ? `JD 提及 ${c.term}` : ({cities:"城市符合",employment_type:"招聘类型符合",excluded_cities:"符合城市排除条件",excluded_employment_types:"符合招聘类型排除条件"}[c.field] || "条件符合")).join("；"), "small muted"));
      if (recommendation.profile_mentions?.length) analysis.append(node("p", "画像与 JD 均提及：" + recommendation.profile_mentions.join("、"), "small muted"));
      if (recommendation.profile_evidence) analysis.append(node("p", "画像原文：" + recommendation.profile_evidence, "small muted"));
      if (recommendation.skills_without_profile_evidence?.length) analysis.append(node("p", "画像未提供依据的技能：" + recommendation.skills_without_profile_evidence.join("、") + "。未提及不等于不会。", "small muted"));
      if (analysis.children.length > 1) body.append(analysis);
    }
    window.JC.careerControls?.(job, body);
    const section = node("section", undefined, "detail-section"); section.append(node("h3", "岗位正文"));
    const text = node("div", undefined, "jd-lines"); text.tabIndex = 0; text.setAttribute("aria-label", "岗位正文"); const cited = new Set((recommendation?.citations || []).map(c => c.line));
    job.jd_text.split("\n").forEach((line, i) => text.append(node("div", line, "jd-line" + (cited.has(i + 1) ? " cited" : ""))));
    section.append(text); body.append(section);
    const links = node("div", undefined, "detail-links"); const external = sourceLink(job); if (external) links.append(external);
    const original = job.provenance?.[0];
    if (original) { const link = node("a", "原始记录"); link.href = "/api/raw?path=" + encodeURIComponent(original.raw_file); link.target = "_blank"; link.rel = "noopener noreferrer"; links.append(link); }
    body.append(links, node("p", `${sourceName(job)} · 采集于 ${date(job.collected_at)}`, "footer-note"));
    if (!$("job-dialog").open) $("job-dialog").showModal();
  }
  function jobCard(job, result, recommendation, queryId) {
    const article = node("article", undefined, "job-card"); article.dataset.jobKey = job.job_key; article.dataset.queryId = queryId || "";
    const company = node("div", undefined, "company-row"); company.append(node("span", (job.company || "岗位").slice(0, 1), "company-avatar"), node("span", job.company || "公司未知", "company-name")); article.append(company);
    const head = node("div", undefined, "job-head"); const heading = node("h3"); heading.append(action(job.title, () => showJob(job, recommendation, queryId), "job-title")); head.append(heading, node("span", job.salary_text || "薪资未知", "salary")); article.append(head);
    const meta = node("div", undefined, "job-meta"); for (const value of [(job.cities || []).join(" / ") || "城市未知", job.experience_text || "经验未知", job.education_text || "学历未知", job.employment_type_text].filter(Boolean)) meta.append(node("span", value)); article.append(meta);
    if (recommendation?.citations?.length) {
      const block = node("div", undefined, "evidence"); const caption = node("p", "原文依据", "evidence-caption"); caption.prepend(icon("quote"));
      block.append(caption, node("blockquote", recommendation.citations[0].quote)); article.append(block);
    } else article.append(node("p", job.jd_text.replace(/\n/g, " "), "jd-preview"));
    if (result?.personal_signals) {
      const signal = result.personal_signals; const chips = node("div", undefined, "chips");
      const skills = [...new Set([...(signal.skills || []), ...(signal.project_skills || [])])].slice(0, 3);
      for (const skill of skills) chips.append(node("span", skill, "chip"));
      if (signal.preference === "like") chips.append(node("span", "已喜欢", "chip"));
      const statuses = { not_applied: "未投递", applied: "已投递", interviewing: "面试中", offer: "Offer", rejected: "被拒绝" };
      if (signal.application_status) chips.append(node("span", statuses[signal.application_status], "chip"));
      if (result.background_checks?.some(c => c.status === "unknown")) chips.append(node("span", "背景待确认", "chip"));
      if (chips.children.length) article.append(chips);
    }
    const footer = node("div", undefined, "job-footer"); footer.append(node("span", `${sourceName(job)} · ${date(job.collected_at)}`), action("查看详情", () => showJob(job, recommendation, queryId), "text-button", "arrow")); article.append(footer); return article;
  }
  async function browse(offset = 0) {
    const version = ++state.displayVersion; state.latest = null; state.offset = offset;
    $("feedback").hidden = true; $("message").textContent = "正在读取岗位库…"; $("results").setAttribute("aria-busy", "true");
    try {
      const data = await api(`/api/jobs?offset=${offset}&limit=${state.pageSize}`); if (version !== state.displayVersion) return;
      state.total = data.total; title("岗位库", data.total, "个历史岗位"); $("results-subtitle").textContent = "";
      $("message").replaceChildren(); $("results").replaceChildren(...data.items.map(job => jobCard(job)));
      if (!data.items.length) empty($("results"), "岗位库暂时没有内容", "可以从“数据导入”上传已有的岗位文件。");
      $("pagination").hidden = data.total <= state.pageSize; $("previous-page").disabled = offset === 0; $("next-page").disabled = offset + state.pageSize >= data.total;
      $("page-label").textContent = `第 ${Math.floor(offset / state.pageSize) + 1} / ${Math.max(1, Math.ceil(data.total / state.pageSize))} 页`;
    } catch (error) { if (version === state.displayVersion) { $("message").textContent = error.message; $("pagination").hidden = true; } }
    finally { if (version === state.displayVersion) $("results").setAttribute("aria-busy", "false"); }
  }
  function render(data, historical = false) {
    ++state.displayVersion; state.latest = data; $("pagination").hidden = true; $("message").replaceChildren(); $("results").setAttribute("aria-busy", "false");
    title(historical ? "历史结果" : data.result_kind === "filtered_only" ? "条件筛选结果" : data.agent ? "推荐结果" : "搜索结果", data.results.length, "个历史岗位");
    $("results-subtitle").textContent = data.request.query;
    if (historical) $("message").append(node("p", "历史结果，以保存时状态为准。", "small muted"));
    if (data.generation?.mode === "extractive_fallback") $("message").append(node("p", "解释已降级为原文摘录。", "notice"));
    if (data.ranking?.mode === "fallback_rules") $("message").append(node("p", "模型排序暂不可用，已使用规则排序。", "notice"));
    else if (data.degradations?.length) $("message").append(node("p", "部分处理功能暂不可用，结果已按降级规则处理。", "notice"));
    if (data.unresolved_conditions?.length) $("message").append(node("p", "暂不支持这些条件：" + data.unresolved_conditions.join("；"), "notice"));
    const info = node("details", undefined, "query-details"); info.append(node("summary", "查询详情"));
    info.append(node("p", `检索与证据处理 ${(data.total_ms / 1000).toFixed(2)} 秒，不含网页传输。`, "small muted"));
    if (data.agent) info.append(node("p", data.agent.background_policy === "allow_unknown" ? "允许展示背景待确认的候选。" : "仅保留可核验学历、经验下限的候选。", "small muted"));
    if (data.ranking) info.append(node("p", "排序：" + (data.ranking.mode === "llm" ? "大模型" : data.ranking.mode === "fallback_rules" ? "规则回退" : "规则 / 无需重排"), "small muted"));
    if (data.results.length) $("message").append(info);
    const recommendations = new Map((data.recommendations || []).map(row => [row.job_key, row]));
    $("results").replaceChildren(...data.results.map(row => jobCard(row.job, row, recommendations.get(row.job.job_key), data.query_id)));
    if (!data.results.length) empty($("results"), "暂无匹配岗位", data.no_result_reason || data.answer || "调整条件后重试。");
    $("feedback").hidden = !data.query_id; $("feedback-status").textContent = ""; $("note").value = "";
  }
  function setBusy(busy) {
    $("submit").disabled = busy; $("recommend").disabled = busy; $("submit-label").textContent = busy ? "检索中…" : "搜索岗位";
    $("results").classList.toggle("is-loading", busy); $("results").setAttribute("aria-busy", String(busy));
  }
  async function search(event, useAgent = false) {
    event?.preventDefault(); if (!$("search").reportValidity()) return;
    if (!state.user) { window.JC.openAuth("登录后即可保存这次查询及岗位依据。"); return; }
    const version = ++state.displayVersion; setBusy(true); $("message").textContent = useAgent ? "正在检索并排序，理由可稍后生成…" : "正在检索…";
    const value = { query: $("query").value, cities: split($("cities").value), skills_all: split($("skills").value), employment_types: $("employment").value ? [$("employment").value] : [],
      profile: $("use-profile").checked ? state.user.profile?.text || "" : "", mode: $("mode").value, rerank: $("rerank").checked, generate: $("generate").checked, top_k: 5 };
    if (useAgent) { value.allow_unknown_background = $("allow-unknown-background").checked; value.include_closed = $("include-closed").checked; value.profile = ""; value.generate = false; value.ranking_strategy = $("ranking-strategy").value; }
    try { const data = await api(useAgent ? "/api/recommendations" : "/api/search", jsonOptions(value)); if (version === state.displayVersion) render(data); }
    catch (error) { if (version === state.displayVersion) { $("message").replaceChildren(node("p", error.message, "notice")); if (error.status === 401) window.JC.openAuth(); } }
    finally { setBusy(false); }
  }
  async function loadHistory() {
    if (!state.user) { empty($("history-list"), "登录后查看你的探索记录", "每次检索的岗位和证据会保存在你的账号下。", "clock"); return; }
    $("history-list").textContent = "正在读取查询历史…";
    try {
      const rows = await api("/api/history"); $("history-list").replaceChildren();
      if (!rows.length) { empty($("history-list"), "还没有查询记录", "从“岗位探索”开始，你的下一次查询会出现在这里。", "clock"); return; }
      for (const row of rows) {
        const value = row.payload.request; const item = action("", () => { $("query").value = value.query; $("cities").value = (value.cities || []).join("、"); $("skills").value = (value.skills_all || []).join("、"); $("employment").value = value.employment_types?.[0] || ""; $("mode").value = value.mode || "hybrid"; $("rerank").checked = Boolean(value.rerank); updateFilters(); render({ ...row.payload, query_id: row.query_id }, true); navigate("explore"); }, "history-item");
        const text = node("div"); text.append(node("strong", value.query), node("p", `${date(row.created_at)}　${row.payload.results?.length || 0} 条岗位候选`)); item.append(text, icon("arrow")); $("history-list").append(item);
      }
    } catch (error) { $("history-list").textContent = error.message; }
  }
  function updateFilters() {
    const count = [$("cities").value.trim(), $("skills").value.trim(), $("employment").value].filter(Boolean).length;
    $("filter-count").textContent = String(count); $("filter-count").hidden = !count;
  }
  function navigate(view) { if (location.hash !== "#" + view) location.hash = view; else showView(); }
  function showView() {
    const labels = { explore: "岗位探索", history: "查询历史", profile: "个人画像", imports: "数据导入", applications: "投递记录" };
    const selected = location.hash.slice(1); const view = selected in labels ? selected : "explore"; const changed = state.view !== view; state.view = view;
    document.querySelectorAll("[data-page]").forEach(page => { page.hidden = page.dataset.page !== view; });
    document.querySelectorAll("[data-view]").forEach(link => { if (link.dataset.view === view) link.setAttribute("aria-current", "page"); else link.removeAttribute("aria-current"); });
    if (changed) window.scrollTo(0, 0);
    $("view-label").textContent = labels[view]; document.title = `${labels[view]} · JobCompass`;
    if (view === "history") loadHistory(); window.dispatchEvent(new CustomEvent("jc:view", { detail: view }));
  }
  window.JC = { $, state, api, jsonOptions, node, icon, action, empty, notify, navigate, showView, render, browse, loadHistory, showJob, jobCard };
  document.querySelectorAll("[data-icon]").forEach(element => element.append(icon(element.dataset.icon)));
  document.querySelectorAll("[data-close]").forEach(button => button.addEventListener("click", () => $(button.dataset.close).close()));
  document.querySelectorAll("dialog").forEach(dialog => dialog.addEventListener("click", event => { if (event.target === dialog) { const rect = dialog.getBoundingClientRect(); if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close(); } }));
  $("search").addEventListener("submit", search);
  $("recommend").addEventListener("click", event => search(event, true));
  $("query").addEventListener("keydown", event => { if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) { event.preventDefault(); $("search").requestSubmit(); } });
  $("toggle-filters").addEventListener("click", () => { const open = $("filter-panel").hidden; $("filter-panel").hidden = !open; $("toggle-filters").setAttribute("aria-expanded", String(open)); });
  $("generate").addEventListener("change", () => { $("egress-note").hidden = !$("generate").checked; });
  for (const name of ["cities", "skills", "employment"]) $(name).addEventListener("input", updateFilters);
  $("clear-filters").addEventListener("click", () => { for (const name of ["cities", "skills", "employment"]) $(name).value = ""; updateFilters(); notify("已清空本轮硬条件"); });
  document.querySelectorAll("[data-query]").forEach(button => button.addEventListener("click", () => { $("query").value = button.dataset.query; $("query").focus(); }));
  $("browse-jobs").addEventListener("click", () => browse()); $("previous-page").addEventListener("click", () => browse(Math.max(0, state.offset - state.pageSize))); $("next-page").addEventListener("click", () => browse(state.offset + state.pageSize));
  $("load-history").addEventListener("click", loadHistory);
  document.querySelectorAll("[data-useful]").forEach(button => button.addEventListener("click", async () => {
    if (!state.latest?.query_id) return; const controls = [...document.querySelectorAll("[data-useful]")]; controls.forEach(control => { control.disabled = true; });
    try { await api("/api/feedback", jsonOptions({ query_id: state.latest.query_id, useful: button.dataset.useful === "true", note: $("note").value })); $("feedback-status").textContent = "已保存。"; notify("反馈已保存"); }
    catch (error) { $("feedback-status").textContent = error.message; } finally { controls.forEach(control => { control.disabled = false; }); }
  }));
  window.addEventListener("hashchange", showView); showView(); browse();
  api("/api/health").then(result => { const ready = Object.values(result).every(value => value === "ok"); $("health").textContent = ready ? "服务就绪" : "部分服务未就绪"; $("health").dataset.ready = String(ready); }).catch(() => { $("health").textContent = "连接暂不可用"; });
})();
