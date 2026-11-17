(() => {
  "use strict";
  const { $, state, api, jsonOptions, notify, navigate } = window.JC;
  let authMode = "login", dirtyProfile = false, importTask = null, pollTimer = null, pollCount = 0;
  const statusText = { queued: "等待处理", running: "正在处理岗位", retrying: "正在等待重试", completed: "导入已完成", failed: "导入未完成" };
  function stopPolling() { clearTimeout(pollTimer); pollTimer = null; }
  function setAuthMode(mode) {
    authMode = mode; const registering = mode === "register";
    $("auth-title").textContent = registering ? "创建你的本地账号" : "登录 JobCompass";
    $("auth-submit").textContent = registering ? "注册并登录" : "登录";
    $("auth-switch-label").textContent = registering ? "已经有账号？" : "还没有账号？";
    $("auth-switch").textContent = registering ? "返回登录" : "注册本地账号";
    $("password").autocomplete = registering ? "new-password" : "current-password"; $("auth-error").textContent = "";
  }
  function openAuth(description) {
    $("auth-description").textContent = description || "登录后保存画像与查询。";
    $("account-menu").hidden = true; setAuthMode("login");
    if (!$("auth-dialog").open) $("auth-dialog").showModal(); $("username").focus();
  }
  window.JC.openAuth = openAuth;
  async function refreshAccount() {
    try {
      const user = await api("/api/me"); const changed = state.user?.id !== user.id; const switchingAccount = changed && Boolean(state.user); if (changed && state.user) { dirtyProfile = false; $("profile-text").value = ""; $("history-list").replaceChildren(); window.JC.browse(); } state.user = user;
      $("account-name").textContent = user.username; $("account-avatar").textContent = user.username.slice(0, 1).toUpperCase();
      $("account-status").textContent = `已登录：${user.username}`; $("account-trigger").removeAttribute("aria-haspopup"); $("account-trigger").setAttribute("aria-controls", "account-menu"); $("account-trigger").setAttribute("aria-expanded", "false");
      if (!dirtyProfile) $("profile-text").value = user.profile?.text || ""; $("profile-status").textContent = "已保存。";
      if (changed) { if (switchingAccount) { $("import-text-panel").reset(); selectImportMode("file"); } window.dispatchEvent(new CustomEvent("jc:account")); }
      if (changed) { stopPolling(); pollCount = 0; importTask = sessionStorage.getItem("jc_task:" + user.id); $("task-state").hidden = !importTask; if (importTask) refreshImport(); }
      if (state.view === "history") window.JC.loadHistory(); return user;
    } catch (error) {
      state.user = null; $("account-name").textContent = "登录 / 注册"; $("account-avatar").textContent = "访";
      $("profile-status").textContent = "登录后可以保存画像。"; return null;
    }
  }
  $("account-trigger").addEventListener("click", () => { if (!state.user) openAuth(); else { $("account-menu").hidden = !$("account-menu").hidden; $("account-trigger").setAttribute("aria-expanded", String(!$("account-menu").hidden)); } });
  document.addEventListener("keydown", event => { if (event.key === "Escape" && !$("account-menu").hidden) { $("account-menu").hidden = true; $("account-trigger").setAttribute("aria-expanded", "false"); $("account-trigger").focus(); } });
  document.addEventListener("click", event => { if (!$("account-trigger").contains(event.target) && !$("account-menu").contains(event.target)) $("account-menu").hidden = true; });
  $("auth-switch").addEventListener("click", () => setAuthMode(authMode === "login" ? "register" : "login"));
  $("auth-form").addEventListener("submit", async event => {
    event.preventDefault(); $("auth-submit").disabled = true; $("auth-error").textContent = "";
    const credentials = { username: $("username").value, password: $("password").value };
    try {
      if (authMode === "register") await api("/api/auth/register", jsonOptions(credentials));
      await api("/api/auth/login", jsonOptions(credentials)); $("password").value = "";
      const user = await refreshAccount(); if (!user) throw new Error("登录状态未能确认，请重新登录。");
      $("auth-dialog").close(); notify(authMode === "register" ? "账号已创建，可以开始探索岗位" : "已登录，可以继续探索岗位");
    } catch (error) { $("auth-error").textContent = error.message; $("password").focus(); }
    finally { $("auth-submit").disabled = false; }
  });
  $("logout").addEventListener("click", async () => {
    $("logout").disabled = true;
    try { await api("/api/auth/logout", jsonOptions({})); if (state.user) sessionStorage.removeItem("jc_task:" + state.user.id); stopPolling(); location.reload(); }
    catch (error) { notify(error.message); $("logout").disabled = false; }
  });
  window.JC.hasUnsavedProfile = () => dirtyProfile;
  window.addEventListener("jc:profile-saved", event => { dirtyProfile = false; $("profile-text").value = event.detail.text || ""; $("profile-status").textContent = "已保存。"; });
  $("profile-text").addEventListener("input", () => { dirtyProfile = true; $("profile-status").textContent = "有未保存的修改。"; });
  $("save-profile").addEventListener("click", async () => {
    if (!state.user) { openAuth("登录后，把画像保存到你的账号下。"); return; }
    $("save-profile").disabled = true;
    try {
      const value = await api("/api/profile", { ...jsonOptions({ text: $("profile-text").value }), method: "PUT" });
      state.user.profile = { ...state.user.profile, ...value }; dirtyProfile = false; $("profile-status").textContent = "已保存。"; notify("画像已保存");
    } catch (error) { $("profile-status").textContent = error.message; } finally { $("save-profile").disabled = false; }
  });
  window.addEventListener("beforeunload", event => { if (dirtyProfile) { event.preventDefault(); event.returnValue = ""; } });
  $("import-file").addEventListener("change", () => { const file = $("import-file").files[0]; $("file-name").textContent = file ? `${file.name}（${file.size < 1024 ? new Intl.NumberFormat("zh-CN").format(file.size) + " 字节" : new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 1 }).format(file.size / 1024) + " KB"}）` : ""; $("upload-error").textContent = ""; });
  function drawTask(task) {
    $("task-state").hidden = false; $("task-state").dataset.status = task.status; $("task-label").textContent = statusText[task.status] || "状态未知"; $("task-id").textContent = "任务编号：" + task.id;
    const phase = task.status === "completed" ? 2 : ["running", "retrying"].includes(task.status) ? 1 : 0;
    document.querySelectorAll(".task-step").forEach((step, index) => step.classList.toggle("active", index <= phase));
    $("retry-import").hidden = task.status !== "failed" || task.result?.retryable === false;
    const explanations = { queued: "文件已收到，等待后台处理。", running: "正在校验文件、保存岗位并更新检索索引，请稍候…", retrying: "暂时未能完成处理，系统将自动重试。", completed: "岗位与检索索引已更新，可以返回岗位探索查看。", failed: task.result?.retryable === false ? "文件格式、必填字段或岗位归属未通过检查。请核对文件后重新提交。" : "处理暂未完成，可以重试；若持续失败，请检查服务是否就绪。" };
    $("import-status").textContent = (explanations[task.status] || "请稍后刷新状态。") + (task.attempts ? ` 已尝试 ${task.attempts} 次。` : "");
    $("import-status").classList.toggle("error", task.status === "failed");
  }
  async function refreshImport() {
    stopPolling(); if (!importTask || !state.user) return;
    try {
      const requested = importTask; const task = await api("/api/imports/" + encodeURIComponent(requested)); if (requested !== importTask || !state.user) return; drawTask(task);
      if (["queued", "running", "retrying"].includes(task.status) && pollCount++ < 60) pollTimer = setTimeout(refreshImport, 3000);
    } catch (error) { $("import-status").textContent = error.message; }
  }
  function selectImportMode(mode) {
    for (const value of ["file", "text"]) {
      $("import-" + value + "-panel").hidden = mode !== value;
      $("import-tab-" + value).setAttribute("aria-pressed", String(mode === value));
    }
    $("upload-error").textContent = "";
  }
  $("import-tab-file").addEventListener("click", () => selectImportMode("file"));
  $("import-tab-text").addEventListener("click", () => selectImportMode("text"));
  function trackImport(task) {
    importTask = task.id; sessionStorage.setItem("jc_task:" + state.user.id, importTask); pollCount = 0; drawTask(task); notify("已提交导入"); refreshImport();
  }
  $("import-text-panel").addEventListener("submit", async event => {
    event.preventDefault();
    if (!state.user) { openAuth("登录后导入岗位。"); return; }
    if (!$("import-text-panel").reportValidity()) return;
    const value = {title:$("import-title").value, jd_text:$("import-jd").value, company:$("import-company").value || null, cities:$("import-cities").value.split(/[,，、;；]/).map(s=>s.trim()).filter(Boolean), employment_type_text:$("import-type").value || null, education_text:$("import-education").value || null, experience_text:$("import-experience").value || null, salary_text:$("import-salary").value || null, url:$("import-url").value || null, source_job_id:$("import-source-id").value || null};
    if (!value.title.trim() || !value.jd_text.trim()) { $("upload-error").textContent = "请填写岗位名称和JD正文。"; return; }
    const owner = state.user.id; $("import-text-submit").disabled = true; $("upload-error").textContent = "";
    try { const task = await api("/api/imports/text", jsonOptions(value)); if (state.user?.id === owner) trackImport(task); }
    catch (error) { $("upload-error").textContent = error.message; }
    finally { $("import-text-submit").disabled = false; }
  });
  $("upload").addEventListener("click", async () => {
    if (!state.user) { openAuth("登录后导入岗位文件，并查看属于你的处理任务。"); return; }
    const file = $("import-file").files[0];
    if (!file) { $("upload-error").textContent = "请先选择一份 JSON 或 CSV 文件。"; $("import-file").focus(); return; }
    if (file.size > 5_000_000 || !/\.(json|csv)$/i.test(file.name)) { $("upload-error").textContent = "请选择不超过 5 MB 的 JSON 或 CSV 文件。"; return; }
    $("upload").disabled = true; $("upload-error").textContent = "";
    try {
      const content = await file.text(); const hash = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(content));
      const key = Array.from(new Uint8Array(hash), value => value.toString(16).padStart(2, "0")).join("");
      // API中filename是格式标识；完整上传内容保留，允许本地选择中文文件名。
      const filename = /^[a-zA-Z0-9_.-]{1,100}\.(json|csv)$/.test(file.name) ? file.name : "jobs." + file.name.split(".").pop().toLowerCase();
      const task = await api("/api/imports", jsonOptions({ filename, content, idempotency_key: key }));
      trackImport(task);
    } catch (error) { $("upload-error").textContent = error.message; } finally { $("upload").disabled = false; }
  });
  $("refresh-import").addEventListener("click", () => { pollCount = 0; refreshImport(); });
  $("retry-import").addEventListener("click", async () => {
    if (!importTask) return; $("retry-import").disabled = true;
    try { await api("/api/imports/" + encodeURIComponent(importTask) + "/retry", jsonOptions({})); pollCount = 0; await refreshImport(); notify("任务已重新排队"); }
    catch (error) { $("import-status").textContent = error.message; } finally { $("retry-import").disabled = false; }
  });
  window.addEventListener("jc:view", () => { $("account-menu").hidden = true; });
  refreshAccount();
})();
