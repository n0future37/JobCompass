(() => {
  "use strict";
  const { $, state, api, jsonOptions, node, action, notify } = window.JC;
  const labels = { not_applied: "未投递", applied: "已投递", interviewing: "面试中", offer: "获得 Offer", rejected: "被拒绝" };
  const split = value => value.split(/[,，、;；]/).map(v => v.trim()).filter(Boolean);
  let backgroundDirty = false;
  function put(value) { return { ...jsonOptions(value), method: "PUT" }; }
  function drawSummary(value = {}) {
    const pieces = [value.education, value.experience_years == null ? "" : `${value.experience_years} 年经验`, ...(value.skills || []).slice(0, 6)].filter(Boolean);
    $("profile-summary-content").textContent = !state.user ? "登录后查看画像。" : pieces.length ? pieces.join(" · ") : "暂无画像，上传简历或手动填写。";
  }
  function fillBackground(value = {}) {
    drawSummary(value);
    $("bg-education").value = value.education || ""; $("bg-experience").value = value.experience_years ?? "";
    for (const field of ["major", "projects"]) $("bg-" + field).value = value[field] || "";
    for (const field of ["skills", "interests", "preferred_cities"]) $("bg-" + field).value = (value[field] || []).join("、");
  }
  async function loadBackground() {
    if (!state.user || backgroundDirty) return;
    try { fillBackground(await api("/api/background")); } catch (error) { $("background-status").textContent = error.message; }
  }
  window.JC.hasUnsavedBackground = () => backgroundDirty;
  window.addEventListener("jc:profile-saved", event => { backgroundDirty = false; fillBackground(event.detail.background); $("background-status").textContent = "简历推荐背景已保存。"; });
  $("background-form").addEventListener("input", () => { backgroundDirty = true; $("background-status").textContent = "有未保存的修改。"; });
  $("background-form").addEventListener("submit", async event => {
    event.preventDefault(); if (!state.user) { window.JC.openAuth(); return; }
    const value = { education: $("bg-education").value || null, experience_years: $("bg-experience").value === "" ? null : Number($("bg-experience").value), major: $("bg-major").value, projects: $("bg-projects").value };
    for (const field of ["skills", "interests", "preferred_cities"]) value[field] = split($("bg-" + field).value);
    $("save-background").disabled = true;
    try { await api("/api/background", put(value)); drawSummary(value); backgroundDirty = false; $("background-status").textContent = "已保存。"; notify("背景已保存"); }
    catch (error) { $("background-status").textContent = error.message; }
    finally { $("save-background").disabled = false; }
  });
  window.addEventListener("beforeunload", event => { if (backgroundDirty) { event.preventDefault(); event.returnValue = ""; } });
  window.addEventListener("jc:account", () => { backgroundDirty = false; fillBackground(); if (state.user) loadBackground(); });
  window.addEventListener("jc:view", event => { if (event.detail === "profile") loadBackground(); if (event.detail === "applications") loadApplications(); });

  async function loadApplications() {
    const list = $("application-list");
    if (!state.user) { list.textContent = "登录后查看投递记录。"; return; }
    list.textContent = "正在读取投递记录…";
    try {
      const rows = await api("/api/applications"); list.replaceChildren();
      if (!rows.length) { list.textContent = "暂无记录，可在岗位详情中添加。"; return; }
      for (const row of rows) {
        const item = node("div", undefined, "career-record");
        item.append(node("h3", row.snapshot.title || row.job_key), node("p", row.snapshot.company || "公司未知", "muted"));
        const edit = action("查看岗位与更新", async () => {
          try { window.JC.showJob(await api("/api/jobs/" + encodeURIComponent(row.job_key))); }
          catch (error) { if (error.status === 404) window.JC.showJob({ job_key: row.job_key, title: row.snapshot.title || row.job_key, company: row.snapshot.company, jd_text: "岗位已下架。此处保留投递记录，无法展示当前 JD。", url: row.snapshot.url }); else notify(error.message); }
        });
        item.append(node("span", labels[row.status], "chip"), ...(row.note ? [node("p", row.note, "small muted")] : []), edit); list.append(item);
      }
    } catch (error) { list.textContent = error.message; }
  }
  $("refresh-applications").addEventListener("click", loadApplications);

  window.JC.careerControls = function (job, target) {
    const box = node("details", undefined, "career-controls"); box.append(node("summary", "偏好与投递")); target.append(box);
    if (!state.user) { box.append(action("登录后记录", () => window.JC.openAuth())); return; }
    const owner = state.user.id;
    const statusLine = node("p", "正在读取你的记录…", "status-line"); box.append(statusLine);
    Promise.all([api("/api/job-preferences"), api("/api/applications")]).then(([prefs, apps]) => {
      if (!box.isConnected || state.user?.id !== owner) return;
      const pref = prefs.find(r => r.job_key === job.job_key); let app = apps.find(r => r.job_key === job.job_key);
      const preferenceRow = node("div", undefined, "career-actions"); const prefText = node("span");
      let currentPreference = pref?.preference || "neutral";
      function drawHeading() { box.firstChild.textContent = ["偏好与投递", currentPreference === "like" ? "已喜欢" : currentPreference === "dislike" ? "已排除" : "", app ? labels[app.status] : ""].filter(Boolean).join(" · "); }
      function drawPreference(value) { currentPreference = value || "neutral"; drawHeading(); prefText.textContent = ({ like: "已喜欢", dislike: "已排除", neutral: "" })[value || "neutral"]; }
      drawPreference(pref?.preference);
      for (const [value, label] of [["like", "喜欢"], ["dislike", "不喜欢"], ["neutral", "清除"]]) {
        const button = action(label, async () => {
          button.disabled = true;
          try { const row = await api("/api/job-preferences/" + encodeURIComponent(job.job_key), put({ preference: value })); drawPreference(row.preference); statusLine.textContent = "偏好已保存。"; }
          catch (error) { statusLine.textContent = error.message; } finally { button.disabled = false; }
        }, "button secondary"); preferenceRow.append(button);
      }
      const form = node("form"); const statusLabel = node("label", "投递状态"); statusLabel.htmlFor = "career-state";
      const select = node("select"); select.id = "career-state";
      const unset = node("option", "未记录"); unset.value = ""; select.append(unset);
      for (const [value, label] of Object.entries(labels)) { const option = node("option", label); option.value = value; select.append(option); }
      select.value = app?.status || ""; select.required = true;
      const noteLabel = node("label", "备注"); noteLabel.htmlFor = "career-note";
      const note = node("textarea"); note.id = "career-note"; note.maxLength = 1000; note.rows = 2; note.value = app?.note || "";
      const save = node("button", "保存记录", "button primary"); save.type = "submit";
      form.append(statusLabel, select, noteLabel, note, save);
      form.addEventListener("submit", async event => {
        event.preventDefault(); save.disabled = true;
        try { app = await api("/api/applications/" + encodeURIComponent(job.job_key), put({ status: select.value, note: note.value, expected_version: app?.version || 0 })); statusLine.textContent = "已保存。"; drawHeading(); if (state.view === "applications") loadApplications(); }
        catch (error) { statusLine.textContent = error.message; } finally { save.disabled = false; }
      });
      box.append(preferenceRow, prefText, form); statusLine.textContent = "";
    }).catch(error => { statusLine.textContent = error.message; });
  };
  if (state.user) loadBackground();
})();
