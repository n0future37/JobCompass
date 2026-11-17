(() => {
  "use strict";
  const { $, state, api, jsonOptions, node, action, notify } = window.JC;
  let draft = null, dirty = false;
  const split = value => value.split(/[,，、;；]/).map(s => s.trim()).filter(Boolean);
  const names = {education:"学历",experience_years:"工作年限",major:"专业",skills:"自述技能",projects:"项目经历",interests:"兴趣方向",preferred_cities:"意向城市"};
  function manualDirty() { return window.JC.hasUnsavedProfile?.() || window.JC.hasUnsavedBackground?.(); }
  function guardDraft() { if (manualDirty()) throw new Error("手动画像有未保存修改，请先保存后再载入简历预览。"); if (dirty) throw new Error("当前简历预览有未保存修改，请先确认保存或放弃预览。"); }
  function clearDraft() { draft = null; dirty = false; $("resume-preview").hidden = true; $("resume-preview-status").textContent = ""; }
  function render(value) {
    draft = value; dirty = false; $("resume-preview").hidden = false; $("resume-preview-title").textContent = value.filename;
    const bg = value.proposed_background;
    $("resume-education").value = bg.education || ""; $("resume-experience_years").value = bg.experience_years ?? "";
    for (const field of ["major","projects"]) $("resume-"+field).value = bg[field] || "";
    for (const field of ["skills","interests","preferred_cities"]) $("resume-"+field).value = (bg[field] || []).join("、");
    $("resume-profile").value = value.proposed_profile_text; $("resume-extracted").textContent = value.text;
    $("resume-conflicts").textContent = value.conflicts.length ? "与现有画像不同，已保留原值：" + value.conflicts.map(k=>names[k]).join("、") : "";
    const warnings = [...value.warnings];
    $("resume-warnings").replaceChildren(...warnings.map(s=>node("p",s,"small muted")));
    const evidence = $("resume-evidence"); evidence.replaceChildren();
    for (const [field, sources] of Object.entries(value.evidence)) {
      const label = field.startsWith("skills:") ? "技能 "+field.slice(7) : names[field] || field;
      evidence.append(node("p",label,"resume-evidence-label"));
      for (const source of sources) evidence.append(node("blockquote",`提取文本第 ${source.line} 行：${source.quote}`,"detail-quote"));
    }
    if (!Object.keys(value.evidence).length) evidence.append(node("p","未识别到可靠字段，请参考原文手动填写。","muted"));
    $("resume-preview-status").textContent = "核对后保存。";
  }
  async function refreshList() {
    const list = $("resume-list");
    if (!state.user) { list.textContent = "登录后上传和查看自己的简历。"; return; }
    const owner = state.user.id;
    try {
      const rows = await api("/api/resumes"); if (state.user?.id !== owner) return;
      list.replaceChildren(); if (!rows.length) { list.textContent = "还没有上传简历。"; return; }
      for (const row of rows) {
        const item = node("div",undefined,"resume-item"); const description = node("div"); description.append(node("strong",row.filename),node("p",row.status === "confirmed" ? "已保存画像" : "待核对","small muted"));
        const load = action("查看解析",async()=>{
          try { guardDraft(); const result = await api("/api/resumes/"+row.id); if (state.user?.id === owner) render(result); }
          catch (error) { $("resume-status").textContent = error.message; }
        },"button secondary");
        const download = node("a","下载原件","text-button"); download.href = "/api/resumes/"+row.id+"/original"; download.setAttribute("download","");
        let confirming = false;
        const remove = action("删除此简历",async()=>{
          if (!confirming) { confirming = true; remove.textContent = "再次点击确认删除原件"; return; }
          remove.disabled = true;
          try { await api("/api/resumes/"+row.id,{method:"DELETE"}); if (draft?.id === row.id) clearDraft(); await refreshList(); notify("原件和解析稿已删除；已保存画像保留。"); }
          catch (error) { $("resume-status").textContent = error.message; remove.disabled = false; }
        },"text-button");
        const actions = node("div",undefined,"career-actions"); actions.append(load,download,remove); item.append(description,actions); list.append(item);
      }
    } catch(error) { list.textContent = error.message; }
  }
  $("resume-file").addEventListener("change",()=>{ $("resume-status").textContent = $("resume-file").files[0]?.name || ""; });
  $("upload-resume").addEventListener("click",async()=>{
    if (!state.user) { window.JC.openAuth("登录后，将简历保存到你的私有画像空间。"); return; }
    const owner = state.user.id; const file = $("resume-file").files[0];
    try {
      guardDraft();
      if (!file || !/\.(pdf|docx|txt)$/i.test(file.name)) throw new Error("请选择PDF、DOCX或UTF-8 TXT简历。");
      if (file.size === 0 || file.size > 5_000_000) throw new Error("请选择非空且不超过5MB的文件。");
      $("upload-resume").disabled = true; $("resume-status").textContent = "正在解析…";
      const result = await api("/api/resumes",{method:"POST",headers:{"Content-Type":"application/octet-stream","X-Resume-Filename":encodeURIComponent(file.name)},body:file});
      if (state.user?.id !== owner) return;
      render(result); $("resume-status").textContent = "解析完成。"; await refreshList();
    } catch (error) { $("resume-status").textContent = error.message; }
    finally { $("upload-resume").disabled = false; }
  });
  $("resume-review-form").addEventListener("input",()=>{dirty = true; $("resume-preview-status").textContent = "有未保存修改。";});
  $("resume-review-form").addEventListener("submit",async event=>{
    event.preventDefault(); if (!draft || !state.user) return;
    if (manualDirty()) { $("resume-preview-status").textContent = "手动画像有未保存修改，请先保存并重新载入预览。"; return; }
    const owner = state.user.id;
    const background = {education:$("resume-education").value || null,experience_years:$("resume-experience_years").value === "" ? null : Number($("resume-experience_years").value),major:$("resume-major").value,projects:$("resume-projects").value};
    for (const field of ["skills","interests","preferred_cities"]) background[field] = split($("resume-"+field).value);
    $("confirm-resume").disabled = true;
    try {
      const result = await api("/api/resumes/"+draft.id+"/confirm",jsonOptions({profile_etag:draft.profile_etag,profile_text:$("resume-profile").value,background}));
      if (state.user?.id !== owner) return;
      state.user.profile = result.profile; dirty = false; window.dispatchEvent(new CustomEvent("jc:profile-saved",{detail:result.profile}));
      $("resume-preview-status").textContent = "已保存，下次推荐生效。";
      $("resume-status").textContent = ""; notify("简历画像已保存"); await refreshList();
      draft = await api("/api/resumes/"+draft.id);
    } catch (error) { $("resume-preview-status").textContent = error.message; }
    finally { $("confirm-resume").disabled = false; }
  });
  $("discard-resume-preview").addEventListener("click",()=>{clearDraft(); $("resume-status").textContent = "已取消预览。";});
  $("refresh-resumes").addEventListener("click",refreshList);
  window.addEventListener("jc:view",event=>{if(event.detail === "profile") refreshList();});
  window.addEventListener("jc:account",()=>{clearDraft(); $("resume-file").value=""; $("resume-list").replaceChildren(); $("resume-status").textContent=""; refreshList();});
  window.addEventListener("beforeunload",event=>{if(dirty){event.preventDefault();event.returnValue="";}});
  if (state.user) refreshList();
})();
