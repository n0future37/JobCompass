"""实验用token预算切块，保留原JD字符和行范围，不修改生产默认。"""

from .preprocessing import digest


def token_chunks(jobs, tokenizer, budget=384, overlap=48):
    if not 32 <= budget <= 512 or not 0 <= overlap < budget // 2:
        raise ValueError("token预算或重叠参数不合法")
    result = []
    for job in jobs:
        text = job["jd_text"]
        offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        title_tokens = len(tokenizer(job["title"], add_special_tokens=False)["input_ids"])
        capacity = budget - title_tokens - 4
        if capacity <= overlap:
            raise ValueError("岗位标题过长，无法满足分块预算")
        start = 0
        while start < len(offsets):
            end = min(len(offsets), start + capacity)
            left, right = offsets[start][0], offsets[end - 1][1]
            while len(tokenizer(job["title"] + "\n" + text[left:right])["input_ids"]) > budget:
                end -= 1
                if end <= start:
                    raise ValueError("单token分块仍超过预算")
                right = offsets[end - 1][1]
            result.append(
                {
                    "id": digest([job["job_key"], job["content_hash"], budget, overlap, left, right])[:32],
                    "job_key": job["job_key"],
                    "content_hash": job["content_hash"],
                    "version": job["version"],
                    "line_start": text[:left].count("\n") + 1,
                    "line_end": text[:right].count("\n") + 1,
                    "char_start": left,
                    "char_end": right,
                    "text": job["title"] + "\n" + text[left:right],
                }
            )
            if end == len(offsets):
                break
            start = max(start + 1, end - overlap)
    return result
