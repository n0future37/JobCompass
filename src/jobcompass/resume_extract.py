"""受父进程超时控制的本地文件文本提取；不执行宏、链接、脚本或模型调用。"""

import io
import json
import sys
import unicodedata
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

MAX_BYTES = 5_000_000
MAX_TEXT = 30_000
MAX_PAGES = 15


def extract(content, extension):
    if not content or len(content) > MAX_BYTES:
        raise ValueError("文件为空或超过5MB")
    warnings = []
    if extension == ".pdf":
        from pypdf import PdfReader

        if not content.startswith(b"%PDF-"):
            raise ValueError("PDF格式不正确")
        reader = PdfReader(io.BytesIO(content), strict=True)
        if reader.is_encrypted:
            raise ValueError("暂不支持加密PDF，请先在本地导出未加密副本")
        if len(reader.pages) > MAX_PAGES:
            raise ValueError("简历不能超过15页")
        pages = []
        length = 0
        for page in reader.pages:
            part = page.extract_text() or ""
            length += len(part)
            if length > MAX_TEXT:
                raise ValueError("提取正文超过30000字符，请精简后重传")
            if len(part.strip()) < 10:
                warnings.append("部分页面缺少可提取文字，请检查扫描图片或文字丢失")
            pages.append(part)
        value = "\n".join(pages)
        warnings.append("PDF多栏、表格的阅读顺序可能有偏差，请对照原件核对")
    elif extension == ".docx":
        if not content.startswith(b"PK"):
            raise ValueError("DOCX格式不正确")
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            entries = archive.infolist()
            if len(entries) > 1000 or sum(e.file_size for e in entries) > 12_000_000:
                raise ValueError("DOCX解压内容过大")
            data = archive.read("word/document.xml")
        if b"<!DOCTYPE" in data or b"<!ENTITY" in data:
            raise ValueError("不支持含实体声明的DOCX")
        document = ET.fromstring(data)
        ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        paragraphs = []
        for paragraph in document.iter(ns + "p"):
            paragraphs.append("".join(el.text or "" for el in paragraph.iter(ns + "t")))
        value = "\n".join(paragraphs)
        warnings.append("DOCX提取正文和表格中的段落，不读取页眉、批注或图片")
    elif extension == ".txt":
        if b"\x00" in content:
            raise ValueError("TXT含二进制内容；请使用UTF-8文本")
        try:
            value = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise ValueError("TXT需要使用UTF-8编码") from None
    else:
        raise ValueError("仅支持PDF、DOCX和UTF-8 TXT")
    value = unicodedata.normalize("NFKC", value).replace("\r\n", "\n").replace("\r", "\n")
    value = "\n".join(" ".join(line.split()) for line in value.splitlines() if line.strip())
    if len(value) < 10:
        raise ValueError("未提取到足够文字；扫描件PDF暂不支持OCR，请上传文字版")
    if len(value) > MAX_TEXT:
        raise ValueError("提取正文超过30000字符，请精简后重传")
    if "\ufffd" in value:
        warnings.append("正文存在无法识别字符，请核对文字编码")
    return dict(text=value, warnings=list(dict.fromkeys(warnings)))


if __name__ == "__main__":
    try:
        result = extract(Path(sys.argv[1]).read_bytes(), sys.argv[2])
        print(json.dumps(dict(ok=True, **result), ensure_ascii=True))
    except Exception as error:
        message = str(error) if type(error) is ValueError else "文件损坏或格式暂不支持，请导出标准文件后重传"
        print(json.dumps(dict(ok=False, error=message), ensure_ascii=True))
