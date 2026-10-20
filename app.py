
import io
import os
import re
import zipfile
from pathlib import PurePosixPath
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import pandas as pd
import pytesseract
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

APP_NAME = "NEXUS | TIFF OCR Renamer"

MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_TIFF_BYTES = 25 * 1024 * 1024
MAX_EXPANDED_BYTES = 250 * 1024 * 1024
MAX_FILES = 200

DEFAULT_CONFIG = {
    "Policy": {
        "labels": ["Open Policy No", "Policy No", "Policy Number"],
        "replace_characters": {"/": "_"},
        "preserve_hyphen": True,
    },
    "Claim": {
        "labels": ["Claim No", "Claim Number", "Claim #"],
        "replace_characters": {"/": "_"},
        "preserve_hyphen": True,
    },
    "Certificate": {
        "labels": ["Certificate No", "Certificate Number", "Certificate #"],
        "replace_characters": {"/": "_"},
        "preserve_hyphen": True,
    },
    "Endorsement": {
        "labels": ["Endorsement No", "Endorsement Number", "Endorsement #"],
        "replace_characters": {"/": "_"},
        "preserve_hyphen": True,
    },
}


def clean_ref(value, config):
    value = value.strip()

    stop_labels = (
        r"Open Policy Period|Policy Period|Policy No|Policy Number|"
        r"Claim No|Claim Number|Certificate No|Certificate Number|"
        r"Endorsement No|Endorsement Number|Certificate Period|TRN"
    )
    value = re.split(
        r"\s{2,}|(?=\b(?:" + stop_labels + r")\b)",
        value,
        maxsplit=1,
        flags=re.I,
    )[0]

    value = value.strip().strip(":;,. ")
    value = re.sub(r"\s*([/-])\s*", r"\1", value)
    value = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", value)

    for old, new in config.get("replace_characters", {"/": "_"}).items():
        value = value.replace(old, new)

    if not config.get("preserve_hyphen", True):
        value = value.replace("-", "_")

    value = re.sub(r'[<>:"\\|?*\x00-\x1f]', "_", value)
    return value.rstrip(" .")


def extract_reference(text, config):
    labels = sorted(config.get("labels", []), key=len, reverse=True)

    for line in text.splitlines():
        line = line.strip()

        for label in labels:
            label_pattern = re.escape(label).replace(r"\ ", r"\s*")
            pattern = re.compile(
                r"(?<![A-Za-z])" + label_pattern
                + r"\s*\.?\s*:?\s*(?:[:.-]\s*)?(.+?)\s*$",
                re.I,
            )
            match = pattern.search(line)

            if match:
                result = clean_ref(match.group(1), config)
                if result and re.search(r"[A-Za-z0-9]", result):
                    return result

    for label in labels:
        label_pattern = re.escape(label).replace(r"\ ", r"\s*")
        match = re.search(
            label_pattern + r"\s*\.?\s*:?\s*([A-Z0-9][A-Z0-9/\- _]{1,})",
            text,
            re.I,
        )
        if match:
            result = clean_ref(match.group(1), config)
            if result:
                return result

    return ""


def ocr_tiff(file_bytes, config):
    with Image.open(io.BytesIO(file_bytes)) as source:
        source.seek(0)
        page = source.copy()

    page = ImageOps.grayscale(page)
    page = ImageOps.autocontrast(page)

    if page.width < 1600:
        ratio = 1600 / page.width
        page = page.resize(
            (1600, max(1, int(page.height * ratio)))
        )

    page = ImageEnhance.Contrast(page).enhance(1.5)
    page = page.filter(ImageFilter.SHARPEN)

    text = pytesseract.image_to_string(page, config="--psm 6")
    return extract_reference(text, config)


def safe_zip_path(name):
    # ZIP files are read in memory; no uploaded path is extracted to disk.
    name = name.replace("\\", "/")
    path = PurePosixPath(name)

    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Unsafe path found in ZIP archive.")

    if not path.parts or not path.name:
        return None

    return str(path)


def unique_input_path(path, seen):
    candidate = path
    stem = str(PurePosixPath(path).with_suffix(""))
    suffix = PurePosixPath(path).suffix
    number = 2

    while candidate.casefold() in seen:
        candidate = f"{stem}-{number}{suffix}"
        number += 1

    seen.add(candidate.casefold())
    return candidate


def read_uploads(uploads):
    items = []
    seen = set()
    expanded_bytes = 0

    for upload in uploads:
        raw = upload.getvalue()

        if len(raw) > MAX_UPLOAD_BYTES:
            raise ValueError(
                f"{upload.name} exceeds the 100 MB upload limit."
            )

        if upload.name.lower().endswith(".zip"):
            try:
                archive = zipfile.ZipFile(io.BytesIO(raw))
            except zipfile.BadZipFile:
                raise ValueError(f"{upload.name} is not a valid ZIP file.")

            with archive:
                for info in archive.infolist():
                    if info.is_dir():
                        continue

                    safe_path = safe_zip_path(info.filename)
                    if not safe_path:
                        continue

                    if not safe_path.lower().endswith((".tif", ".tiff")):
                        continue

                    if len(items) >= MAX_FILES:
                        raise ValueError(
                            f"Maximum limit is {MAX_FILES} TIFF files."
                        )

                    if info.file_size > MAX_TIFF_BYTES:
                        raise ValueError(
                            f"{safe_path} exceeds the 25 MB per-file limit."
                        )

                    expanded_bytes += info.file_size
                    if expanded_bytes > MAX_EXPANDED_BYTES:
                        raise ValueError(
                            "The ZIP contents exceed the 250 MB limit."
                        )

                    data = archive.read(info)
                    final_path = unique_input_path(safe_path, seen)
                    items.append({
                        "path": final_path,
                        "data": data,
                    })

        elif upload.name.lower().endswith((".tif", ".tiff")):
            if len(raw) > MAX_TIFF_BYTES:
                raise ValueError(
                    f"{upload.name} exceeds the 25 MB per-file limit."
                )

            if len(items) >= MAX_FILES:
                raise ValueError(
                    f"Maximum limit is {MAX_FILES} TIFF files."
                )

            final_path = unique_input_path(
                PurePosixPath(upload.name).name, seen
            )
            items.append({"path": final_path, "data": raw})
            expanded_bytes += len(raw)

        else:
            raise ValueError(
                f"Unsupported file: {upload.name}. Upload TIFF or ZIP."
            )

    if not items:
        raise ValueError("No TIFF files were found in the uploads.")

    return items


def scan_one(item, config):
    try:
        ref = ocr_tiff(item["data"], config)
        status = "Ready" if ref else "Missing: no reference detected"
    except Exception as exc:
        ref = ""
        status = "OCR error: " + str(exc)[:100]

    return {
        "path": item["path"],
        "ref": ref,
        "status": status,
    }


def make_rows(items, config, workers):
    results = [None] * len(items)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(scan_one, item, config): index
            for index, item in enumerate(items)
        }

        progress = st.progress(0, text="Starting OCR...")
        completed = 0

        for future in as_completed(future_map):
            index = future_map[future]
            results[index] = future.result()
            completed += 1
            progress.progress(
                completed / len(items),
                text=f"OCR processed {completed} of {len(items)} TIFF files",
            )

        progress.empty()

    return results


def calculate_destinations(rows, edited_refs):
    counts = {}
    output = []

    for index, row in enumerate(rows):
        original = PurePosixPath(row["path"])
        folder = str(original.parent)
        if folder == ".":
            folder = ""

        ref = clean_ref(
            str(edited_refs[index]).strip(),
            DEFAULT_CONFIG[st.session_state["doc_type"]],
        ) if str(edited_refs[index]).strip() else ""

        if ref:
            key = (folder.casefold(), ref.casefold())
            counts[key] = counts.get(key, 0) + 1
            occurrence = counts[key]
            stem = ref if occurrence == 1 else f"{ref}-{occurrence}"
            filename = stem + original.suffix.lower()
            destination = str(PurePosixPath(folder) / filename) if folder else filename

            status = (
                f"Duplicate #{occurrence}" if occurrence > 1
                else ("Manually corrected" if ref != row["ref"] else "Ready")
            )
        else:
            box = original.parent.name or "Uploaded"
            missing_folder = PurePosixPath(folder) / "Missing" / box
            destination = str(missing_folder / original.name)
            status = (
                "Missing: reference cleared"
                if row["ref"] else row["status"]
            )

        output.append({
            "path": row["path"],
            "ref": ref,
            "proposed": destination,
            "status": status,
        })

    # Prevent duplicate destination paths, including in Missing folders.
    reserved = set()
    for row in output:
        destination = row["proposed"]
        key = destination.casefold()

        if key in reserved:
            path = PurePosixPath(destination)
            stem, suffix = path.stem, path.suffix
            number = 2

            while True:
                candidate = str(path.with_name(f"{stem}-{number}{suffix}"))
                if candidate.casefold() not in reserved:
                    row["proposed"] = candidate
                    key = candidate.casefold()
                    break
                number += 1

        reserved.add(key)

    return output


def build_zip(items, destinations):
    data_by_path = {item["path"]: item["data"] for item in items}
    memory_file = io.BytesIO()

    with zipfile.ZipFile(
        memory_file, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for destination in destinations:
            source = destination["path"]
            archive.writestr(
                destination["proposed"],
                data_by_path[source],
            )

    memory_file.seek(0)
    return memory_file.getvalue()


# ---------------- Web interface ----------------

st.set_page_config(
    page_title=APP_NAME,
    page_icon="📄",
    layout="wide",
)

st.title("NEXUS")
st.subheader("TIFF OCR Renamer")
st.caption(
    "Document naming automation • First-page OCR • "
    "Reference correction • ZIP download"
)

with st.sidebar:
    st.header("Workspace settings")
    doc_type = st.selectbox(
        "Document type",
        list(DEFAULT_CONFIG.keys()),
    )
    workers = st.select_slider(
        "OCR workers",
        options=[1, 2, 4, 6],
        value=2,
    )
    st.info(
        "Limits: 100 MB per upload, 25 MB per TIFF, "
        "250 MB expanded ZIP contents, and 200 TIFF files."
    )

st.session_state["doc_type"] = doc_type

uploads = st.file_uploader(
    "Upload TIFF files or ZIP folders containing TIFF files",
    type=["tif", "tiff", "zip"],
    accept_multiple_files=True,
)

if st.button("Scan / Preview", type="primary", disabled=not uploads):
    try:
        with st.spinner("Reading uploaded files..."):
            items = read_uploads(uploads)

        with st.spinner("Running OCR..."):
            rows = make_rows(items, DEFAULT_CONFIG[doc_type], workers)

        st.session_state["items"] = items
        st.session_state["rows"] = rows
        st.session_state["scan_type"] = doc_type
        st.session_state.pop("download_zip", None)

    except Exception as exc:
        st.error(str(exc))

if (
    "rows" in st.session_state
    and st.session_state.get("scan_type") != doc_type
):
    st.warning("The selected document type differs from the last scan. Scan again.")

if "rows" in st.session_state:
    items = st.session_state["items"]
    rows = st.session_state["rows"]

    matched = sum(bool(row["ref"]) for row in rows)
    col1, col2, col3 = st.columns(3)
    col1.metric("Files scanned", len(rows))
    col2.metric("References detected", matched)
    col3.metric("Missing / errors", len(rows) - matched)

    st.markdown("### Review and correct references")
    st.caption(
        "Edit the Reference column. Proposed filenames update below. "
        "The original uploaded files are not changed."
    )

    edit_df = pd.DataFrame([
        {
            "Folder": str(PurePosixPath(row["path"]).parent),
            "Current file": PurePosixPath(row["path"]).name,
            "Reference": row["ref"],
            "OCR status": row["status"],
        }
        for row in rows
    ])

    edited_df = st.data_editor(
        edit_df,
        key="nexus_reference_editor",
        hide_index=True,
        use_container_width=True,
        disabled=["Folder", "Current file", "OCR status"],
        column_config={
            "Reference": st.column_config.TextColumn(
                "Reference (editable)",
                help="Correct the detected reference or leave blank for Missing.",
            )
        },
    )

    destinations = calculate_destinations(
        rows, edited_df["Reference"].tolist()
    )

    st.markdown("### Proposed filenames")
    preview_df = pd.DataFrame([
        {
            "Current file": PurePosixPath(row["path"]).name,
            "Reference": destination["ref"] or "—",
            "Proposed path": destination["proposed"],
            "Status": destination["status"],
        }
        for row, destination in zip(rows, destinations)
    ])
    st.dataframe(
        preview_df,
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("### Document preview")
    selected_index = st.selectbox(
        "Choose a TIFF to preview",
        range(len(items)),
        format_func=lambda i: items[i]["path"],
    )

    try:
        with Image.open(io.BytesIO(items[selected_index]["data"])) as image_file:
            image_file.seek(0)
            page = image_file.copy()
        page.thumbnail((1000, 1000))
        st.image(page, caption=items[selected_index]["path"])
    except Exception as exc:
        st.warning(f"Preview unavailable: {exc}")

    if st.button("Build processed ZIP", type="primary"):
        try:
            zip_bytes = build_zip(items, destinations)
            st.session_state["download_zip"] = zip_bytes
        except Exception as exc:
            st.error(f"Could not build ZIP: {exc}")

    if "download_zip" in st.session_state:
        st.download_button(
            "Download processed TIFFs (ZIP)",
            data=st.session_state["download_zip"],
            file_name="NEXUS_Processed_TIFFs.zip",
            mime="application/zip",
            type="primary",
        )

st.divider()
st.caption(
    "NEXUS TIFF OCR Renamer • Browser version. "
    "Review filenames before using the downloaded documents."
)
