import io
import copy
import os
import re
import zipfile
from pathlib import PurePosixPath

import pandas as pd
import streamlit as st
import pytesseract
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

# If Tesseract is installed in the standard Windows location, use it.
if os.name == "nt":
    default_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(default_tesseract):
        pytesseract.pytesseract.tesseract_cmd = default_tesseract

st.set_page_config(page_title="NEXUS | TIFF OCR Renamer", page_icon="📄", layout="wide")

DEFAULT_CONFIG = {
    "Policy": ["Open Policy No", "Policy No", "Policy Number"],
    "Claim": ["Claim No", "Claim Number", "Claim #"],
    "Certificate": ["Certificate No", "Certificate Number", "Certificate #"],
    "Endorsement": ["Endorsement No", "Endorsement Number", "Endorsement #"],
}

st.title("NEXUS | TIFF OCR Renamer")
st.caption("Upload TIFF files or a ZIP of TIFF files, review detected references, and download a renamed ZIP. Original uploads are not changed.")

if "templates" not in st.session_state:
    st.session_state["templates"] = copy.deepcopy(DEFAULT_CONFIG)


def clean_ref(value: str) -> str:
    value = (value or "").strip()
    value = value.replace("/", "_")
    value = re.sub(r'[<>:"\\|?*\x00-\x1f]', "_", value)
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip(" ._")


def extract_reference(text: str, doc_type: str, templates=None) -> str:
    labels = (templates or DEFAULT_CONFIG).get(doc_type, [])
    # First try to capture text appearing after a configured label.
    for label in labels:
        pattern = re.compile(re.escape(label) + r"\s*[:#\-]?\s*([A-Z0-9][A-Z0-9/_\- ]{2,40})", re.I)
        match = pattern.search(text or "")
        if match:
            candidate = match.group(1).strip().splitlines()[0]
            candidate = re.split(r"\s{2,}|\s+(?:Insured|Customer|Name|Date|Period|Address)\b", candidate, maxsplit=1, flags=re.I)[0]
            candidate = clean_ref(candidate)
            if candidate:
                return candidate
    # Fallback: look for likely reference-shaped codes.
    candidates = re.findall(r"\b[A-Z0-9]+(?:[/_-][A-Z0-9]+){1,7}\b", (text or "").upper())
    if candidates:
        return clean_ref(candidates[0])
    return ""


def ocr_tiff(data: bytes) -> tuple[str, Image.Image]:
    with Image.open(io.BytesIO(data)) as image:
        image.seek(0)
        preview = image.convert("RGB")
        processed = image.convert("L")
        processed = ImageOps.autocontrast(processed)
        if processed.width < 1600:
            new_height = max(1, int(processed.height * (1600 / processed.width)))
            processed = processed.resize((1600, new_height))
        processed = ImageEnhance.Contrast(processed).enhance(1.5)
        processed = processed.filter(ImageFilter.SHARPEN)
        text = pytesseract.image_to_string(processed, config="--psm 6")
        return text, preview


def safe_archive_name(name: str) -> str:
    # Strip absolute paths and traversal components from ZIP entries.
    parts = [p for p in PurePosixPath(name.replace("\\", "/")).parts if p not in ("", ".", "..", "/")]
    return "/".join(parts) or "uploaded.tif"


def collect_uploads(uploaded_files, max_files: int, max_file_mb: int):
    items = []
    errors = []
    for uploaded in uploaded_files:
        raw = uploaded.getvalue()
        if uploaded.name.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                    for info in archive.infolist():
                        if info.is_dir():
                            continue
                        name = safe_archive_name(info.filename)
                        if not name.lower().endswith((".tif", ".tiff")):
                            continue
                        if info.file_size > max_file_mb * 1024 * 1024:
                            errors.append(f"Skipped {name}: exceeds {max_file_mb} MB.")
                            continue
                        items.append((name, archive.read(info)))
                        if len(items) >= max_files:
                            return items, errors + [f"Stopped at the {max_files}-file limit." ]
            except (zipfile.BadZipFile, RuntimeError) as exc:
                errors.append(f"Could not read {uploaded.name}: {exc}")
        elif uploaded.name.lower().endswith((".tif", ".tiff")):
            if len(raw) > max_file_mb * 1024 * 1024:
                errors.append(f"Skipped {uploaded.name}: exceeds {max_file_mb} MB.")
                continue
            items.append((safe_archive_name(uploaded.name), raw))
        else:
            errors.append(f"Skipped {uploaded.name}: upload TIFF/TIFF or ZIP files only.")
        if len(items) >= max_files:
            return items, errors + [f"Stopped at the {max_files}-file limit."]
    return items, errors


def make_rows(items, doc_type: str, templates=None):
    rows = []
    previews = {}
    progress = st.progress(0, text="Starting OCR…")
    total = len(items)
    for index, (name, data) in enumerate(items, start=1):
        row = {"Original path": name, "Detected reference": "", "Status": "Pending", "OCR text": ""}
        try:
            text, preview = ocr_tiff(data)
            reference = extract_reference(text, doc_type, templates)
            row["Detected reference"] = reference
            row["Status"] = "Matched" if reference else "Missing"
            row["OCR text"] = text[:4000]
            previews[name] = preview
        except Exception as exc:
            row["Status"] = f"Error: {exc}"
            row["OCR text"] = ""
        rows.append(row)
        progress.progress(index / total, text=f"OCR {index} of {total}: {name}")
    progress.empty()
    return pd.DataFrame(rows), previews


def calculate_destinations(df: pd.DataFrame, doc_type: str):
    used = {}
    destinations = []
    for _, row in df.iterrows():
        original = str(row["Original path"])
        parent = str(PurePosixPath(original).parent)
        basename = PurePosixPath(original).name
        extension = PurePosixPath(basename).suffix or ".tif"
        reference = clean_ref(str(row.get("Detected reference", "")))
        status = str(row.get("Status", ""))
        if not reference or status.startswith("Error"):
            folder = "Missing"
            stem = PurePosixPath(basename).stem
            target = f"{folder}/{stem}{extension}"
        else:
            folder = parent if parent != "." else ""
            stem = f"{reference}-1"
            key = (folder.lower(), stem.lower())
            count = used.get(key, 0) + 1
            used[key] = count
            # Ensure duplicate names are unique even if source names already collide.
            final_stem = f"{reference}-{count}"
            target = f"{folder}/{final_stem}{extension}" if folder else f"{final_stem}{extension}"
        destinations.append(target)
    return destinations


def build_zip(items, df, destinations):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        seen = set()
        for (original, data), destination in zip(items, destinations):
            candidate = destination
            if candidate.lower() in seen:
                path = PurePosixPath(candidate)
                stem, suffix = path.stem, path.suffix
                n = 2
                while f"{path.parent}/{stem}-{n}{suffix}".lower() in seen:
                    n += 1
                candidate = f"{path.parent}/{stem}-{n}{suffix}"
            seen.add(candidate.lower())
            archive.writestr(candidate, data)
    output.seek(0)
    return output.getvalue()

tab_process, tab_templates = st.tabs(["📄 Rename TIFFs", "🛠️ Edit templates"])

with st.sidebar:
    st.header("Processing settings")
    doc_type = st.selectbox("Document type", list(st.session_state["templates"].keys()))
    max_files = st.number_input("Maximum TIFF files per run", min_value=1, max_value=500, value=100, step=10)
    max_file_mb = st.number_input("Maximum size per TIFF (MB)", min_value=1, max_value=100, value=25, step=5)
    st.caption("For confidential documents, deploy only in an approved, access-controlled environment.")

with tab_templates:
    st.subheader("OCR reference templates")
    st.write("Configure the labels NEXUS looks for when detecting a document reference. Enter one label per line.")
    template_type = st.selectbox("Document type to edit", list(st.session_state["templates"].keys()), key="template_type_editor")
    current_labels = st.session_state["templates"].get(template_type, [])
    template_text = st.text_area(
        "Reference labels",
        value="\n".join(current_labels),
        height=150,
        help="Example: Policy No, Open Policy No, Policy Number — put each label on a separate line.",
        key=f"template_text_{template_type}",
    )
    save_col, reset_col = st.columns(2)
    with save_col:
        if st.button("Save template", type="primary", key="save_template"):
            new_labels = list(dict.fromkeys(line.strip() for line in template_text.splitlines() if line.strip()))
            if not new_labels:
                st.error("Please enter at least one reference label.")
            else:
                st.session_state["templates"][template_type] = new_labels
                st.success(f"Saved {len(new_labels)} labels for {template_type}. They will be used for the next OCR run in this session.")
                st.rerun()
    with reset_col:
        if st.button("Reset to defaults", key="reset_template"):
            st.session_state["templates"][template_type] = copy.deepcopy(DEFAULT_CONFIG[template_type])
            st.session_state[f"template_text_{template_type}"] = "\n".join(DEFAULT_CONFIG[template_type])
            st.success(f"Restored default labels for {template_type}.")
            st.rerun()
    st.caption("Note: template edits are kept in this browser session. They are not shared with other users or permanently stored after the session ends.")
    st.markdown("**Current templates**")
    st.dataframe(
        pd.DataFrame([{"Document type": name, "Reference labels": "\n".join(labels)} for name, labels in st.session_state["templates"].items()]),
        use_container_width=True,
        hide_index=True,
    )

with tab_process:
    uploads = st.file_uploader("Upload TIFF files or a ZIP containing TIFF files", type=["tif", "tiff", "zip"], accept_multiple_files=True)

    if uploads:
        if st.button("Run OCR and detect references", type="primary"):
            items, errors = collect_uploads(uploads, int(max_files), int(max_file_mb))
            st.session_state["upload_items"] = items
            st.session_state["upload_errors"] = errors
            if not items:
                st.error("No usable TIFF files were found in the uploads.")
                st.stop()
            with st.spinner("Running OCR. Large batches may take a few minutes…"):
                df, previews = make_rows(items, doc_type, st.session_state["templates"])
            st.session_state["ocr_df"] = df
            st.session_state["previews"] = previews
            st.session_state["doc_type"] = doc_type

    if st.session_state.get("upload_errors"):
        with st.expander("Upload notes", expanded=False):
            for error in st.session_state["upload_errors"]:
                st.write(f"• {error}")

    if "ocr_df" in st.session_state:
        st.subheader("Review detected references")
        st.info("Edit the ‘Detected reference’ column if OCR has read a reference incorrectly. Blank references will be placed in the Missing folder.")
        edited = st.data_editor(
            st.session_state["ocr_df"],
            use_container_width=True,
            hide_index=True,
            disabled=["Original path", "Status", "OCR text"],
            column_config={
                "Detected reference": st.column_config.TextColumn("Detected reference", help="Edit this value before creating the renamed ZIP."),
                "OCR text": st.column_config.TextColumn("OCR text", width="large"),
            },
            key="review_editor",
        )
        destinations = calculate_destinations(edited, st.session_state.get("doc_type", doc_type))
        preview_df = pd.DataFrame({"Original path": edited["Original path"], "Proposed destination": destinations, "Status": edited["Status"]})
        with st.expander("Preview proposed filenames", expanded=True):
            st.dataframe(preview_df, use_container_width=True, hide_index=True)
        if st.button("Build renamed ZIP", type="primary"):
            try:
                archive_bytes = build_zip(st.session_state["upload_items"], edited, destinations)
                st.session_state["output_zip"] = archive_bytes
                st.session_state["output_name"] = f"NEXUS_{st.session_state.get('doc_type', doc_type)}_Renamed.zip"
            except Exception as exc:
                st.error(f"Could not create ZIP: {exc}")
        if "output_zip" in st.session_state:
            st.download_button(
                "Download renamed TIFF ZIP",
                data=st.session_state["output_zip"],
                file_name=st.session_state.get("output_name", "NEXUS_Renamed.zip"),
                mime="application/zip",
            )
        if st.session_state.get("previews"):
            with st.expander("Preview a TIFF first page", expanded=False):
                chosen = st.selectbox("Choose a file", list(st.session_state["previews"].keys()))
                st.image(st.session_state["previews"][chosen], caption=chosen, use_container_width=True)

st.divider()
st.caption("NEXUS TIFF OCR Renamer • Web prototype. OCR accuracy depends on scan quality and Tesseract language configuration.")
