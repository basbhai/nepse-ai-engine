#!/usr/bin/env python3
"""
modules/email_mcp_server.py — MCP server exposing Gmail send/read as tools.

Credentials (EMAIL_USER / EMAIL_APP_PASS / EMAIL_RECEIVER) are read from the
project root .env — the same variables helper/notifier.py already uses for
its SMTP-only send_email(). This server adds attachment support to sending,
plus IMAP read access (list_emails / read_email), reusing the same Gmail App
Password for both protocols.

Unlike the DB/ATrad MCP servers, this one is NOT read-only by design —
send_email has an external side effect (an email actually goes out). It is
wired into /claude and /play's allowedTools in trading_core.py, so
send_email hardcodes a block on attaching any .env or .py file (see
_is_blocked_attachment) regardless of caller — the one exfiltration path
that mattered once this server is reachable from those bot commands.

read_email can optionally save attachments to disk (data/email_attachments/,
created on first use) — attachment filenames come from the email itself
(untrusted), so they're sanitized to a bare filename before being written.
"""
import email
import imaplib
import mimetypes
import os
import re
import smtplib
import sys
from email import encoders
from email.header import decode_header
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from mcp.server import MCPServer

EMAIL_USER      = os.getenv("EMAIL_USER", "")
EMAIL_APP_PASS  = os.getenv("EMAIL_APP_PASS", "")
EMAIL_RECEIVERS = [e.strip() for e in os.getenv("EMAIL_RECEIVER", "").split(",") if e.strip()]

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
IMAP_HOST = "imap.gmail.com"
IMAP_PORT = 993

MAX_ATTACHMENT_TOTAL_BYTES = 25 * 1024 * 1024  # Gmail's outgoing combined-attachment cap
ATTACHMENTS_DIR = ROOT / "data" / "email_attachments"

mcp = MCPServer("email")


def _decode_header_value(value: str | None) -> str:
    if not value:
        return ""
    parts = decode_header(value)
    out = ""
    for text, enc in parts:
        out += text.decode(enc or "utf-8", errors="replace") if isinstance(text, bytes) else text
    return out


def _parse_recipients(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(v).strip() for v in value if str(v).strip()]


def _is_blocked_attachment(path: str) -> bool:
    """Hardcoded: never allow attaching a .env file or a .py file — either
    could leak secrets or source code out via email regardless of caller."""
    base = os.path.basename(path)
    return base == ".env" or base.endswith(".env") or base.endswith(".py")


def _attach_files(msg: MIMEMultipart, paths: list[str]) -> None:
    for p in paths:
        path = Path(p)
        if not path.is_file():
            raise FileNotFoundError(f"Attachment not found: {p}")
        ctype, encoding = mimetypes.guess_type(str(path))
        if ctype is None or encoding is not None:
            ctype = "application/octet-stream"
        maintype, subtype = ctype.split("/", 1)
        part = MIMEBase(maintype, subtype)
        part.set_payload(path.read_bytes())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f'attachment; filename="{path.name}"')
        msg.attach(part)


def _safe_attachment_path(base_dir: Path, filename: str) -> Path:
    """Sanitize an (untrusted, email-supplied) filename and avoid collisions."""
    name = re.sub(r"[^\w.\-]", "_", os.path.basename(filename or "attachment"))
    stem, ext = os.path.splitext(name)
    path = base_dir / name
    i = 1
    while path.exists():
        path = base_dir / f"{stem}_{i}{ext}"
        i += 1
    return path


def _imap_connect() -> imaplib.IMAP4_SSL:
    if not EMAIL_USER or not EMAIL_APP_PASS:
        raise RuntimeError("EMAIL_USER/EMAIL_APP_PASS not configured in .env")
    imap = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)
    imap.login(EMAIL_USER, EMAIL_APP_PASS)
    return imap


def _extract_body_and_attachments(
    msg, save_attachments: bool, attachments_dir: Path | None,
) -> tuple[str | None, str | None, list[dict]]:
    body_text: str | None = None
    body_html: str | None = None
    attachments: list[dict] = []

    def _decode_payload(part) -> str:
        payload = part.get_payload(decode=True) or b""
        charset = part.get_content_charset() or "utf-8"
        return payload.decode(charset, errors="replace")

    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_maintype() == "multipart":
                continue
            filename = _decode_header_value(part.get_filename())
            disposition = str(part.get("Content-Disposition") or "")
            if filename:
                payload = part.get_payload(decode=True) or b""
                entry = {
                    "filename": filename,
                    "size": len(payload),
                    "content_type": part.get_content_type(),
                }
                if save_attachments and attachments_dir:
                    attachments_dir.mkdir(parents=True, exist_ok=True)
                    dest = _safe_attachment_path(attachments_dir, filename)
                    dest.write_bytes(payload)
                    entry["saved_path"] = str(dest)
                attachments.append(entry)
            elif part.get_content_type() == "text/plain" and "attachment" not in disposition:
                body_text = (body_text or "") + _decode_payload(part)
            elif part.get_content_type() == "text/html" and "attachment" not in disposition:
                body_html = (body_html or "") + _decode_payload(part)
    else:
        text = _decode_payload(msg)
        if msg.get_content_type() == "text/html":
            body_html = text
        else:
            body_text = text

    return body_text, body_html, attachments


@mcp.tool()
def send_email(
    subject: str,
    body: str,
    to: list[str] | None = None,
    cc: list[str] | None = None,
    attachments: list[str] | None = None,
    html: bool = False,
) -> dict:
    """Send an email via Gmail SMTP, optionally with file attachments.

    `to` defaults to EMAIL_RECEIVER from .env if omitted. `attachments` is a
    list of absolute file paths on this machine to attach (combined size
    capped at 25MB, Gmail's outgoing limit). Set `html=True` to send `body`
    as HTML instead of plain text. Never raises — check the `sent` field.
    """
    if not EMAIL_USER or not EMAIL_APP_PASS:
        return {"sent": False, "error": "EMAIL_USER/EMAIL_APP_PASS not configured in .env"}

    recipients = _parse_recipients(to) or EMAIL_RECEIVERS
    if not recipients:
        return {"sent": False, "error": "No recipients — pass `to` or set EMAIL_RECEIVER in .env"}
    cc_list = _parse_recipients(cc)

    msg = MIMEMultipart()
    msg["From"] = EMAIL_USER
    msg["To"] = ", ".join(recipients)
    if cc_list:
        msg["Cc"] = ", ".join(cc_list)
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "html" if html else "plain"))

    if attachments:
        blocked = [p for p in attachments if _is_blocked_attachment(p)]
        if blocked:
            return {
                "sent": False,
                "error": f"Blocked: .env and .py files may not be emailed as attachments: {blocked}",
            }
        try:
            total_size = sum(Path(p).stat().st_size for p in attachments if Path(p).is_file())
        except OSError as exc:
            return {"sent": False, "error": f"Could not stat attachment: {exc}"}
        if total_size > MAX_ATTACHMENT_TOTAL_BYTES:
            return {
                "sent": False,
                "error": f"Attachments total {total_size / 1e6:.1f}MB, exceeds the "
                         f"{MAX_ATTACHMENT_TOTAL_BYTES // (1024 * 1024)}MB limit",
            }
        try:
            _attach_files(msg, attachments)
        except FileNotFoundError as exc:
            return {"sent": False, "error": str(exc)}

    try:
        server = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_APP_PASS)
        server.sendmail(EMAIL_USER, recipients + cc_list, msg.as_string())
        server.quit()
        return {
            "sent": True,
            "to": recipients,
            "cc": cc_list,
            "attachments": [Path(p).name for p in (attachments or [])],
        }
    except smtplib.SMTPAuthenticationError:
        return {"sent": False, "error": "SMTP authentication failed — check EMAIL_APP_PASS"}
    except Exception as exc:
        return {"sent": False, "error": str(exc)}


@mcp.tool()
def list_emails(
    folder: str = "INBOX",
    limit: int = 20,
    unseen_only: bool = False,
    search: str | None = None,
) -> list[dict] | dict:
    """List recent email headers (newest first) via IMAP — does not mark
    messages as read. `search` is an optional raw IMAP SEARCH criteria
    string (e.g. 'FROM "alice@example.com"', 'SUBJECT "invoice"'), ANDed
    with `unseen_only` if both are given. Returns each message's `uid`
    (pass to read_email), from, subject, date, and unread flag.
    """
    try:
        imap = _imap_connect()
    except Exception as exc:
        return {"error": str(exc)}

    try:
        imap.select(folder, readonly=True)
        criteria_parts = []
        if unseen_only:
            criteria_parts.append("UNSEEN")
        if search:
            criteria_parts.append(search)
        criteria = " ".join(criteria_parts) if criteria_parts else "ALL"

        typ, data = imap.uid("search", None, criteria)
        if typ != "OK":
            return {"error": f"IMAP search failed: {data}"}

        uids = data[0].split()
        uids = uids[-limit:][::-1]  # most recent first

        results = []
        for uid in uids:
            typ, msg_data = imap.uid(
                "fetch", uid, "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)] FLAGS)",
            )
            if typ != "OK" or not msg_data or msg_data[0] is None:
                continue
            raw_headers = msg_data[0][1]
            flags_blob = msg_data[0][0].decode(errors="replace") if isinstance(msg_data[0][0], bytes) else str(msg_data[0][0])
            msg = email.message_from_bytes(raw_headers)
            results.append({
                "uid": uid.decode(),
                "from": _decode_header_value(msg.get("From")),
                "subject": _decode_header_value(msg.get("Subject")),
                "date": msg.get("Date", ""),
                "unread": "\\Seen" not in flags_blob,
            })
        return results
    except Exception as exc:
        return {"error": str(exc)}
    finally:
        try:
            imap.logout()
        except Exception:
            pass


@mcp.tool()
def read_email(
    uid: str,
    folder: str = "INBOX",
    mark_read: bool = False,
    save_attachments: bool = False,
) -> dict:
    """Fetch one email's full content by `uid` (from list_emails). By default
    does NOT mark it as read (uses BODY.PEEK). Set `mark_read=True` to mark
    it seen. Set `save_attachments=True` to write attachment files to
    data/email_attachments/ and include their saved_path; otherwise
    attachments are listed by filename/size/content_type only (no binary
    content is returned inline).
    """
    try:
        imap = _imap_connect()
    except Exception as exc:
        return {"error": str(exc)}

    try:
        imap.select(folder, readonly=not mark_read)
        fetch_cmd = "(RFC822)" if mark_read else "(BODY.PEEK[])"
        typ, msg_data = imap.uid("fetch", uid, fetch_cmd)
        if typ != "OK" or not msg_data or msg_data[0] is None:
            return {"error": f"No message found for uid={uid} in {folder}"}

        raw = msg_data[0][1]
        msg = email.message_from_bytes(raw)
        body_text, body_html, attachments = _extract_body_and_attachments(
            msg, save_attachments, ATTACHMENTS_DIR if save_attachments else None,
        )

        return {
            "uid": uid,
            "from": _decode_header_value(msg.get("From")),
            "to": _decode_header_value(msg.get("To")),
            "subject": _decode_header_value(msg.get("Subject")),
            "date": msg.get("Date", ""),
            "body_text": body_text,
            "body_html": body_html,
            "attachments": attachments,
        }
    except Exception as exc:
        return {"error": str(exc)}
    finally:
        try:
            imap.logout()
        except Exception:
            pass


if __name__ == "__main__":
    mcp.run()
