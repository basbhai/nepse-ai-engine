from dotenv import load_dotenv
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

load_dotenv()

sender = os.getenv("EMAIL_USER")
app_password = os.getenv("EMAIL_APP_PASS")
receivers = os.getenv("EMAIL_RECEIVER").split(",")

# Create email
msg = MIMEMultipart()
msg["From"] = sender
msg["To"] = ", ".join(receivers)
msg["Subject"] = "✅ NEPSE AI Engine - Email Test"

body = """
NEPSE AI Engine - Email Connection Test

Status: Connected Successfully
System: Email notifications working
Time: Phase 1 Complete

You will receive trading signals
and alerts on this email.

--- NEPSE AI Engine ---
"""

msg.attach(MIMEText(body, "plain"))

try:
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(sender, app_password)
    server.sendmail(sender, receivers, msg.as_string())
    server.quit()
    print(f"✅ Test 6 OK - Email sent to {len(receivers)} addresses")
    for r in receivers:
        print(f"   → {r}")
except Exception as e:
    print(f"❌ Email failed: {e}")