import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

SMTP_EMAIL = "aneeshr200418@gmail.com"
SMTP_PASSWORD = "ceqrsqbvaitklmyb"

def send_newsletter(to_list, subject, body):
    msg = MIMEMultipart()
    msg["From"] = SMTP_EMAIL
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "html"))

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(SMTP_EMAIL, SMTP_PASSWORD)

    for addr in to_list:
        msg["To"] = addr
        server.send_message(msg)

    server.quit()
