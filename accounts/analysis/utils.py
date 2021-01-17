from django.core.mail import EmailMultiAlternatives
from django.template.loader import render_to_string
from django.utils.html import strip_tags
from ...crm1.settings import EMAIL_HOST_USER

def send_email(data):
    html_content = render_to_string("email_template.html",data)
    text_content = strip_tags(html_content)
    email = EmailMultiAlternatives(
        #subject
        "testing",
        #content
        text_content,
        #from
        EMAIL_HOST_USER,
        #recepient list
        ["emkiarie0@gmail.com"]
    )
    email.attach_alternative(html_content,"text/html")
    email.send()
    return None


