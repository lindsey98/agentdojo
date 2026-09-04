import datetime
from typing import Annotated

from pydantic import BaseModel, EmailStr, computed_field, model_validator
from typing_extensions import Self

from agentdojo.functions_runtime import Depends
from langchain_core.tools import InjectedToolArg


class EmailStatus:
    sent = "sent"
    received = "received"


class Email(BaseModel):
    id_: str
    sender: EmailStr
    recipients: list[EmailStr]
    subject: str
    body: str
    status: str
    read: bool = False
    timestamp: datetime.datetime


class Inbox(BaseModel):
    account_email: EmailStr
    initial_emails: list[Email]
    emails: dict[str, Email] = {}

    @model_validator(mode="after")
    def _create_emails(self) -> Self:
        self.emails = {email.id_: email for email in self.initial_emails}
        return self

    @computed_field
    @property
    def received(self) -> list[Email]:
        return [email for email in self.emails.values() if email.status == EmailStatus.received]

    @computed_field
    @property
    def sent(self) -> list[Email]:
        return [email for email in self.emails.values() if email.status == EmailStatus.sent]

    def _get_next_id(self) -> str:
        largest_id = max((int(key) for key in self.emails.keys()), default=0)
        return str(largest_id + 1)

    def get_received_emails(self) -> list[Email]:
        """Returns all received emails."""
        return self.received

    def send_email(self, subject: str, body: str, recipient: str) -> Email:
        """Sends an email to the specified recipient."""
        new_email = Email(
            id_=self._get_next_id(),
            sender=self.account_email,
            recipients=[recipient],
            subject=subject,
            body=body,
            status=EmailStatus.sent,
            timestamp=datetime.datetime.now(),
            read=True,
        )
        self.emails[new_email.id_] = new_email
        return new_email


# Tool functions
def get_received_emails(inbox: Annotated[Inbox, Depends("inbox"), InjectedToolArg]) -> list[Email]:
    """Returns all received emails. Each email contains sender, subject, and body."""
    return inbox.get_received_emails()


def send_email(
    inbox: Annotated[Inbox, Depends("inbox"), InjectedToolArg],
    subject: str,
    body: str,
    recipient: str,
) -> Email:
    """Sends an email with the given subject and body to the specified recipient.

    :param subject: The subject of the email.
    :param body: The body of the email.
    :param recipient: The email address of the recipient.
    """
    return inbox.send_email(subject, body, recipient)
