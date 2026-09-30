"""
Tool argument schemas for the Nepal fintech function-calling project.

Design rules (see /docs/behavior_spec.md for the full per-tool contract):
- All money fields are integer paisa (1 NPR = 100 paisa). Never float.
- Every field the model must fill has a description — this text is what
  the model actually sees at inference time, so it doubles as prompt content.
- extra="forbid" on every model: the model must not invent unlisted args.
- Transfers take a beneficiary NAME. The API layer resolves name -> ID and
  handles "not found" / "ambiguous" errors; the model never sees IDs.
"""

from datetime import date
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, PositiveInt, conint, constr


# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------

class Biller(str, Enum):
    NTC = "NTC"
    NCELL = "Ncell"
    NEA = "NEA"


class Telecom(str, Enum):
    NTC = "NTC"
    NCELL = "Ncell"


class TxnCategory(str, Enum):
    FOOD = "food"
    BILLS = "bills"
    TRANSFER = "transfer"
    SHOPPING = "shopping"
    TOPUP = "topup"
    OTHER = "other"


# NRB-quoted currencies for the exchange-rate tool. INR is quoted per 100
# units by NRB, not per 1 — flagged here so the executor/dataset both know.
CurrencyCode = Literal[
    "USD", "EUR", "GBP", "AUD", "INR", "JPY", "AED", "SAR", "QAR", "KWD", "MYR"
]

NepaliPhone = constr(pattern=r"^(97|98)\d{8}$")
AccountId = constr(pattern=r"^[0-9]{6,16}$")
CardLast4 = constr(pattern=r"^\d{4}$")
Paisa = conint(gt=0)  # strictly positive integer paisa


class ArgModel(BaseModel):
    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Read tools
# ---------------------------------------------------------------------------

class GetBalanceArgs(ArgModel):
    account_id: AccountId = Field(..., description="The user's account ID.")


class GetTransactionsArgs(ArgModel):
    account_id: AccountId = Field(..., description="The user's account ID.")
    start_date: Optional[date] = Field(None, description="Inclusive start date (YYYY-MM-DD).")
    end_date: Optional[date] = Field(None, description="Inclusive end date (YYYY-MM-DD).")
    min_amount_paisa: Optional[PositiveInt] = Field(None, description="Minimum transaction amount, in paisa.")
    max_amount_paisa: Optional[PositiveInt] = Field(None, description="Maximum transaction amount, in paisa.")
    category: Optional[TxnCategory] = Field(None, description="Filter by transaction category.")


class GetExchangeRateArgs(ArgModel):
    currency_code: CurrencyCode = Field(..., description="ISO currency code to get the NPR exchange rate for.")


class GetBillDueArgs(ArgModel):
    biller: Biller = Field(..., description="The billing provider.")
    account_number: AccountId = Field(..., description="The customer's account/subscriber number with the biller.")


class CheckPaymentStatusArgs(ArgModel):
    pidx: constr(min_length=1) = Field(..., description="The Khalti payment identifier (pidx) returned by create_payment_link.")


class ListBeneficiariesArgs(ArgModel):
    account_id: AccountId = Field(..., description="The user's account ID whose saved beneficiaries to list.")


# ---------------------------------------------------------------------------
# Write tools (irreversible ones get a confirmation gate at execution time,
# not in the schema — see behavior_spec.md)
# ---------------------------------------------------------------------------

class TransferMoneyArgs(ArgModel):
    from_account_id: AccountId = Field(..., description="The sender's account ID.")
    beneficiary_name: constr(min_length=1) = Field(..., description="Name of a saved beneficiary to send money to.")
    amount_paisa: Paisa = Field(..., description="Amount to transfer, in paisa (1 NPR = 100 paisa).")
    note: Optional[str] = Field(None, description="Optional note/memo for the transfer.")


class PayBillArgs(ArgModel):
    biller: Biller = Field(..., description="The billing provider.")
    account_number: AccountId = Field(..., description="The customer's account/subscriber number with the biller.")
    amount_paisa: Paisa = Field(..., description="Amount to pay, in paisa.")


class TopupMobileArgs(ArgModel):
    phone_number: NepaliPhone = Field(..., description="10-digit Nepali mobile number, starting with 97 or 98.")
    telecom: Telecom = Field(..., description="The mobile network operator.")
    amount_paisa: Paisa = Field(..., description="Top-up amount, in paisa.")


class CreatePaymentLinkArgs(ArgModel):
    amount_paisa: Paisa = Field(..., description="Amount the payment link should request, in paisa.")
    purpose: constr(min_length=1, max_length=120) = Field(..., description="Short description of what the payment is for.")


class AddBeneficiaryArgs(ArgModel):
    name: constr(min_length=1) = Field(..., description="Full name of the beneficiary.")
    account_number: AccountId = Field(..., description="The beneficiary's bank account number.")
    bank_name: constr(min_length=1) = Field(..., description="Name of the beneficiary's bank.")


class BlockCardArgs(ArgModel):
    card_last4: CardLast4 = Field(..., description="Last 4 digits of the card to block.")
    reason: Optional[str] = Field(None, description="Optional reason for blocking (e.g. 'lost', 'stolen', 'suspicious activity').")


# ---------------------------------------------------------------------------
# Registry: what the model actually sees, in Anthropic/OpenAI tool-schema shape
# ---------------------------------------------------------------------------

TOOLS: dict[str, dict] = {
    "get_balance": {
        "description": "Get the current balance of a specific account.",
        "args_model": GetBalanceArgs,
    },
    "get_transactions": {
        "description": "List a user's past transactions, optionally filtered by date range, amount range, or category.",
        "args_model": GetTransactionsArgs,
    },
    "get_exchange_rate": {
        "description": "Get the current NPR exchange rate for a foreign currency, from Nepal Rastra Bank.",
        "args_model": GetExchangeRateArgs,
    },
    "get_bill_due": {
        "description": "Check the amount due on a utility or telecom bill.",
        "args_model": GetBillDueArgs,
    },
    "check_payment_status": {
        "description": "Check the status of a previously created payment link.",
        "args_model": CheckPaymentStatusArgs,
    },
    "list_beneficiaries": {
        "description": "List the saved transfer beneficiaries for an account.",
        "args_model": ListBeneficiariesArgs,
    },
    "transfer_money": {
        "description": "Transfer money from the user's account to a saved beneficiary, identified by name. Irreversible.",
        "args_model": TransferMoneyArgs,
    },
    "pay_bill": {
        "description": "Pay a utility or telecom bill. Irreversible.",
        "args_model": PayBillArgs,
    },
    "topup_mobile": {
        "description": "Top up a mobile phone balance. Irreversible.",
        "args_model": TopupMobileArgs,
    },
    "create_payment_link": {
        "description": "Create a Khalti payment link to request a payment from someone.",
        "args_model": CreatePaymentLinkArgs,
    },
    "add_beneficiary": {
        "description": "Save a new beneficiary for future transfers.",
        "args_model": AddBeneficiaryArgs,
    },
    "block_card": {
        "description": "Block a card, e.g. if it's lost or stolen. Urgent, irreversible.",
        "args_model": BlockCardArgs,
    },
}


def get_tool_schemas() -> list[dict]:
    """Return tool definitions in {name, description, parameters} shape,
    ready to hand to a model as its available-tools list."""
    return [
        {
            "name": name,
            "description": spec["description"],
            "parameters": spec["args_model"].model_json_schema(),
        }
        for name, spec in TOOLS.items()
    ]

def to_openai_tools(names: Optional[list[str]] = None) -> list[dict]:
    """Return tool defs in OpenAI/Qwen apply_chat_template(tools=...) shape.
    Pass a subset of names to show only some tools + distractors in a prompt
    (keeps sequences short — see project token-budget notes)."""
    selected = names if names is not None else list(TOOLS.keys())
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": TOOLS[name]["description"],
                "parameters": TOOLS[name]["args_model"].model_json_schema(),
            },
        }
        for name in selected
    ]

if __name__ == "__main__":
    import json
    print(json.dumps(get_tool_schemas(), indent=2, default=str))