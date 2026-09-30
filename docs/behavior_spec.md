# Tool Behavior Spec (v2)

Every request has exactly one correct target:
**CALL** (valid tool call), **ASK** (one short clarifying question, no tool call),
**REPLY** (plain text, no tool call: no tool fits, or the request can't be served as asked).

## Context the system prompt provides
- Today's date (resolve "yesterday", "last month", "this week" against it)
- The user's default account (use it unless the user names another account)

## Global rules
1. Never invent a value for a required argument. If it is missing and can't be
   taken from the user's message or the system prompt, ASK.
2. Do NOT ask for anything the system prompt already gives (default account, date).
3. Ask ONE question that covers everything missing. Don't ask piecemeal.
4. Constraints the user states must appear in the call (dates, amounts, category).
   Optional filters the user didn't state are omitted, not guessed.
5. Rupees become integer paisa (Rs. 1 = 100 paisa). Common names map to codes
   ("dollar" = USD, "euro" = EUR, "pound" = GBP).
6. The model never claims knowledge of state it cannot see: saved beneficiaries,
   past payments, balances, whether a link exists. It calls a tool or asks.
7. No tool fits, or the value is unsupported (crypto, invalid phone, unknown
   biller) -> REPLY, say what is unsupported and what the assistant can do.

## Per tool
| Tool | CALL when | ASK when | REPLY when |
|---|---|---|---|
| `get_balance` | Always (default or named account) | Never | No tool fits |
| `get_transactions` | Always; filters optional | Never | -- |
| `get_exchange_rate` | Currency stated or aliased | No currency stated at all | Currency unsupported (e.g. Bitcoin) |
| `get_bill_due` | Biller + account number known | Either is missing | Biller not NTC/Ncell/NEA |
| `check_payment_status` | `pidx` given | `pidx` missing | Asked about payments made outside a payment link |
| `list_beneficiaries` | Always | Never | -- |
| `transfer_money` | Amount + beneficiary name known | Either is missing | -- |
| `pay_bill` | Biller + account number + amount known | Any is missing | -- |
| `topup_mobile` | Valid number + telecom + amount known | Any is missing | Number fails the 97/98 pattern |
| `create_payment_link` | Amount + purpose known | Either is missing | -- |
| `add_beneficiary` | Name + account number + bank known | Any is missing | -- |
| `block_card` | Last 4 digits known | Last 4 digits missing | -- |

## Transfer edge case
If the user says the recipient is not saved, ASK for what `add_beneficiary` needs
(account number, bank). Otherwise CALL `transfer_money` with the name as given.

## Handled by the API layer, NOT the model
- Resolving `beneficiary_name` to an ID; "not found" and "ambiguous name" errors
- Checking balance sufficiency, limits, and authentication
- (Stretch) confirmation step before any money movement