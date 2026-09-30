"""Deterministic scenario builder.

CODE decides the correct answer (`target`). The LLM only words the user's message.
Every fact the user must state is listed in `must` and is checked verbatim later.
Nothing here calls an API, so it is fast, free and testable.

Known gap (by design, for now): "top up with a phone number but no operator" is not
generated. Nepali prefixes imply the operator, so asking would be over-asking, and
guessing needs a prefix table we haven't specified yet.
"""
import calendar
import random
import string
from collections import defaultdict, namedtuple
from datetime import date, timedelta

Ctx = namedtuple("Ctx", "today account style")

FIRST = ["Sandesh", "Rita", "Bikash", "Sunita", "Prakash", "Anita", "Sujan", "Puja",
         "Ramesh", "Kamala", "Niraj", "Sabina", "Dipesh", "Manisha", "Aayush", "Sarita",
         "Roshan", "Binita", "Kiran", "Nabin", "Samikshya", "Suman", "Pratik", "Bibek"]
LAST = ["Sharma", "Rai", "Thapa", "Gurung", "Shrestha", "Karki", "Adhikari", "Tamang",
        "Magar", "Basnet", "Poudel", "Khadka", "Limbu", "Bhattarai", "Joshi", "Maharjan"]
BANKS = ["NIC Asia Bank", "Nabil Bank", "Global IME Bank", "Nepal Investment Mega Bank",
         "Siddhartha Bank", "Himalayan Bank", "Machhapuchchhre Bank", "Prabhu Bank",
         "Sanima Bank", "Laxmi Sunrise Bank", "Kumari Bank", "Citizens Bank"]
PURPOSES = ["wedding photography deposit", "tuition fee for October", "momo catering advance",
            "trek guide deposit", "laptop repair", "tailoring advance", "room rent share",
            "birthday cake order", "yoga class fee", "bike service", "website design advance",
            "school uniform"]
STYLES = [("plain", 25), ("casual", 20), ("romanized", 20), ("terse", 15),
          ("formal", 10), ("mixed", 10)]
CATEGORY_WORDS = {"food": "food", "bills": "bills", "transfer": "transfers",
                  "shopping": "shopping", "topup": "top-ups"}
SUPPORTED_FX = "USD, EUR, GBP, AUD, INR, JPY, AED, SAR, QAR, KWD, MYR"
CURRENCY_PHRASES = [
    ("dollar", "USD"), ("dollar", "USD"), ("US dollar", "USD"), ("USD", "USD"), ("USD", "USD"),
    ("euro", "EUR"), ("EUR", "EUR"), ("pound", "GBP"), ("British pound", "GBP"), ("GBP", "GBP"),
    ("Australian dollar", "AUD"), ("Indian rupee", "INR"), ("INR", "INR"), ("yen", "JPY"),
    ("Japanese yen", "JPY"), ("UAE dirham", "AED"), ("dirham", "AED"), ("Saudi riyal", "SAR"),
    ("Qatari riyal", "QAR"), ("Kuwaiti dinar", "KWD"), ("Malaysian ringgit", "MYR"), ("ringgit", "MYR"),
]
UNSUPPORTED_FX = ["Bitcoin", "Ethereum", "Chinese yuan", "Swiss franc", "Canadian dollar",
                  "Singapore dollar", "Thai baht", "South Korean won"]
UNSUPPORTED_BILLERS = ["Worldlink", "Vianet", "Dish Home", "Subisu", "the water (Khanepani) bill"]
BILLER_PHRASES = {"NTC": ["NTC", "Nepal Telecom"], "Ncell": ["Ncell"],
                  "NEA": ["NEA", "electricity", "Nepal Electricity Authority"]}
TELECOM_PHRASES = {"NTC": ["NTC", "Nepal Telecom"], "Ncell": ["Ncell"]}
OOS_TOPICS = ["take out a personal loan", "invest in mutual funds or shares", "open a new savings account",
              "check their credit score", "change their app PIN", "close their account",
              "book a flight ticket", "tell them today's weather", "write an email to their boss",
              "tell them a cricket score"]
OOS_REPLY = ("I'm not able to help with that. I can check balances and transactions, look up "
             "exchange rates, pay bills, top up mobiles, send transfers, create payment links "
             "and block cards.")


# ----------------------------- helpers -----------------------------
def call_t(name, args):
    return {"type": "call", "name": name, "arguments": args}


def ask_t(text):
    return {"type": "ask", "content": text}


def reply_t(text):
    return {"type": "reply", "content": text}


def out(tools, intent, must, target):
    return {"tools": tools, "intent": intent, "must": must, "target": target}


def _rand_digits(rng, n):
    return "".join(rng.choice(string.digits) for _ in range(n))


def acct(rng):
    return str(rng.randint(1, 9)) + _rand_digits(rng, rng.randint(7, 11))  # 8-12 digits


def phone(rng):
    return rng.choice(["97", "98"]) + _rand_digits(rng, 8)


def bad_phone(rng):
    kind = rng.choice(["prefix", "short", "long"])
    if kind == "prefix":
        return rng.choice(["91", "92", "95", "96", "99", "12", "45", "10"]) + _rand_digits(rng, 8)
    if kind == "short":
        return rng.choice(["97", "98"]) + _rand_digits(rng, rng.randint(5, 7))
    return rng.choice(["97", "98"]) + _rand_digits(rng, rng.randint(9, 10))


def pidx(rng):
    while True:
        s = "".join(rng.choice(string.ascii_letters + string.digits) for _ in range(rng.randint(8, 22)))
        if any(ch.isalpha() for ch in s) and any(ch.isdigit() for ch in s):
            return s


def person(rng):
    first = rng.choice(FIRST)
    return first if rng.random() < 0.6 else f"{first} {rng.choice(LAST)}"


def full_name(rng):
    return f"{rng.choice(FIRST)} {rng.choice(LAST)}"


SMALL = [100, 150, 200, 250, 300, 400, 500, 600, 750, 800, 1000, 1200, 1500, 2000, 2500,
         3000, 4000, 5000, 7500, 10000, 15000, 20000, 25000, 50000, 100000]


def money(rng, style, rupees=None):
    """Return (amount_paisa, phrase). `phrase` is what the user will literally say."""
    r = rupees or rng.choice(SMALL)
    forms = [f"Rs. {r}", f"Rs. {r:,}", f"Rs {r}", f"{r} rupees", f"NPR {r}", f"rs{r}"]
    if r % 1000 == 0:
        forms += [f"{r // 1000}k"] * 3
        if style in ("romanized", "mixed") and r < 100000:
            forms += [f"{r // 1000} hajar"] * 2
    if r % 100000 == 0:
        forms += [f"{r // 100000} lakh"] * 3
    return r * 100, rng.choice(forms)


MONTHS = list(calendar.month_name)[1:]


def date_filter(rng, today):
    opts = ["yesterday", "today", "this month", "last month"]
    if today.month > 1:
        opts.append("month")
    kind = rng.choice(opts)
    if kind == "yesterday":
        d = today - timedelta(days=1)
        return "yesterday", d, d
    if kind == "today":
        return "today", today, today
    if kind == "this month":
        return "this month", today.replace(day=1), today
    if kind == "last month":
        end = today.replace(day=1) - timedelta(days=1)
        return "last month", end.replace(day=1), end
    m = rng.randint(1, today.month - 1)
    last = calendar.monthrange(today.year, m)[1]
    return f"in {MONTHS[m - 1]}", date(today.year, m, 1), date(today.year, m, last)


def biller_pick(rng):
    b = rng.choice(list(BILLER_PHRASES))
    return b, rng.choice(BILLER_PHRASES[b])


def telecom_pick(rng):
    t = rng.choice(list(TELECOM_PHRASES))
    return t, rng.choice(TELECOM_PHRASES[t])


# ----------------------------- builders -----------------------------
def balance_default(rng, c):
    return out(["get_balance"], "The user asks for their current account balance. They mention no account number.",
               [], call_t("get_balance", {"account_id": c.account}))


def balance_override(rng, c):
    other = acct(rng)
    while other == c.account:
        other = acct(rng)
    return out(["get_balance"], f"The user asks for the balance of a specific account, number {other}.",
               [other], call_t("get_balance", {"account_id": other}))


def transactions(rng, c):
    args, must, parts = {"account_id": c.account}, [], []
    banned = False

    def add_date():
        phrase, s, e = date_filter(rng, c.today)
        args["start_date"], args["end_date"] = s.isoformat(), e.isoformat()
        must.append(phrase)
        parts.append(f'for the period "{phrase}"')

    if rng.random() > 0.15:
        if rng.random() < 0.7:
            add_date()
        if rng.random() < 0.4:
            cat = rng.choice(list(CATEGORY_WORDS))
            args["category"] = cat
            must.append(CATEGORY_WORDS[cat])
            parts.append(f'only the "{CATEGORY_WORDS[cat]}" category')
        roll = rng.random()
        if roll < 0.15:
            paisa, ph = money(rng, c.style)
            args["min_amount_paisa"] = paisa
            must.append(ph)
            parts.append(f"a minimum amount of {ph} (inclusive)")
            banned = True
        elif roll < 0.27:
            paisa, ph = money(rng, c.style)
            args["max_amount_paisa"] = paisa
            must.append(ph)
            parts.append(f"a maximum amount of {ph} (inclusive)")
            banned = True
        elif roll < 0.35:
            r1 = rng.choice([500, 1000, 2000, 5000])
            r2 = r1 * rng.choice([2, 4, 10])
            p1, ph1 = money(rng, c.style, r1)
            p2, ph2 = money(rng, c.style, r2)
            args["min_amount_paisa"], args["max_amount_paisa"] = p1, p2
            must += [ph1, ph2]
            parts.append(f"amounts between {ph1} (minimum, inclusive) and {ph2} (maximum, inclusive)")
            banned = True
        if not parts:
            add_date()
    intent = "The user asks to see their transactions" + (
        " " + ", ".join(parts) if parts else " (recent ones, no filters)") + ". They mention no account number."
    d = out(["get_transactions"], intent, must, call_t("get_transactions", args))
    d["ban_comparators"] = banned
    return d


def fx_call(rng, c):
    ph, code = rng.choice(CURRENCY_PHRASES)
    return out(["get_exchange_rate"], f'The user asks for the current exchange rate of "{ph}" against the Nepali rupee.',
               [ph], call_t("get_exchange_rate", {"currency_code": code}))


def fx_ask(rng, c):
    text = rng.choice(["Which currency would you like the rate for, for example USD, EUR or INR?",
                       f"Which currency should I look up? I support {SUPPORTED_FX}."])
    return out(["get_exchange_rate"], "The user asks about today's exchange rate but never names any currency.",
               [], ask_t(text))


def fx_reply(rng, c):
    ph = rng.choice(UNSUPPORTED_FX)
    if ph in ("Bitcoin", "Ethereum"):
        text = f"I can only look up NRB-listed fiat currencies ({SUPPORTED_FX}). I can't check cryptocurrency rates."
    else:
        text = f"I can't look up {ph}. I can only check these currencies against NPR: {SUPPORTED_FX}."
    return out(["get_exchange_rate"], f'The user asks for the rate of "{ph}" against the Nepali rupee.',
               [ph], reply_t(text))


def bill_call(rng, c):
    biller, ph_b = biller_pick(rng)
    num = acct(rng)
    return out(["get_bill_due"], f'The user asks how much is due on their "{ph_b}" bill, giving bill account number {num}.',
               [ph_b, num], call_t("get_bill_due", {"biller": biller, "account_number": num}))


def bill_ask_acct(rng, c):
    biller, ph_b = biller_pick(rng)
    text = rng.choice([f"What's your {biller} account/subscriber number?",
                       f"Could you share your {biller} account number?"])
    return out(["get_bill_due"], f'The user asks about their "{ph_b}" bill but gives no account number.',
               [ph_b], ask_t(text))


def bill_ask_biller(rng, c):
    num = acct(rng)
    text = rng.choice(["Which provider is this bill with: NTC, Ncell or NEA?",
                       "Which provider is that bill with: NTC, Ncell or NEA?"])
    return out(["get_bill_due"], f"The user asks how much is due on a bill for account number {num} but never says which provider.",
               [num], ask_t(text))


def bill_ask_both(rng, c):
    return out(["get_bill_due"], "The user asks to check their bill but names no provider and no account number.",
               [], ask_t("Which provider is the bill with (NTC, Ncell or NEA), and what's your account/subscriber number?"))


def bill_reply(rng, c):
    ph = rng.choice(UNSUPPORTED_BILLERS)
    return out(["get_bill_due"], f'The user asks about their "{ph}" bill.', [ph],
               reply_t(f"I can only check bills for NTC, Ncell and NEA, so I can't look up {ph}."))


def status_call(rng, c):
    p = pidx(rng)
    return out(["check_payment_status"], f"The user asks whether the payment with payment ID {p} has been paid.",
               [p], call_t("check_payment_status", {"pidx": p}))


def status_ask(rng, c):
    text = rng.choice(["Which payment link? Could you share its payment ID (pidx)?",
                       "Could you share the payment ID (pidx) of that payment link?"])
    return out(["check_payment_status"], "The user asks whether someone has paid the payment link they created, but gives no payment ID.",
               [], ask_t(text))


def status_reply(rng, c):
    thing = rng.choice(["rent", "salary", "tuition fee", "a bank deposit", "a payment from a client"])
    text = ("I can only check payments made through a payment link, using its payment ID (pidx), "
            "so I can't look that up. If you have a pidx, share it and I'll check.")
    return out(["check_payment_status"], f"The user asks whether their {thing} has been paid or received. It is unrelated to any payment link and no payment ID is given.",
               [], reply_t(text))


def benlist(rng, c):
    return out(["list_beneficiaries"], "The user asks to see their saved beneficiaries. They mention no account number.",
               [], call_t("list_beneficiaries", {"account_id": c.account}))


def transfer_call(rng, c):
    name = person(rng)
    paisa, ph = money(rng, c.style)
    return out(["transfer_money"], f'The user asks to send {ph} to their saved beneficiary "{name}". They mention no account number.',
               [name, ph], call_t("transfer_money", {"from_account_id": c.account, "beneficiary_name": name, "amount_paisa": paisa}))


def transfer_ask_amount(rng, c):
    name = person(rng)
    text = rng.choice([f"How much would you like to send to {name}?", f"How much should I send to {name}?"])
    return out(["transfer_money"], f'The user wants to send money to "{name}" but does NOT say how much.', [name], ask_t(text))


def transfer_ask_name(rng, c):
    _, ph = money(rng, c.style)
    text = rng.choice(["Who would you like to send it to? Please give the beneficiary's name.", "Who should I send it to?"])
    return out(["transfer_money"], f"The user wants to send {ph} but does NOT say to whom.", [ph], ask_t(text))


def transfer_ask_both(rng, c):
    text = rng.choice(["Who would you like to send money to, and how much?",
                       "Sure. Who is the recipient, and how much should I send?"])
    return out(["transfer_money"], "The user says they want to send money but gives neither a recipient nor an amount.", [], ask_t(text))


def transfer_ask_unsaved(rng, c):
    name = person(rng)
    _, ph = money(rng, c.style)
    text = (f"Since {name} isn't saved yet, I'll need to add them as a beneficiary first. "
            "What's their account number and bank name?")
    return out(["transfer_money", "add_beneficiary"],
               f'The user wants to send {ph} to "{name}" and explicitly says this person is not saved as a beneficiary yet.',
               [name, ph], ask_t(text))


def paybill_call(rng, c):
    biller, ph_b = biller_pick(rng)
    num = acct(rng)
    paisa, ph = money(rng, c.style)
    return out(["pay_bill"], f'The user asks to pay their "{ph_b}" bill of {ph}, giving bill account number {num}.',
               [ph_b, num, ph], call_t("pay_bill", {"biller": biller, "account_number": num, "amount_paisa": paisa}))


def paybill_ask_amount(rng, c):
    biller, ph_b = biller_pick(rng)
    num = acct(rng)
    text = rng.choice(["How much would you like to pay? You can also ask me to check the amount due first.",
                       "What amount would you like to pay? I can check the amount due first if you like."])
    return out(["pay_bill", "get_bill_due"], f'The user asks to pay their "{ph_b}" bill, giving bill account number {num}, but does NOT say how much.',
               [ph_b, num], ask_t(text))


def paybill_ask_acct(rng, c):
    biller, ph_b = biller_pick(rng)
    _, ph = money(rng, c.style)
    text = rng.choice([f"What's your {biller} account/subscriber number?", f"Could you share your {biller} account number?"])
    return out(["pay_bill"], f'The user asks to pay {ph} on their "{ph_b}" bill but gives no account number.',
               [ph_b, ph], ask_t(text))


def topup_call(rng, c):
    num = phone(rng)
    tel, ph_t = telecom_pick(rng)
    paisa, ph = money(rng, c.style)
    return out(["topup_mobile"], f'The user asks to top up {ph} on the {ph_t} number {num}.',
               [num, ph_t, ph], call_t("topup_mobile", {"phone_number": num, "telecom": tel, "amount_paisa": paisa}))


def topup_ask_phone(rng, c):
    _, ph = money(rng, c.style)
    if rng.random() < 0.5:
        return out(["topup_mobile"], f"The user wants to top up {ph} on a phone but gives no phone number and no operator.",
                   [ph], ask_t("What's the phone number, and is it NTC or Ncell?"))
    _, ph_t = telecom_pick(rng)
    return out(["topup_mobile"], f"The user wants to top up {ph} on a {ph_t} phone but gives no phone number.",
               [ph, ph_t], ask_t("What's the phone number you'd like to top up?"))


def topup_ask_amount(rng, c):
    num = phone(rng)
    _, ph_t = telecom_pick(rng)
    return out(["topup_mobile"], f"The user wants to top up the {ph_t} number {num} but does NOT say how much.",
               [num, ph_t], ask_t("How much would you like to top up?"))


def topup_reply(rng, c):
    bad = bad_phone(rng)
    _, ph_t = telecom_pick(rng)
    _, ph = money(rng, c.style)
    text = ("That doesn't look like a valid Nepali mobile number. It should be 10 digits starting "
            "with 97 or 98. Could you double check it?")
    return out(["topup_mobile"], f"The user asks to top up {ph} on the {ph_t} number {bad}.",
               [bad, ph_t, ph], reply_t(text))


def paylink_call(rng, c):
    paisa, ph = money(rng, c.style)
    purpose = rng.choice(PURPOSES)
    return out(["create_payment_link"], f'The user asks for a payment link of {ph} for the purpose "{purpose}".',
               [ph, purpose], call_t("create_payment_link", {"amount_paisa": paisa, "purpose": purpose}))


def paylink_ask_amount(rng, c):
    purpose = rng.choice(PURPOSES)
    return out(["create_payment_link"], f'The user asks for a payment link for "{purpose}" but does NOT say the amount.',
               [purpose], ask_t("How much should the payment link request?"))


def paylink_ask_purpose(rng, c):
    _, ph = money(rng, c.style)
    return out(["create_payment_link"], f"The user asks for a payment link of {ph} but does NOT say what it is for.",
               [ph], ask_t("What's the payment link for?"))


def paylink_ask_both(rng, c):
    return out(["create_payment_link"], "The user asks for a payment link but gives neither an amount nor a purpose.",
               [], ask_t("How much should the payment link request, and what's it for?"))


def addben_call(rng, c):
    name, num, bank = full_name(rng), acct(rng), rng.choice(BANKS)
    return out(["add_beneficiary"], f'The user asks to save "{name}" as a beneficiary with account number {num} at {bank}.',
               [name, num, bank], call_t("add_beneficiary", {"name": name, "account_number": num, "bank_name": bank}))


def addben_ask(rng, c):
    name = person(rng)
    kind = rng.choice(["both", "bank", "acct"])
    if kind == "both":
        return out(["add_beneficiary"], f'The user wants to save "{name}" as a new beneficiary but gives neither account number nor bank.',
                   [name], ask_t(f"What's {name}'s account number and bank name?"))
    if kind == "bank":
        num = acct(rng)
        return out(["add_beneficiary"], f'The user wants to save "{name}" (account number {num}) as a beneficiary but does not say the bank.',
                   [name, num], ask_t(f"Which bank is {name}'s account with?"))
    bank = rng.choice(BANKS)
    return out(["add_beneficiary"], f'The user wants to save "{name}" (bank: {bank}) as a beneficiary but gives no account number.',
               [name, bank], ask_t(f"What's {name}'s account number?"))


def block_call(rng, c):
    last4 = _rand_digits(rng, 4)
    args, must = {"card_last4": last4}, [last4]
    if rng.random() < 0.75:
        reason = rng.choice(["lost", "stolen"])
        args["reason"] = reason
        must.append(reason)
        intent = f"The user asks to block their card ending {last4} and says it was {reason}."
    else:
        intent = f"The user asks to block their card ending {last4}, giving no reason."
    return out(["block_card"], intent, must, call_t("block_card", args))


def block_ask(rng, c):
    why = rng.choice(["lost", "stolen", "used by someone else without permission"])
    text = rng.choice(["Which card? Could you give me the last 4 digits?",
                       "Which card should I block? Please share its last 4 digits."])
    return out(["block_card"], f"The user wants to block their card because it was {why}, but gives no card digits.",
               [], ask_t(text))


def greeting(rng, c):
    text = rng.choice(["Hello! I can help with balances, transactions, bills, mobile top-ups, transfers, payment links and blocking cards. What do you need?",
                       "Hi there! What can I help you with today?"])
    return out([], "The user only greets the assistant (hello / hi / namaste) with no request.", [], reply_t(text))


def thanks(rng, c):
    text = rng.choice(["You're welcome! Let me know if you need anything else.",
                       "Happy to help! Anything else I can do for you?"])
    return out([], "The user thanks the assistant or says goodbye and needs nothing else.", [], reply_t(text))


def out_of_scope(rng, c):
    topic = rng.choice(OOS_TOPICS)
    return out([], f"The user asks the assistant to {topic}.", [], reply_t(OOS_REPLY))


GROUPS = [
    ("balance_default", 3, balance_default), ("balance_override", 1.5, balance_override),
    ("transactions", 5, transactions),
    ("fx_call", 3, fx_call), ("fx_ask", 1, fx_ask), ("fx_reply", 1.5, fx_reply),
    ("bill_call", 2.5, bill_call), ("bill_ask_acct", 1.5, bill_ask_acct),
    ("bill_ask_biller", 1, bill_ask_biller), ("bill_ask_both", 0.7, bill_ask_both),
    ("bill_reply", 1, bill_reply),
    ("status_call", 2, status_call), ("status_ask", 1.5, status_ask), ("status_reply", 1.5, status_reply),
    ("benlist", 2, benlist),
    ("transfer_call", 4, transfer_call), ("transfer_ask_amount", 1.5, transfer_ask_amount),
    ("transfer_ask_name", 1.2, transfer_ask_name), ("transfer_ask_both", 0.8, transfer_ask_both),
    ("transfer_ask_unsaved", 1.2, transfer_ask_unsaved),
    ("paybill_call", 3, paybill_call), ("paybill_ask_amount", 1.5, paybill_ask_amount),
    ("paybill_ask_acct", 1, paybill_ask_acct),
    ("topup_call", 3, topup_call), ("topup_ask_phone", 1.2, topup_ask_phone),
    ("topup_ask_amount", 1.2, topup_ask_amount), ("topup_reply", 1.5, topup_reply),
    ("paylink_call", 2.5, paylink_call), ("paylink_ask_amount", 1, paylink_ask_amount),
    ("paylink_ask_purpose", 1, paylink_ask_purpose), ("paylink_ask_both", 0.6, paylink_ask_both),
    ("addben_call", 2.5, addben_call), ("addben_ask", 2, addben_ask),
    ("block_call", 2.5, block_call), ("block_ask", 1.5, block_ask),
    ("greeting", 1, greeting), ("thanks", 0.8, thanks), ("out_of_scope", 3.5, out_of_scope),
]


def build_scenarios(n=720, seed=7):
    rng = random.Random(seed)
    names = [g[0] for g in GROUPS]
    weights = [g[1] for g in GROUPS]
    fns = {g[0]: g[2] for g in GROUPS}
    style_names = [s for s, _ in STYLES]
    style_w = [w for _, w in STYLES]

    scenarios = []
    for i, g in enumerate(rng.choices(names, weights=weights, k=n)):
        today = date(2026, 1, 5) + timedelta(days=rng.randint(0, 357))
        ctx = Ctx(today, acct(rng), rng.choices(style_names, style_w)[0])
        s = fns[g](rng, ctx)
        s.update(sid=f"s{i:04d}", group=g, kind=s["target"]["type"], style=ctx.style,
                 today=today.isoformat(), account=ctx.account)
        s.setdefault("ban_comparators", False)
        scenarios.append(s)

    # Split by SCENARIO, stratified per group: ~70% train / 10% val / 20% test.
    by_group = defaultdict(list)
    for s in scenarios:
        by_group[s["group"]].append(s)
    for items in by_group.values():
        rng.shuffle(items)
        pattern = ["train"] * 14 + ["val"] * 2 + ["test"] * 4
        block = pattern[:]
        for j, s in enumerate(items):
            if j % 20 == 0:
                block = pattern[:]
                rng.shuffle(block)
            s["split"] = block[j % 20]
    return scenarios