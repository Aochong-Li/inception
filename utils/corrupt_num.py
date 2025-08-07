import re
import random
from typing import Set, Dict

_NUMBER_RE = re.compile(
    r'(?:'
    r'\d{1,3}(?:,\d{3})+(?:\.\d+)?'  # 1,234 or 1,234.56
    r'|'                             # ────────
    r'\d+\.\d+'                      # 123.456
    r'|'                             # ────────
    r'\d+\.(?!\d)'                   # 123.  (trailing dot, look-ahead: next char not a digit)
    r'|'                             # ────────
    r'\.\d+'                         # .456
    r'|'                             # ────────
    r'\d+'                           # 123
    r')'
    )

def extract_number(text: str) -> Set[str]:
    """Return the set of all (unique) numeric literals in `text`.
    Handles integers, decimals with either leading or trailing digits,
    """
    return {m.group(0) for m in _NUMBER_RE.finditer(text)}

def perturb_number(num: str, rng: random.Random, max_retries: int = 3) -> str:
    """One-step, human-like numeric typo."""
    sign = ''
    if num[0] in '+-':            # keep explicit sign, if any
        sign, num = num[0], num[1:]
    
    # ---------------------- remove commas ------------------------------------------
    if ',' in num:
        num = num.replace(',', '')
    
    def strip_zero(s: str) -> str:
        return re.sub(r'^0+(?=\d)', '', s) or '0'

    # ---------------------- floats ------------------------------------------
    if '.' in num:
        digits = num.replace('.', '')
        if len(digits) < 2:      # nothing to shuffle, just flip the point
            new_core = num[::-1]  # '1.' -> '.1'
        else:
            pos = rng.randint(1, len(digits) - 1)  # different position
            new_core = digits[:pos] + '.' + digits[pos:]
        new_core = strip_zero(new_core)

    # ---------------------- single digit integers ----------------------------------------
    elif len(num) == 1:
        new_core = rng.choice([d for d in '0123456789' if d != num])

    # ---------------------- multi-digit integers ----------------------------------------
    elif len(num) > 1:
        r = rng.random()
        if r < 1 / 3:  # shuffle
            new_core = ''.join(rng.sample(num, len(num)))
        elif r < 2 / 3:  # delete
            i = rng.randrange(len(num))
            new_core = num[:i] + num[i + 1:]
        else:  # duplicate
            i = rng.randrange(len(num))
            new_core = num[:i + 1] + num[i] + num[i + 1:]
        new_core = strip_zero(new_core)
    
    if new_core == num and max_retries > 0:
        return perturb_number(num, rng, max_retries - 1)

    return sign + new_core
        
def replace_number(text: str, replacement: Dict[str, str]) -> str:
    return _NUMBER_RE.sub(
        lambda m: replacement.get(m.group(0), m.group(0)),
        text
    )