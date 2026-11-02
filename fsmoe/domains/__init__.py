"""The three domains of the test bed: arithmetic, code, language."""

from typing import Dict

from .arith import ArithDomain
from .base import Domain, Item, table
from .code import CodeDomain
from .lang import LangDomain

NAMES = ("arith", "code", "lang")


def all_domains() -> Dict[str, Domain]:
    return {"arith": ArithDomain(), "code": CodeDomain(), "lang": LangDomain()}


__all__ = ["ArithDomain", "CodeDomain", "LangDomain", "Domain", "Item", "table", "all_domains", "NAMES"]
