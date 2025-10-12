import click
from enum import Enum


class ForgeModel(str, Enum):
    V0 = "forge-v0"

    @property
    def repo_id(self) -> str:
        return f"yk0/{self.value}"


class ForgeModelParam(click.ParamType):
    name = "ForgeModel"

    def convert(self, value, param, ctx):
        try:
            return ForgeModel(value)
        except ValueError:
            self.fail(
                f"{value} is not valid. Choose from {[m.value for m in ForgeModel]}"
            )
