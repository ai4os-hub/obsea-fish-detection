import tomllib as toml

import torch
from pydantic_settings import (
    BaseSettings,
    CliSettingsSource,
    PydanticBaseSettingsSource,
)
from rich_argparse import RichHelpFormatter

from obsea import config


class BaseArguments(BaseSettings):
    """Base class for all scripts"""

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Enable CLI formatter_class to work properly."""
        return (
            CliSettingsSource(
                settings_cls,
                formatter_class=RichHelpFormatter,
                cli_parse_args=True,
            ),
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )


def load_config(version: str) -> dict:
    """Load a TOML file and return the content as a dictionary."""
    file_path = config.config_path / f"settings_{version}.toml"
    with open(file_path, "rb") as file:
        return toml.load(file)


def save_model(model, output: str):
    """Save the model to a file."""
    file_path = config.models_path / output
    torch.save(model, file_path)
