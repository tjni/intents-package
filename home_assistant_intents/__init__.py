"""API for home_assistant_intents package."""

import importlib.resources
import json
import os
import typing
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import IO, Any, Callable, Collection, Dict, List, Optional, Set, Tuple


from .languages import LANGUAGES

_PACKAGE = "home_assistant_intents"
_DIR = Path(typing.cast(os.PathLike, importlib.resources.files(_PACKAGE)))
_DATA_DIR = _DIR / "data"
_FUZZY_DIR = _DIR / "fuzzy"


class ErrorKey(str, Enum):
    """Keys for home assistant intent errors."""

    NO_INTENT = "no_intent"
    """Intent was not recognized."""

    HANDLE_ERROR = "handle_error"
    """Unexpected error while handling intent."""

    NO_AREA = "no_area"
    """Area does not exist."""

    NO_FLOOR = "no_floor"
    """Floor does not exist."""

    NO_DOMAIN = "no_domain"
    """No devices exist for a domain."""

    NO_DOMAIN_EXPOSED = "no_domain_exposed"
    """No devices are exposed for a domain."""

    NO_DOMAIN_IN_AREA = "no_domain_in_area"
    """No devices in an area exist for a domain."""

    NO_DOMAIN_IN_AREA_EXPOSED = "no_domain_in_area_exposed"
    """No devices in an area are exposed for a domain."""

    NO_DOMAIN_IN_FLOOR = "no_domain_in_floor"
    """No devices in an floor exist for a domain."""

    NO_DOMAIN_IN_FLOOR_EXPOSED = "no_domain_in_floor_exposed"
    """No devices in an floor are exposed for a domain."""

    NO_DEVICE_CLASS = "no_device_class"
    """No devices of a class exist."""

    NO_DEVICE_CLASS_EXPOSED = "no_device_class_exposed"
    """No devices of a class are exposed."""

    NO_DEVICE_CLASS_IN_AREA = "no_device_class_in_area"
    """No devices of a class exist in an area."""

    NO_DEVICE_CLASS_IN_AREA_EXPOSED = "no_device_class_in_area_exposed"
    """No devices of a class are exposed in an area."""

    NO_DEVICE_CLASS_IN_FLOOR = "no_device_class_in_floor"
    """No devices of a class exist in an floor."""

    NO_DEVICE_CLASS_IN_FLOOR_EXPOSED = "no_device_class_in_floor_exposed"
    """No devices of a class are exposed in an floor."""

    NO_ENTITY = "no_entity"
    """Entity does not exist."""

    NO_ENTITY_EXPOSED = "no_entity_exposed"
    """Entity is not exposed."""

    NO_ENTITY_IN_AREA = "no_entity_in_area"
    """Entity does not exist in area."""

    NO_ENTITY_IN_AREA_EXPOSED = "no_entity_in_area_exposed"
    """Entity in area is not exposed."""

    NO_ENTITY_IN_FLOOR = "no_entity_in_floor"
    """Entity does not exist in floor."""

    NO_ENTITY_IN_FLOOR_EXPOSED = "no_entity_in_floor_exposed"
    """Entity in floor is not exposed."""

    DUPLICATE_ENTITIES = "duplicate_entities"
    """More than one entity matched with the same name."""

    DUPLICATE_ENTITIES_IN_AREA = "duplicate_entities_in_area"
    """More than one entity in an area matched with the same name."""

    DUPLICATE_ENTITIES_IN_FLOOR = "duplicate_entities_in_floor"
    """More than one entity in an floor matched with the same name."""

    FEATURE_NOT_SUPPORTED = "feature_not_supported"
    """Entity does not support a required feature."""

    ENTITY_WRONG_STATE = "entity_wrong_state"
    """Entity is not in the correct state."""

    TIMER_NOT_FOUND = "timer_not_found"
    """No timer matched the provided constraints."""

    MULTIPLE_TIMERS_MATCHED = "multiple_timers_matched"
    """More than one timer targeted for an action matched the constraints."""

    NO_TIMER_SUPPORT = "no_timer_support"
    """Voice satellite does not support timers."""


@dataclass
class LanguageScores:
    """Support scores for a language from 0 (no support) to 3 (full support)."""

    cloud: int
    focused_local: int
    full_local: int


def get_intents(
    language: str,
    json_load: Callable[[IO[str]], Dict[str, Any]] = json.load,
) -> Optional[Dict[str, Any]]:
    """Load intents by language."""
    intents_path = _DATA_DIR / f"{language}.json"
    if not intents_path.exists():
        return None

    with intents_path.open(encoding="utf-8") as intents_file:
        return json_load(intents_file)


def get_languages() -> List[str]:
    """Return a list of available languages."""
    return LANGUAGES


def get_language_scores(
    json_load: Callable[[IO[str]], Dict[str, Any]] = json.load,
) -> Dict[str, LanguageScores]:
    """Get support scores by language."""
    scores_path = _DIR / "language_scores.json"
    if not scores_path.exists():
        return {}

    with scores_path.open(encoding="utf-8") as scores_file:
        scores_dict = json_load(scores_file)
        return {
            lang_key: LanguageScores(
                cloud=lang_scores.get("cloud", 0),
                focused_local=lang_scores.get("focused_local", 0),
                full_local=lang_scores.get("full_local", 0),
            )
            for lang_key, lang_scores in scores_dict.items()
        }


# -----------------------------------------------------------------------------


@dataclass
class FuzzySlotCombinationInfo:
    """Information about a fuzzy slot combination."""

    context_area: bool
    name_domains: Set[str]


@dataclass
class FuzzyConfig:
    """Shared configuration for fuzzy matching."""

    # intent -> (slot, slot) -> slot combo info
    slot_combinations: Dict[str, Dict[Tuple[str, ...], FuzzySlotCombinationInfo]]
    """info for all intent slot combinations."""

    # list name -> [slot names]
    slot_list_names: Dict[str, List[str]]
    """Mapping between list and slot names."""


@dataclass
class FuzzyNgramModel:
    """N-gram model for fuzzy matching."""

    order: int
    """N-gram order."""

    words: Dict[str, int]
    """Words to integer ids."""

    database_path: Path
    """Path to sqlite3 database file."""


FuzzyLanguageResponses = Dict[str, Dict[Tuple[str, ...], Dict[str, str]]]


@dataclass
class FuzzyLanguageInfo:
    """Language specific information for fuzzy matching."""

    language: str

    # domain -> [keywords]
    domain_keywords: Dict[str, List[str]]
    """Keywords that hint at a domain."""

    # intent -> model
    ngram_models: Dict[str, FuzzyNgramModel]
    """N-gram model for each intent."""

    # intent -> (slot, slot) -> domain -> response
    responses: FuzzyLanguageResponses
    """Response code mapping per slot combination."""

    stop_words: Optional[Collection[str]] = None
    """Words that can be ignored if unknown."""


def get_fuzzy_config(
    json_load: Callable[[IO[str]], Dict[str, Any]] = json.load,
) -> FuzzyConfig:
    """Return shared configuration for fuzzy matching."""
    with open(_FUZZY_DIR / "config.json", "r", encoding="utf-8") as config_file:
        config_dict = json_load(config_file)

    return FuzzyConfig(
        slot_combinations={
            intent_name: {
                tuple(sorted(combo_key_str.split())): FuzzySlotCombinationInfo(
                    context_area=combo_info.get("context_area", False),
                    name_domains=set(combo_info.get("name_domains", [])),
                )
                for combo_key_str, combo_info in intent_combos.items()
            }
            for intent_name, intent_combos in config_dict["slot_combinations"].items()
        },
        slot_list_names=config_dict["slot_list_names"],
    )


def get_fuzzy_languages() -> Set[str]:
    """Return languages with fuzzy matching support."""
    return {lang_dir.name for lang_dir in _FUZZY_DIR.iterdir() if lang_dir.is_dir()}


def get_fuzzy_language(
    language: str,
    json_load: Callable[[IO[str]], Dict[str, Any]] = json.load,
) -> Optional[FuzzyLanguageInfo]:
    """Return fuzzy matching information for a language."""
    lang_fuzzy_dir = _FUZZY_DIR / language
    lang_ngram_dir = lang_fuzzy_dir / "ngram"

    if not lang_ngram_dir.is_dir():
        return None

    with open(lang_fuzzy_dir / "config.json", "r", encoding="utf-8") as config_file:
        lang_config = json_load(config_file)

    ngram_models: Dict[str, FuzzyNgramModel] = {}
    for intent_ngram_path in lang_ngram_dir.glob("*.json"):
        intent_db_path = intent_ngram_path.with_suffix(".db")
        if not intent_db_path.exists():
            continue

        with open(intent_ngram_path, "r", encoding="utf-8") as intent_ngram_file:
            intent_ngram_dict = json_load(intent_ngram_file)

        intent_name = intent_ngram_path.stem
        ngram_models[intent_name] = FuzzyNgramModel(
            order=intent_ngram_dict["order"],
            words=intent_ngram_dict["words"],
            database_path=intent_db_path,
        )

    with open(
        lang_fuzzy_dir / "responses.json", "r", encoding="utf-8"
    ) as responses_file:
        lang_responses_dict = json_load(responses_file)
        lang_responses = {
            intent_name: {
                tuple(sorted(combo_key_str.split())): domain_responses
                for combo_key_str, domain_responses in intent_responses.items()
            }
            for intent_name, intent_responses in lang_responses_dict.items()
        }

    return FuzzyLanguageInfo(
        language=language,
        domain_keywords=lang_config["domain_keywords"],
        stop_words=lang_config.get("stop_words"),
        ngram_models=ngram_models,
        responses=lang_responses,
    )
