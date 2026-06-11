"""Test loading intents for available languages."""

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import pytest
import yaml
from hassil import Intents, TextSlotList, recognize_best

from home_assistant_intents import get_intents, get_languages

# The intents repo (submodule) lives next to the package. Its test fixtures are
# the source of the example sentences (and their entities/areas/floors) used to
# verify that the new slot-combination sentences survive the conversion to the
# old format that is exported to JSON.
_REPO_DIR = Path(__file__).parent.parent / "intents"
_TESTS_DIR = _REPO_DIR / "tests"

# Sentinel for slot combinations whose area comes from the voice satellite's
# context (context_area: true) rather than from the sentence itself.
CONTEXT_AREA = "__context_area__"


@pytest.mark.parametrize("language", get_languages())
def test_load_intents(language: str) -> None:
    """Test that we can load intents for every supported language."""
    lang_intents = get_intents(language)
    assert lang_intents


def _discover_slot_combination_fixtures() -> List[Tuple[str, str, str]]:
    """Find (language, intent, slot_combination) for every new-format fixture.

    New-format test fixtures live in
    ``intents/tests/<language>/<intent>/<slot_combination>.yaml``.
    """
    fixtures: List[Tuple[str, str, str]] = []
    if not _TESTS_DIR.is_dir():
        return fixtures

    for lang_dir in sorted(p for p in _TESTS_DIR.iterdir() if p.is_dir()):
        for intent_dir in sorted(p for p in lang_dir.iterdir() if p.is_dir()):
            for combo_file in sorted(intent_dir.glob("*.yaml")):
                fixtures.append((lang_dir.name, intent_dir.name, combo_file.stem))

    return fixtures


SLOT_COMBINATION_FIXTURES = _discover_slot_combination_fixtures()


@lru_cache(maxsize=None)
def _packaged_intents(
    language: str,
) -> Tuple[Optional[Intents], FrozenSet[Tuple[str, str]]]:
    """Load the exported (old-format) intents for a language.

    Also returns the set of (intent, slot_combination) pairs that were produced
    from the new format, identified by the ``slot_combination`` metadata that
    ``script/merged_output.py`` writes during conversion.
    """
    intents_dict = get_intents(language)
    if intents_dict is None:
        return None, frozenset()

    active: Set[Tuple[str, str]] = set()
    for intent_name, intent_info in intents_dict.get("intents", {}).items():
        for data in intent_info.get("data", []):
            combo = data.get("metadata", {}).get("slot_combination")
            if combo:
                active.add((intent_name, combo))

    return Intents.from_dict(intents_dict), frozenset(active)


def _slot_lists_from_fixture(fixture: Dict[str, Any]) -> Dict[str, TextSlotList]:
    """Build name/area/floor slot lists from a fixture's own entities.

    Each fixture is self-contained: it declares exactly the entities, areas, and
    floors that its example sentences reference. Mining names per fixture (rather
    than per language) keeps the lists unambiguous — e.g. a "Garage" area in one
    fixture won't cause "open the garage door" in another to match as an area.
    """
    name_tuples = [
        # text in, value out, context, metadata
        (entity["name"], entity["name"], {"domain": entity.get("domain")}, {})
        for entity in fixture.get("entities", [])
        if isinstance(entity, dict) and "name" in entity
    ]

    return {
        "name": TextSlotList.from_tuples(name_tuples, name="name"),
        "area": TextSlotList.from_strings(
            [a["name"] for a in fixture.get("areas", [])], name="area"
        ),
        "floor": TextSlotList.from_strings(
            [f["name"] for f in fixture.get("floors", [])], name="floor"
        ),
    }


@pytest.mark.parametrize("language,intent,combo", SLOT_COMBINATION_FIXTURES)
def test_slot_combination_examples(language: str, intent: str, combo: str) -> None:
    """Each new-format example sentence is recognized as its intent + slots.

    Verifies that the new slot-combination sentences, once converted to the old
    format and exported to JSON, are still recognized with the correct intent,
    slot combination, and slot values.
    """
    intents, active_combinations = _packaged_intents(language)
    if intents is None:
        pytest.skip(f"No exported intents for language: {language}")

    if (intent, combo) not in active_combinations:
        # The intent still uses the old format in the export (the language
        # hasn't finished migrating this intent), so there is nothing new to
        # test here.
        pytest.skip(f"Slot combination not migrated in export: {intent}/{combo}")

    fixture_path = _TESTS_DIR / language / intent / f"{combo}.yaml"
    fixture = yaml.safe_load(fixture_path.read_text(encoding="utf-8"))
    slot_lists = _slot_lists_from_fixture(fixture)

    for test_group in fixture.get("tests", []):
        expected_slots = test_group.get("slots", {})

        for sentence in test_group.get("sentences", []):
            error = f"sentence={sentence!r}, {language}/{intent}/{combo}"

            result = recognize_best(
                sentence,
                intents,
                slot_lists=slot_lists,
                intent_context={"area": CONTEXT_AREA},
                best_slot_name="name",
            )

            assert result is not None, f"Not recognized: {error}"
            assert result.intent.name == intent, f"Wrong intent: {error}"
            assert result.intent_metadata is not None, f"No metadata: {error}"
            assert (
                result.intent_metadata.get("slot_combination") == combo
            ), f"Wrong slot combination: {error}"

            actual_slots = {name: e.value for name, e in result.entities.items()}

            # Area supplied from context is not part of the example's slots.
            if actual_slots.get("area") == CONTEXT_AREA:
                actual_slots.pop("area")

            for slot_name, expected_value in expected_slots.items():
                assert slot_name in actual_slots, f"Missing slot {slot_name}: {error}"
                actual_value = actual_slots[slot_name]

                # Some slots (e.g. an inferred domain) may match several values.
                if isinstance(actual_value, (list, set)):
                    assert (
                        expected_value in actual_value
                    ), f"Slot {slot_name} missing {expected_value!r}: {error}"
                else:
                    assert (
                        actual_value == expected_value
                    ), f"Slot {slot_name} = {actual_value!r}, expected {expected_value!r}: {error}"


# TODO: Need to add support for kw and sr-Latn
# def test_language_scores() -> None:
#     """Test that all supported languages are in language scores."""
#     scores = get_language_scores()
#     lang_families = {lang.split("-", maxsplit=1)[0] for lang in scores}

#     for lang in get_languages():
#         assert (lang in scores) or (lang in lang_families)
