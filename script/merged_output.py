"""Command to generate merged output."""

import argparse
import collections
import json
import logging
from pathlib import Path

import yaml

_LOGGER = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
INTENTS_DIR = ROOT / "intents"

IMPORTANT_INTENTS = {"HassTurnOn", "HassTurnOff"}


def convert_slot_combination_group(
    group: dict, combo_name: str, combo_info: dict
) -> dict:
    """Convert a single new-format data group into the old hassil data format.

    The new slot-combination format keeps domain information in ``name_domains``
    / ``inferred_domain`` and relies on ``intents.yaml`` for ``context_area``.
    The old format expects this to be expressed through ``slots`` and
    ``requires_context`` instead. This mirrors the conversion done in
    ``intents/tests/test_slot_combinations.py``.

    ``name_domains`` may be a string naming a reusable set defined in
    ``intents.yaml`` (``name_domain_groups``); it is resolved to a concrete
    list of domains here so the exported JSON only ever contains lists.
    """
    slots = dict(group.get("slots", {}))
    requires_context = dict(group.get("requires_context", {}))
    metadata = dict(group.get("metadata", {}))

    name_domains = group.get("name_domains")
    inferred_domain = group.get("inferred_domain")
    if name_domains:
        if isinstance(name_domains, str):
            # Named group defined in intents.yaml (name_domain_groups)
            name_domains = combo_info["name_domain_groups"][name_domains]
        # {name} is restricted to entities with one of these domains
        requires_context["domain"] = name_domains
    elif inferred_domain:
        # Domain is inferred from the words in the sentence
        slots["domain"] = inferred_domain

    if combo_info.get("context_area"):
        # Area comes from the voice satellite's context
        requires_context["area"] = {"slot": True}

    # Record the slot combination so consumers (and tests) can identify it
    metadata["slot_combination"] = combo_name

    entry: dict = {"sentences": list(group["sentences"]), "metadata": metadata}
    if slots:
        entry["slots"] = slots
    if requires_context:
        entry["requires_context"] = requires_context
    if "response" in group:
        entry["response"] = group["response"]

    return entry


def convert_slot_combinations(lang_dir: Path, intent_info: dict) -> dict:
    """Convert new-format slot-combination dirs into old-format intent data.

    Returns a mapping of intent name -> {"data": [...]} for every intent that
    has a ``sentences/<language>/<intent>/`` directory.
    """
    converted: dict = {}
    for intent_dir in sorted(p for p in lang_dir.iterdir() if p.is_dir()):
        intent_name = intent_dir.name
        combos = intent_info.get(intent_name, {}).get("slot_combinations", {})

        data = []
        for combo_file in sorted(intent_dir.glob("*.yaml")):
            combo_name = combo_file.stem
            combo_info = combos.get(combo_name, {})
            combo_dict = yaml.safe_load(combo_file.read_text())
            for group in combo_dict.get("data", []):
                if not group.get("sentences"):
                    continue
                data.append(
                    convert_slot_combination_group(group, combo_name, combo_info)
                )

        if data:
            converted[intent_name] = {"data": data}

    return converted


def merge_dict(base_dict, new_dict):
    """Merges new_dict into base_dict."""
    for key, value in new_dict.items():
        if key in base_dict:
            old_value = base_dict[key]
            if isinstance(old_value, collections.abc.MutableMapping):
                # Combine dictionary
                assert isinstance(
                    value, collections.abc.Mapping
                ), f"Not a dict: {value}"
                merge_dict(old_value, value)
            elif isinstance(old_value, collections.abc.MutableSequence):
                # Combine list
                assert isinstance(
                    value, collections.abc.Sequence
                ), f"Not a list: {value}"
                old_value.extend(value)
            else:
                # Overwrite
                base_dict[key] = value
        else:
            base_dict[key] = value


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("target")
    parser.add_argument(
        "--intents-dir", default=INTENTS_DIR, help="Intents repo directory"
    )
    args = parser.parse_args()

    intents_dir = Path(args.intents_dir)
    sentence_dir = intents_dir / "sentences"
    response_dir = intents_dir / "responses"
    lists_dir = intents_dir / "lists"
    rules_dir = intents_dir / "rules"
    intents_file = intents_dir / "intents.yaml"
    languages = sorted(p.name for p in sentence_dir.iterdir() if p.is_dir())

    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)

    intent_info = yaml.safe_load(intents_file.read_text())

    # Skip intents that are not supported in Home Assistant
    supported_intents = set(
        intent for intent, info in intent_info.items() if info.get("supported")
    )

    # Create one JSON file per language
    num_processed_languages = 0
    for language in languages:
        # Merge language's sentence template YAML files
        merged_sentences: dict = {}
        for sentence_file in (sentence_dir / language).glob("*.yaml"):
            merge_dict(merged_sentences, yaml.safe_load(sentence_file.read_text()))

        # Convert new-format slot-combination sentences into the old format.
        # These live in sentences/<language>/<intent>/<slot_combination>.yaml
        # instead of a single sentences/<language>/<...>.yaml file.
        #
        # Migrating an intent means deleting its old-format file(s) and adding a
        # new-format directory. While both exist (a partially-migrated language)
        # the old-format files remain authoritative, so the new directory only
        # takes effect for an intent once its old-format data is gone.
        converted_intents = convert_slot_combinations(
            sentence_dir / language, intent_info
        )
        lang_intent_data = merged_sentences.setdefault("intents", {})
        activated_new_format = False
        for intent_name, intent_dict in converted_intents.items():
            if intent_name in lang_intent_data:
                # Not yet migrated: old-format file(s) still present
                continue
            lang_intent_data[intent_name] = intent_dict
            activated_new_format = True

        if activated_new_format:
            # Migrated intents keep their lists and expansion rules in the
            # dedicated lists/ and rules/ directories instead of _common.yaml.
            # These are authoritative for the new sentences, so they take
            # precedence over anything merged from _common.yaml.
            merged_lists = merged_sentences.setdefault("lists", {})
            for list_file in sorted(lists_dir.glob("*.yaml")):  # shared lists
                merged_lists.update(yaml.safe_load(list_file.read_text())["lists"])
            for list_file in sorted((lists_dir / language).glob("*.yaml")):
                merged_lists.update(yaml.safe_load(list_file.read_text())["lists"])

            merged_rules = merged_sentences.setdefault("expansion_rules", {})
            for rule_file in sorted((rules_dir / language).glob("*.yaml")):
                merged_rules.update(
                    yaml.safe_load(rule_file.read_text())["expansion_rules"]
                )

        # Merge language's response YAML files
        merged_responses: dict = {}
        for response_file in (response_dir / language).glob("*.yaml"):
            merge_dict(merged_responses, yaml.safe_load(response_file.read_text()))

        errors_translated = not any(
            translation.startswith("TODO ")
            for translation in merged_sentences["responses"]["errors"].values()
        )
        if not errors_translated:
            _LOGGER.warning(
                "Skipping language %s because it doesn't have all errors translated",
                language,
            )
            continue

        skip_language = False
        lang_intents: dict = {}
        for intent, info in merged_sentences["intents"].items():
            if intent not in supported_intents:
                continue

            num_intent_sentences = 0
            data = []
            for data_set in info["data"]:
                if len(data_set["sentences"]) > 0:
                    data.append(data_set)
                    num_intent_sentences += len(data_set["sentences"])

            if (num_intent_sentences == 0) and (intent in IMPORTANT_INTENTS):
                skip_language = True
                _LOGGER.warning(
                    "Skipping language %s because it doesn't have sentences for %s",
                    language,
                    intent,
                )
                break

            if not data:
                # No sentence templates
                continue

            lang_intents[intent] = {
                **info,
                "data": data,
            }

        if skip_language:
            # Not usable
            continue

        lang_responses = {
            intent: info
            for intent, info in merged_responses["responses"]["intents"].items()
            if intent in supported_intents
        }

        if not lang_intents and not lang_responses:
            # Nothing to export
            continue

        output: dict = {
            "language": language,
            **merged_sentences,
            "intents": lang_intents,
        }

        if lang_responses:
            # Do this separately because merged_sentences contains error responses
            output.setdefault("responses", {})["intents"] = lang_responses

        # Write as JSON
        target_path = target / f"{language}.json"
        with target_path.open("w", encoding="utf-8") as target_file:
            json.dump(output, target_file, ensure_ascii=False, indent=2)

        num_processed_languages += 1

    num_languages = len(languages)
    if num_processed_languages < num_languages:
        _LOGGER.warning(
            "Skipped %s out of %s language(s)",
            num_languages - num_processed_languages,
            num_languages,
        )


if __name__ == "__main__":
    main()
