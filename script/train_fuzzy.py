import argparse
import itertools
import json
import logging
import os
import shlex
import shutil
import sqlite3
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from hassil import IntentData, Intents
from hassil.expression import (
    Alternative,
    Expression,
    Group,
    ListReference,
    Permutation,
    RuleReference,
    Sequence,
)
from hassil.fst import intents_to_fst
from hassil.intents import WildcardSlotList
from hassil.ngram import BOS, EOS, MemoryNgramModel
from yaml import safe_load

_LOGGER = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
INTENTS_DIR = ROOT / "intents"

SKIP_INTENTS = [
    "HassGetState",
    "HassRespond",
    "HassBroadcast",
    "HassMediaSearchAndPlay",
    "HassStartTimer",
    "HassCancelTimer",
    "HassCancelAllTimers",
    "HassIncreaseTimer",
    "HassDecreaseTimer",
    "HassPauseTimer",
    "HassUnpauseTimer",
]
NAME_SKIP_INTENTS = (
    "HassStartTimer",
    "HassCancelTimer",
    "HassTimerStatus",
    "HassIncreaseTimer",
    "HassDecreaseTimer",
)

NGRAM_ORDER = 4
DOMAIN_KEYWORDS = {
    "en": {
        "light": ["light", "lights"],
        "fan": ["fan", "fans"],
        "cover": [
            "awning",
            "awnings",
            "blind",
            "blinds",
            "curtain",
            "curtains",
            "door",
            "doors",
            "garage door",
            "gate",
            "gates",
            "shade",
            "shades",
            "shutter",
            "shutters",
            "window",
            "windows",
        ],
    }
}
STOP_WORDS = {
    "en": [
        "the",
        "a",
        "an",
        "please",
        "pls",
        "hey",
        "yo",
        "you",
        "your",
        "yours",
        "yourself",
        "so",
        "then",
        "my",
        "me",
        "this",
        "that",
        "these",
        "those",
        "and",
        "but",
        "about",
        "to",
        "any",
        "some",
        "such",
        "only",
        "too",
        "very",
        "can",
        "just",
        "now",
        "though",
        "although",
        "instead",
        "while",
        "besides",
        "hello",
        "hi",
        "man",
        "ok",
        "alright",
        "allright",
        "lol",
    ]
}


@dataclass
class SpeechTools:
    """Local speech tools."""

    openfst_dir: Path
    opengrm_dir: Path
    _extended_env: Optional[Dict[str, Any]] = None

    @staticmethod
    def from_dir(tools_dir: Union[str, Path]) -> "SpeechTools":
        tools_dir = Path(tools_dir).absolute()
        return SpeechTools(
            openfst_dir=tools_dir / "openfst", opengrm_dir=tools_dir / "opengrm"
        )

    @property
    def extended_env(self) -> Dict[str, str]:
        if self._extended_env is None:
            self._extended_env = os.environ.copy()
            bin_dirs: List[str] = [
                str(self.opengrm_dir / "bin"),
                str(self.openfst_dir / "bin"),
            ]
            lib_dirs: List[str] = [
                str(self.opengrm_dir / "lib"),
                str(self.openfst_dir / "lib"),
            ]

            current_path = self._extended_env.get("PATH")
            if current_path:
                bin_dirs.append(current_path)

            current_lib_path = self._extended_env.get("LD_LIBRARY_PATH")
            if current_lib_path:
                lib_dirs.append(current_lib_path)

            self._extended_env["PATH"] = os.pathsep.join(bin_dirs)
            self._extended_env["LD_LIBRARY_PATH"] = os.pathsep.join(lib_dirs)

        return self._extended_env

    def run(self, program: str, args: List[str], **kwargs):
        if "env" not in kwargs:
            kwargs["env"] = self.extended_env

        if "stderr" not in kwargs:
            kwargs["stderr"] = subprocess.PIPE

        _LOGGER.debug("%s %s", program, args)
        proc = subprocess.Popen(
            [program] + args,
            stdout=subprocess.PIPE,
            **kwargs,
        )
        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            error_text = f"Unexpected error running command {program} {args}"
            if stderr:
                error_text += f": {stderr.decode()}"
            elif stdout:
                error_text += f": {stdout.decode()}"

            raise RuntimeError(error_text)

        return stdout

    def run_pipeline(
        self, *commands: List[str], input_bytes: Optional[bytes] = None, **kwargs
    ) -> bytes:
        if "env" not in kwargs:
            kwargs["env"] = self.extended_env

        if "stderr" not in kwargs:
            kwargs["stderr"] = subprocess.PIPE

        if input is not None:
            kwargs["stdin"] = subprocess.PIPE

        command_str = " | ".join((shlex.join(c) for c in commands))
        _LOGGER.debug(command_str)

        proc = subprocess.Popen(
            command_str, stdout=subprocess.PIPE, shell=True, **kwargs
        )
        stdout, stderr = proc.communicate(input=input_bytes)
        if proc.returncode != 0:
            error_text = f"Unexpected error running command {command_str}"
            if stderr:
                error_text += f": {stderr.decode()}"
            elif stdout:
                error_text += f": {stdout.decode()}"

            raise RuntimeError(error_text)

        return stdout


# -----------------------------------------------------------------------------


def _get_slots(
    e: Expression,
    data: IntentData,
    intents: Intents,
    rule_slot_cache: Dict[str, Any],
) -> Iterable[Tuple[bool, Any]]:
    if isinstance(e, Group):
        grp: Group = e
        if isinstance(grp, (Sequence, Permutation)):
            seq_with_slots: List[List[Iterable[Union[str, None]]]] = [[]]
            for item in grp.items:
                for item_has_slot, item_slots in _get_slots(
                    item, data, intents, rule_slot_cache
                ):
                    if not item_has_slot:
                        continue

                    seq_with_slots[-1].append(item_slots)

                if seq_with_slots[-1]:
                    seq_with_slots.append([])

            if not seq_with_slots[-1]:
                seq_with_slots.pop()

            for slot_combo in itertools.product(*seq_with_slots):
                yield (True, slot_combo)
        elif isinstance(grp, Alternative):
            for item in grp.items:
                for item_has_slot, item_slots in _get_slots(
                    item, data, intents, rule_slot_cache
                ):
                    if not item_has_slot:
                        continue

                    yield (True, item_slots)

            if grp.is_optional:
                yield (True, None)
        else:
            raise ValueError(f"Unexpected group type: {grp}")
    elif isinstance(e, ListReference):
        list_ref: ListReference = e
        yield (True, list_ref.slot_name)
    elif isinstance(e, RuleReference):
        rule_ref: RuleReference = e
        rule_body = data.expansion_rules.get(rule_ref.rule_name)
        from_data = True

        if rule_body is None:
            rule_body = intents.expansion_rules[rule_ref.rule_name]
            from_data = False

        if rule_body is None:
            raise ValueError(f"Missing body for expansion rule: {rule_ref.rule_name}")

        cached_slots = None
        if from_data:
            yield from _get_slots(rule_body.expression, data, intents, rule_slot_cache)
        else:
            # Only cache global rules (from _common.yaml)
            cached_slots = rule_slot_cache.get(rule_ref.rule_name)
            if not cached_slots:
                cached_slots = list(
                    _get_slots(rule_body.expression, data, intents, rule_slot_cache)
                )

            rule_slot_cache[rule_ref.rule_name] = cached_slots
            yield from cached_slots


def _flatten(items: Iterable[Any]) -> Iterable[str]:
    for item in items:
        if item is None:
            continue

        if isinstance(item, str):
            yield item
        else:
            yield from _flatten(item)


# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target")
    parser.add_argument(
        "--intents-dir", default=INTENTS_DIR, help="Intents repo directory"
    )
    parser.add_argument("--tools-dir", required=True)
    parser.add_argument("--language", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    _LOGGER.debug(args)

    target = Path(args.target)
    intents_dir = Path(args.intents_dir)
    sentence_dir = intents_dir / "sentences"
    intents_path = intents_dir / "intents.yaml"

    language = args.language
    tools = SpeechTools.from_dir(args.tools_dir)

    lang_sentence_dir = sentence_dir / language
    lang_intents = Intents.from_files(lang_sentence_dir.glob("*.yaml"))

    lang_fuzzy_dir = target / language
    lang_ngram_dir = lang_fuzzy_dir / "ngram"
    if lang_fuzzy_dir.exists():
        shutil.rmtree(lang_fuzzy_dir)

    lang_fuzzy_dir.mkdir(parents=True, exist_ok=True)
    lang_ngram_dir.mkdir(parents=True, exist_ok=True)

    with open(intents_path, "r", encoding="utf-8") as intents_file:
        intents_info = safe_load(intents_file)

    # TODO: virtual intents for: covers/valves, locks, scripts, scenes
    intent_slot_names = set()
    slot_combinations = defaultdict(lambda: defaultdict(list))
    for intent_name, intent_info in intents_info.items():
        if intent_name in SKIP_INTENTS:
            continue

        intent_slot_names.update(intent_info.get("slots", {}).keys())

        for combo_info in intent_info.get("slot_combinations", {}).values():
            combo_key = tuple(sorted(combo_info["slots"]))

            if ("name" in combo_key) and (intent_name in NAME_SKIP_INTENTS):
                # Ignore names that aren't entities
                continue

            name_domains = combo_info.get("name_domains")
            if name_domains:
                name_domains = set(itertools.chain.from_iterable(name_domains.values()))

            slot_combinations[intent_name][combo_key].append(name_domains)

    intent_slot_list_names = defaultdict(set)
    for intent_name, intent_info in lang_intents.intents.items():
        if intent_name in SKIP_INTENTS:
            continue

        for intent_data in intent_info.data:
            if intent_data.expansion_rules:
                expansion_rules = {
                    **lang_intents.expansion_rules,
                    **intent_data.expansion_rules,
                }
            else:
                expansion_rules = lang_intents.expansion_rules

            for sentence in intent_data.sentences:
                if not isinstance(sentence.expression, Group):
                    continue

                for list_ref in sentence.expression.list_references(expansion_rules):
                    if list_ref.slot_name not in intent_slot_names:
                        continue

                    slot_list = intent_data.slot_lists.get(list_ref.list_name)
                    if slot_list is None:
                        slot_list = lang_intents.slot_lists.get(list_ref.list_name)

                    if (slot_list is None) or (slot_list is WildcardSlotList):
                        continue

                    intent_slot_list_names[list_ref.list_name].add(list_ref.slot_name)

    for rule_body in lang_intents.expansion_rules.values():
        if not isinstance(rule_body.expression, Group):
            continue

        for list_ref in rule_body.expression.list_references(
            lang_intents.expansion_rules
        ):
            if list_ref.slot_name in intent_slot_names:
                intent_slot_list_names[list_ref.list_name].add(list_ref.slot_name)

    with open(target / "config.json", "w", encoding="utf-8") as config_file:
        json.dump(
            {
                "slot_combinations": {
                    intent_name: {
                        " ".join(combo_key): list(
                            itertools.chain.from_iterable(filter(None, name_domains))
                        )
                        for combo_key, name_domains in intent_combos.items()
                    }
                    for intent_name, intent_combos in slot_combinations.items()
                },
                "slot_list_names": {
                    list_name: list(slot_names)
                    for list_name, slot_names in intent_slot_list_names.items()
                },
            },
            config_file,
            ensure_ascii=False,
            indent=4,
        )

    with open(lang_fuzzy_dir / "config.json", "w", encoding="utf-8") as config_file:
        json.dump(
            {
                "domain_keywords": DOMAIN_KEYWORDS[args.language],
                "stop_words": STOP_WORDS.get(args.language, []),
            },
            config_file,
            ensure_ascii=False,
            indent=4,
        )

    # intent -> combo key -> {domain, ...}
    response_overrides = defaultdict(lambda: defaultdict(dict))
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        rule_slot_cache: Dict[str, Any] = {}
        common_path = lang_sentence_dir / "_common.yaml"
        for intent_path in lang_sentence_dir.glob("*.yaml"):
            domain_intent_name = intent_path.stem
            if domain_intent_name.startswith("_"):
                continue

            _intent_domain, intent_name = intent_path.stem.rsplit("_", maxsplit=1)
            if intent_name in SKIP_INTENTS:
                continue

            intents = Intents.from_files([common_path, intent_path])
            intent_info = intents.intents[intent_name]

            # Response overrides
            for intent_data in intent_info.data:
                if not intent_data.response:
                    continue

                name_domains = set()
                inferred_domain = None

                if intent_data.requires_context:
                    context_domain = intent_data.requires_context.get("domain")
                    if isinstance(context_domain, str):
                        name_domains.add(context_domain)
                    elif isinstance(context_domain, list):
                        name_domains.update(context_domain)

                if intent_data.slots:
                    inferred_domain = intent_data.slots.get("domain")

                auto_slots: Set[str] = set()
                if intent_data.slots:
                    # Inferred slots
                    auto_slots.update(intent_data.slots.keys())

                # print(intent_name, name_domains, inferred_domain, intent_data.response)
                for sentence in intent_data.sentences:
                    _LOGGER.debug(sentence.text)
                    for _, combo in _get_slots(
                        sentence.expression, intent_data, lang_intents, rule_slot_cache
                    ):
                        combo_tuple = tuple(
                            sorted(itertools.chain(auto_slots, _flatten(combo)))
                        )

                        combo_override = response_overrides[intent_name][combo_tuple]
                        if name_domains:
                            for domain in name_domains:
                                combo_override[domain] = intent_data.response
                        elif inferred_domain:
                            combo_override[inferred_domain] = intent_data.response
                        else:
                            # No domain
                            combo_override[""] = intent_data.response

                        if (len(combo_tuple) > 1) and (
                            name_domains == {inferred_domain}
                        ):
                            # Reuse responses from {"domain", "name"} for {"name"}
                            no_domain_combo = tuple(
                                slot for slot in combo_tuple if slot != "domain"
                            )
                            response_overrides[intent_name][no_domain_combo].update(
                                combo_override
                            )

            # N-gram model
            intent_fst = intents_to_fst(
                intents, intent_names={intent_name}
            ).remove_spaces()

            # Remove paths with missing parts
            intent_fst.prune()

            if not intent_fst.arcs:
                # Empty FST after pruning
                _LOGGER.warning("Empty FST for %s", intent_path)
                continue

            text_fst_path = temp_dir / f"{domain_intent_name}.fst.txt"
            ngram_fst_path = temp_dir / f"{domain_intent_name}.ngram.fst"
            fst_path = temp_dir / f"{domain_intent_name}.fst"
            words_path = temp_dir / f"{domain_intent_name}.symbols.txt"
            arpa_path = temp_dir / f"{domain_intent_name}.arpa"

            with (
                open(text_fst_path, "w", encoding="utf-8") as fst_file,
                open(words_path, "w", encoding="utf-8") as symbols_file,
            ):
                intent_fst.write(fst_file, symbols_file)

            words: Dict[str, int] = {}
            with open(words_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split(maxsplit=1)
                    words[parts[0]] = int(parts[1])

            for meta_word in (BOS, EOS):
                if meta_word in words:
                    continue

                words[meta_word] = len(words)

            tools.run(
                "fstcompile",
                [
                    shlex.quote(f"--isymbols={words_path}"),
                    shlex.quote(f"--osymbols={words_path}"),
                    "--keep_isymbols=true",
                    "--keep_osymbols=true",
                    shlex.quote(str(text_fst_path)),
                    shlex.quote(str(fst_path)),
                ],
            )

            tools.run_pipeline(
                [
                    "ngramcount",
                    f"--order={NGRAM_ORDER}",
                    shlex.quote(str(fst_path)),
                    "-",
                ],
                [
                    "ngrammake",
                    "--method=kneser_ney",
                    "-",
                    shlex.quote(str(ngram_fst_path)),
                ],
            )
            tools.run_pipeline(
                [
                    "ngramprint",
                    "--ARPA",
                    shlex.quote(str(ngram_fst_path)),
                    shlex.quote(str(arpa_path)),
                ],
            )

            with open(arpa_path, "r", encoding="utf-8") as arpa_file:
                ngram_model = MemoryNgramModel.from_arpa(arpa_file)

            with open(
                lang_ngram_dir / f"{domain_intent_name}.json", "w", encoding="utf-8"
            ) as ngram_config_file:
                json.dump(
                    {
                        "order": ngram_model.order,
                        "words": words,
                    },
                    ngram_config_file,
                    ensure_ascii=False,
                    indent=4,
                )

            ###
            # with open(
            #     lang_ngram_dir / f"{intent_name}.ngram.json", "w", encoding="utf-8"
            # ) as ngram_file:
            #     json.dump(
            #         {
            #             "order": ngram_model.order,
            #             "probs": {
            #                 " ".join(ngram): list(ngram_probs)
            #                 for ngram, ngram_probs in ngram_model.probs.items()
            #             },
            #         },
            #         ngram_file,
            #         ensure_ascii=False,
            #         indent=4,
            #     )
            ###

            db_path = lang_ngram_dir / f"{domain_intent_name}.db"
            db_path.unlink(missing_ok=True)
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    "CREATE TABLE ngrams (word_ids TEXT, log_prob REAL, backoff REAL);"
                )
                conn.execute("CREATE INDEX ngrams_index ON ngrams (word_ids);")
                for ngram, ngram_probs in ngram_model.probs.items():
                    word_ids = " ".join(str(words[word]) for word in ngram)
                    log_prob, backoff = ngram_probs
                    conn.execute(
                        "INSERT INTO ngrams (word_ids, log_prob, backoff) VALUES (?, ?, ?);",
                        (word_ids, log_prob, backoff),
                    )
                conn.commit()
                conn.execute("VACUUM;")

    # Response overrides
    with open(
        lang_fuzzy_dir / "responses.json", "w", encoding="utf-8"
    ) as responses_file:
        json.dump(
            {
                intent_name: {
                    " ".join(combo_key): domain_responses
                    for combo_key, domain_responses in intent_combos.items()
                }
                for intent_name, intent_combos in response_overrides.items()
            },
            responses_file,
            ensure_ascii=False,
            indent=4,
        )


if __name__ == "__main__":
    main()
