"""Offline podcast routing checks; no model downloads or audio dependencies.

The source functions are compiled unchanged without the module's eager model
loading and Gradio launch. Only audio, transcription and synthesis boundaries
are replaced with inert doubles.
"""

import ast
import contextlib
import io
from pathlib import Path
import re
import types
import unittest
from unittest.mock import Mock


SOURCE = Path(__file__).resolve().parents[1] / "gradio_app.py"


def load_functions():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    names = {"split_text_into_batches", "infer", "generate_podcast", "podcast_generation"}
    nodes = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in nodes} == names
    split_words = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "SPLIT_WORDS"
                for target in node.targets)
    )
    module = ast.Module(body=[split_words, *nodes], type_ignores=[])
    namespace = {"re": re}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace


class PodcastTests(unittest.TestCase):
    def setUp(self):
        self.ns = load_functions()
        self.files = {}
        self.file_number = 0
        files = self.files

        class Segment:
            def __init__(self, parts=()):
                self.parts = tuple(parts)

            def __len__(self):
                return 1000

            def __add__(self, other):
                return Segment(self.parts + other.parts)

            def __radd__(self, other):
                if other == 0:
                    return self
                return NotImplemented

            def export(self, path, format):
                assert format == "wav"
                files[path] = self

            @staticmethod
            def from_file(path):
                return Segment((path,))

            @staticmethod
            def from_wav(path):
                return files[path]

            @staticmethod
            def silent(duration):
                return Segment(("pause",)) if duration else Segment()

        @contextlib.contextmanager
        def temporary_file(**kwargs):
            self.file_number += 1
            yield types.SimpleNamespace(name=f"mock-{self.file_number}.wav")

        def write_audio(path, data, sample_rate):
            self.assertEqual(sample_rate, 24000)
            files[path] = Segment(data)

        def synthesize(ref_audio, ref_text, batches, model, remove_silence):
            return ((24000, tuple(batches)), "unused-spectrogram.png")

        self.synthesize = Mock(side_effect=synthesize)
        self.transcribe = Mock(return_value={"text": "Transcribed reference."})
        self.ns.update(
            AudioSegment=Segment,
            silence=types.SimpleNamespace(split_on_silence=lambda segment, **kw: [segment]),
            tempfile=types.SimpleNamespace(NamedTemporaryFile=temporary_file),
            sf=types.SimpleNamespace(write=write_audio),
            gr=types.SimpleNamespace(Info=lambda *args: None, Warning=lambda *args: None),
            pipe=self.transcribe,
            infer_batch=self.synthesize,
        )

    def podcast(self, script, *, model="F5-TTS", remove_silence=True,
                names=("Alice", "Bob"), ref_texts=("Reference A.", "Reference B.")):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.ns["podcast_generation"](
                script, names[0], "reference-a.wav", ref_texts[0],
                names[1], "reference-b.wav", ref_texts[1], model, remove_silence,
            )

    def test_two_speakers_keep_their_reference_and_order(self):
        path = self.podcast("Alice: Hello.\nBob: Goodbye.")
        calls = self.synthesize.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual([call.args[1] for call in calls], ["Reference A.", "Reference B."])
        self.assertEqual([self.files[call.args[0]].parts for call in calls],
                         [("reference-a.wav",), ("reference-b.wav",)])
        self.assertEqual(self.files[path].parts, ("Hello.", "pause", "Goodbye.", "pause"))

    def test_single_second_speaker_uses_selected_model_and_silence_option(self):
        path = self.podcast("Bob: One block.", model="E2-TTS", remove_silence=False)
        self.synthesize.assert_called_once()
        self.assertEqual(self.synthesize.call_args.args[1:], (
            "Reference B.", ["One block."], "E2-TTS", False,
        ))
        self.assertEqual(self.files[path].parts, ("One block.", "pause"))

    def test_missing_reference_text_still_runs_transcription(self):
        self.podcast("Alice: Hello.", ref_texts=("", "Reference B."))
        self.transcribe.assert_called_once()
        self.assertEqual(self.synthesize.call_args.args[1], "Transcribed reference.")

    def test_multiline_blocks_and_literal_speaker_names(self):
        path = self.podcast(
            "A.+: First line.\nSecond line.\nB(2): Last line.",
            names=("A.+", "B(2)"),
        )
        self.assertEqual(self.files[path].parts, (
            "First line.\nSecond line.", "pause", "Last line.", "pause",
        ))

    def test_repeated_generation_keeps_requests_separate(self):
        first = self.podcast("Alice: First.")
        second = self.podcast("Bob: Second.")
        self.assertNotEqual(first, second)
        self.assertEqual(self.files[first].parts, ("First.", "pause"))
        self.assertEqual(self.files[second].parts, ("Second.", "pause"))

    def test_ordinary_synthesis_still_accepts_custom_split_input(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result = self.ns["infer"](
                "reference-a.wav", "Reference A.", "Hello.", "F5-TTS", False, "however",
            )
        self.assertEqual(result[0], (24000, ("Hello.",)))
        self.synthesize.assert_called_once()


if __name__ == "__main__":
    unittest.main()
