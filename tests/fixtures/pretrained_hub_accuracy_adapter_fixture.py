"""Importable formal-accuracy adapter fixtures for spawned-worker tests."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

from maple.function.benchmarking import AccuracyModelAdapter


class UnitTestAccuracyAdapter(AccuracyModelAdapter):
    accuracy_precision_policy = "float64"

    def __init__(self, prediction_offset, *, nonfinite_record=None):
        self.prediction_offset = prediction_offset
        self.nonfinite_record = nonfinite_record

    def predict(self, *, context):
        if hasattr(context, "record_experimental_references"):
            raise AssertionError("Label-free context exposed experimental references.")
        if hasattr(context, "experimental_provenance"):
            raise AssertionError("Label-free context exposed experimental provenance.")
        if hasattr(context.molecular_input, "receipt"):
            raise AssertionError(
                "Path-bearing molecular-input receipt reached adapter."
            )
        payload = context.molecular_input.read()
        if hashlib.sha256(payload).hexdigest() != (
            context.molecular_input.molecular_input_sha256
        ):
            raise AssertionError("Verified molecular-input payload hash changed.")
        if context.record_id == self.nonfinite_record:
            return None
        index = int(context.record_id.removeprefix("record_"))
        return float(index) + self.prediction_offset


class LabelStealingAccuracyAdapter(AccuracyModelAdapter):
    accuracy_precision_policy = "float64"

    def predict(self, *, context):
        context.molecular_input.read()
        return context.record_experimental_references[0].value


class StackInspectingAccuracyAdapter(AccuracyModelAdapter):
    accuracy_precision_policy = "float64"

    def predict(self, *, context):
        context.molecular_input.read()
        frame = inspect.currentframe()
        try:
            while frame is not None:
                identity = frame.f_locals.get("identity")
                references = getattr(
                    identity,
                    "record_experimental_references",
                    (),
                )
                for reference in references:
                    if reference.record_id == context.record_id:
                        return reference.value
                frame = frame.f_back
        finally:
            del frame
        index = int(context.record_id.removeprefix("record_"))
        return float(index) + 1.0


class HelperDelegatingAccuracyAdapter(AccuracyModelAdapter):
    accuracy_precision_policy = "float64"

    def predict(self, *, context):
        context.molecular_input.read()
        return self._impl(context.record_id)

    def _impl(self, record_id):
        index = int(record_id.removeprefix("record_"))
        return float(index) + 1.0


class FilesystemScanningAccuracyAdapter(AccuracyModelAdapter):
    accuracy_precision_policy = "float64"

    def __init__(self, experimental_reference_path):
        self.experimental_reference_path = experimental_reference_path

    def predict(self, *, context):
        context.molecular_input.read()
        try:
            payload = json.loads(
                Path(self.experimental_reference_path).read_text(encoding="utf-8")
            )
        except FileNotFoundError:
            index = int(context.record_id.removeprefix("record_"))
            return float(index) + 1.0
        return float(payload[context.record_id]["value"])


class MissingPrecisionAccuracyAdapter(AccuracyModelAdapter):
    def predict(self, *, context):
        context.molecular_input.read()
        return 0.0
