from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from maple.function.benchmarking import load_functional_group_taxonomy

ROOT = Path(__file__).resolve().parents[1]
HUB = ROOT / "docs/pretrained-solvation-hub"


def test_accuracy_taxonomy_is_hash_bound_and_contains_structural_groups_only():
    path = HUB / "functional-group-taxonomy-v1.json"
    expected_sha256 = "46e073d0e253d7534b7f516f1cc2096ba0abfea84ddc170c95fc86634c917b36"
    taxonomy = load_functional_group_taxonomy(
        path,
        expected_sha256=expected_sha256,
    )
    watchlist = json.loads(
        (HUB / "research_watchlist.yaml").read_text(encoding="utf-8")
    )
    frozen = watchlist["functional_group_accuracy_taxonomy"]

    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256
    assert frozen["artifact_sha256"] == expected_sha256
    assert frozen["semantic_fingerprint"] == taxonomy.fingerprint
    assert frozen["taxonomy_id"] == taxonomy.taxonomy_id
    assert (
        "derives its experimental value, unit, provenance, source hash and locator "
        "from a reverified JSON artifact" in frozen["record_input_binding_rule"]
    )
    assert "fresh bubblewrap-confined interpreter" in (
        frozen["record_input_binding_rule"]
    )
    assert (
        "passes only a label-free protocol snapshot"
        in frozen["record_input_binding_rule"]
    )
    assert len(taxonomy.group_ids) >= 10
    assert {
        "alcohol",
        "aldehyde",
        "amide",
        "amine",
        "carboxylic_acid",
        "ester",
        "ether",
        "ketone",
        "nitrile",
        "nitro",
    }.issubset(taxonomy.group_ids)
    assert {
        "alkane",
        "aromatic",
        "aromatic_hydrocarbon",
        "heterocyclic",
        "hydrocarbon",
    }.isdisjoint(taxonomy.group_ids)

    chem = pytest.importorskip("rdkit.Chem")
    assert all(
        chem.MolFromSmarts(item.smarts) is not None for item in taxonomy.definitions
    )


def test_anisolv_six_row_model_card_contains_no_accuracy_metrics():
    card = json.loads(
        (ROOT / "maple/function/calculator/model_cards/anisolv-compact.yaml").read_text(
            encoding="utf-8"
        )
    )
    notes = " ".join(card["notes"])

    assert "six-row CPU/default reproduction" in notes
    assert "runtime smoke only" in notes
    assert "does not calculate or report experimental errors" in notes
    assert "No experimental accuracy was calculated" in notes
    assert "MAE 1.341976" not in notes
    assert "RMSE 1.756562" not in notes
    assert "maximum absolute error 3.361167" not in notes


def test_mnsol_reference_preserves_the_frozen_route2_split_without_rows():
    reference = json.loads(
        (HUB / "mnsol-partition-reference.json").read_text(encoding="utf-8")
    )

    assert reference["partitions"]["development"]["record_count"] == 505
    assert reference["partitions"]["development"]["unique_solute_count"] == 312
    assert reference["partitions"]["confirmation"]["record_count"] == 148
    assert reference["partitions"]["confirmation"]["unique_solute_count"] == 83
    assert reference["total"] == {"record_count": 653, "unique_solute_count": 395}
    assert reference["confirmation_policy"]["sealed"] is True
    assert (
        reference["confirmation_policy"]["failed_confirmation_must_not_trigger_tuning"]
        is True
    )
    assert "records" not in reference


def test_watchlist_has_no_executable_placeholder_backends():
    watchlist = json.loads(
        (HUB / "research_watchlist.yaml").read_text(encoding="utf-8")
    )
    statuses = {
        model["model_id"]: model["adapter_status"] for model in watchlist["models"]
    }

    assert statuses["consolv"] == "blocked_by_public_runtime_and_weights"
    assert statuses["twin"] == "blocked_by_public_runtime_and_weights"
    assert statuses["schake-gnn-protein-ism"] == (
        "blocked_by_uninstalled_upstream_runtime_dependencies"
    )
    assert statuses["mace-off24-sc"] == "blocked_by_checkpoint_identity"
    assert statuses["anisolv-compact"] == (
        "sealed_uma_single_point_energy_runtime_verified"
    )
    assert statuses["c3net"] == (
        "property_only_live_runtime_verified_formal_flexisol_"
        "development_accuracy_rejected"
    )
    assert statuses["cigin"] == "property_only_cpu_runtime_verified"
    assert statuses["gnnis-chem-sci-water-2024"] == (
        "screened_separate_upstream_runtime_not_integrated"
    )
    assert statuses["g-nequip-smdw-water"] == (
        "blocked_by_uninstalled_official_nequip_runtime"
    )
    assert statuses["ml-for-charges-pbe0-esp-water"] == (
        "excluded_charge_provider_runtime_blocked_by_uninstalled_xgboost"
    )
    assert statuses["organic-mpnice-mlff-hfe-water"] == (
        "blocked_by_unreleased_official_model_and_incomplete_protocol_runtime"
    )
    assert statuses["abcg2-gaff2-explicit-multisolv"] == (
        "official_runtime_smoke_verified_route1_only"
    )
    assert statuses["fennix-bio1"] == (
        "blocked_by_uninstalled_runtime_missing_paper_bundle_and_unproven_gpu_parity"
    )
    assert statuses["atomicese"] == ("audit_only_dedicated_scalar_candidate_blocked")
    assert (
        statuses["solvbert"] == "blocked_by_required_local_pretraining_and_finetuning"
    )
    assert statuses["moletosolv"] == (
        "blocked_by_missing_pretrained_artifacts_and_required_local_training"
    )
    assert statuses["mlsolv-a"] == "blocked_by_missing_pretrained_checkpoint"
    assert statuses["solprop-ml-gsolv"] == (
        "blocked_by_uninstalled_upstream_runtime_dependencies"
    )
    assert statuses["solprop-mix-exp"] == (
        "blocked_by_uninstalled_upstream_runtime_dependencies"
    )
    datasets = {
        dataset["dataset_id"]: dataset for dataset in watchlist["benchmark_datasets"]
    }
    flexisol = datasets["flexisol"]
    assert flexisol["adapter_status"] == "static_identity_audit_only_not_scored"
    assert flexisol["source_revision"] == ("7b44798f26c888ef541faa6143a813136921483f")
    assert flexisol["audit_artifact_sha256"] == (
        "cf2cd04845390e7b7574232b43207f352cc8e32c58469a799d8f21592bbd5ca4"
    )
    assert flexisol["dgsolv_scope"]["record_count"] == 530
    assert flexisol["dgsolv_scope"]["pure_solvent_count"] == 7
    assert flexisol["strict_final_holdout"] is False
    assert "directml" in flexisol["known_published_registry_overlap"]
    assert "cigin" in flexisol["known_published_registry_overlap"]
    assert watchlist["property_only_baselines"]["registry"] == "benchmark_only"


def test_model_admission_record_requires_a_released_energy_gauge_and_holdout():
    text = (HUB / "MODEL_ADMISSION.md").read_text(encoding="utf-8")
    watchlist = json.loads(
        (HUB / "research_watchlist.yaml").read_text(encoding="utf-8")
    )
    candidates = {model["model_id"]: model for model in watchlist["models"]}

    normalized = " ".join(text.split())
    assert "partition function ratio" in normalized
    assert "C(molecule, S)" in text
    assert "official inference code and already-trained weights" in text
    assert "independent experimental extrapolation panel" in text
    assert "single-point energy difference" in text
    assert "full solvent-component multiset" in text
    assert "against the seven gates above" in text
    assert candidates["consolv"]["availability"] == "paper_only"
    assert candidates["consolv"]["adapter_status"] == (
        "blocked_by_public_runtime_and_weights"
    )
    assert "user-authorized" in candidates["anisolv-compact"]["runtime_prerequisites"]
    assert "upon article publication" in candidates["twin"]["release_evidence"]
    assert "free-energy path" in candidates["consolv"]["free_energy_form"]
    assert "GBn2" in candidates["schake-gnn-protein-ism"]["free_energy_form"]
    assert candidates["schake-gnn-protein-ism"]["checkpoint"]["sha256"].startswith(
        "1be53976"
    )
    assert "not an absolute" in candidates["anisolv-compact"]["free_energy_form"]
    assert "forces" in candidates["anisolv-compact"]["forbidden_tasks"]
    assert "frequency" in candidates["anisolv-compact"]["forbidden_tasks"]
    assert "property prediction" in candidates["c3net"]["free_energy_form"]
    assert candidates["c3net"]["strict_holdout_status"].startswith(
        "not_provable_from_public_release"
    )
    assert (
        "41 training and 10 validation"
        in candidates["c3net"]["training_manifest_status"]
    )
    assert (
        "strict_holdout_claim_from_public_release"
        in candidates["c3net"]["forbidden_tasks"]
    )
    assert "property prediction" in candidates["cigin"]["free_energy_form"]
    assert "multi_solvent_extrapolation" in candidates["cigin"]["forbidden_tasks"]
    assert (
        "no serialized trained estimator"
        in candidates["moletosolv"]["release_evidence"]
    )
    assert "local_training" in candidates["moletosolv"]["forbidden_tasks"]
    assert "MoletoSolv: relevant scalar literature" in text
    water_control = candidates["gnnis-chem-sci-water-2024"]
    assert water_control["checkpoint"]["sha256"].startswith("393917c4")
    g_nequip = candidates["g-nequip-smdw-water"]
    assert g_nequip["checkpoint"]["sha256"].startswith("5615c000")
    assert "independently_gauged" in g_nequip["forbidden_tasks"][3]
    assert water_control["solvent_scope"].startswith("water only")
    assert "runtime_reimplementation" in water_control["forbidden_tasks"]
    assert "H2O_single_point.py" in text
    assert "local_pretraining" in candidates["solvbert"]["forbidden_tasks"]
    assert "checkpoint_reconstruction" in candidates["mlsolv-a"]["forbidden_tasks"]
    assert candidates["solprop-ml-gsolv"]["checkpoint_bundle"][
        "archive_sha256"
    ].startswith("f66bb046")
    solprop_mix = candidates["solprop-mix-exp"]
    assert solprop_mix["checkpoint_bundle"]["archive_sha256"].startswith("670915e5")
    assert solprop_mix["checkpoint_bundle"]["selected_ensemble_member_count"] == 10
    assert "Tap" in solprop_mix["runtime_prerequisites"]
    assert "runtime_reimplementation" in solprop_mix["forbidden_tasks"]
    charge_provider = candidates["ml-for-charges-pbe0-esp-water"]
    assert (
        charge_provider["source_revision"] == "e8407cc7e500d89cf66ed0673d6cd7421ab5637d"
    )
    assert charge_provider["checkpoint"]["size_bytes"] == 91849931
    assert charge_provider["checkpoint"]["source_git_blob_sha1"] == (
        "8d8ae6e35496ed5587685fa3a24c9a598dec2565"
    )
    assert "no solvent identity" in charge_provider["solvent_scope"]
    assert (
        "explicit-solvent alchemical MD protocol" in charge_provider["free_energy_form"]
    )
    organic_mpnice = candidates["organic-mpnice-mlff-hfe-water"]
    assert (
        organic_mpnice["source_revision"] == "872ed7b5f5dcf45696090604383d777acbfd8d70"
    )
    assert "user-supplied TorchScript model" in organic_mpnice["runtime_prerequisites"]
    assert "no Organic_MPNICE" in organic_mpnice["free_energy_form"]
    abcg2 = candidates["abcg2-gaff2-explicit-multisolv"]
    assert abcg2["model_type"].endswith("not an MLIP")
    assert abcg2["artifacts"]["bcc_parameters"]["sha256"].startswith("7412ef2c")
    assert abcg2["artifacts"]["atom_type_definitions"]["sha256"].startswith("7fb77a85")
    assert "byte-identical" in abcg2["runtime_prerequisites"]
    mace_off24_sc = candidates["mace-off24-sc"]
    assert mace_off24_sc["checkpoint"]["status"] == (
        "unreleased_or_unpinned_for_public_reproduction"
    )
    assert mace_off24_sc["protocol_source"]["revision"] == (
        "c3056287622ba18f9b905e9df45affdff46cb147"
    )
    assert mace_off24_sc["supporting_results"]["hydration_csv"]["sha256"].startswith(
        "0b44cb14"
    )
    assert mace_off24_sc["supporting_results"]["supporting_information_pdf"][
        "sha256"
    ].startswith("7c675c66")
    assert "checkpoint_substitution" in mace_off24_sc["forbidden_tasks"]
    fennix = candidates["fennix-bio1"]
    assert fennix["model_distribution"]["revision"] == (
        "83f299b81c1d62e2a15c892280559a7c0cc2fac3"
    )
    assert fennix["runtime_source"]["revision"] == (
        "d62b8740343b803a2b864140ec79e347f8ba034e"
    )
    assert fennix["protocol_runtime_source"]["revision"] == (
        "384bb7d451f85f2ea825dabd888e64ad4448674b"
    )
    assert (
        fennix["protocol_runtime_source"]["gpu_lambda_interface_sha256"]
        == "4437e0b93a55156d554ba13a3d7c974a4c537a0792b15c3ec3cac7d2d8758967"
    )
    assert fennix["checkpoints"]["small"]["sha256"] == (
        "82c570c57e95cf164b1a1b0ac2122133cb435c89b07d495246773091541c2f07"
    )
    assert fennix["checkpoints"]["medium"]["sha256"] == (
        "5aac1aa309a387484b7ee4d61fe0229abba7d4af23845be17e204c1feb34870a"
    )
    assert fennix["training_overlap_status"].startswith("overlap_unknown")
    assert "Maximum absolute error" in fennix["benchmark_status"]
    assert "no-loss parity unproven" in fennix["gpu_precision_status"]
    assert "claiming_multi_solvent_validation" in fennix["forbidden_tasks"]
    assert (
        "gpu_enablement_without_no_loss_reference_parity" in fennix["forbidden_tasks"]
    )
    atomicese = candidates["atomicese"]
    assert atomicese["source_revision"] == ("31e643c7e8974497c78fc2fb6f3c17778d61fa10")
    assert atomicese["source_tree"] == ("e08ce8bfaf73ca9c35c2b85929067e639d7bf048")
    assert "1317 adjustable parameters" in atomicese["paper_evidence"]["architecture"]
    assert atomicese["paper_evidence"]["supporting_information_byte_status"] == (
        "unavailable_to_this_audit_due_http_403"
    )
    assert atomicese["release_audit"]["artifact_sha256"] == (
        "efae3cd8355d6221a3814e6f1a8abdc06adb108d95d084f0589dbe3d0bfa708a"
    )
    assert atomicese["validation_status"]["accuracy_panel"] == "not_run"
    assert atomicese["validation_status"]["gpu_admission"].startswith(
        "false_no_documented_or_verified_packaged_gpu_path"
    )
    assert atomicese["validation_status"]["matched_qm_timing"] == "not_run"
    assert atomicese["validation_status"]["integration"] == "not_performed"
    assert atomicese["validation_status"]["model_card"] == "absent"
    assert atomicese["validation_status"]["adapter"] == "absent"
    assert "Binary sample timing is not speed evidence" in atomicese["runtime_note"]
    assert "strictly greater than 1" in atomicese["admission_sequence"]


def test_hub_document_keeps_free_energy_and_route_boundaries_explicit():
    text = (HUB / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert "not a claim that an MLIP alone computes" in text
    assert "not thermochemical Gibbs energy" in text
    assert "does not duplicate the Route 3 `#solvfe`" in text
    assert "does not add a public `#bindfe`" in text
    assert "never ranked against" in text
    assert "does **not** complete the four scientific" in text
    assert "MD is disabled" in text
    assert "C3NetPropertyAdapter" in text
    assert "CIGIN" in text
    assert "SolProp-mix Exp" in text
    assert "Reproducible G-NequIP supplied-result audit" in text
    assert "Reproducible MACE-OFF24-SC supplied-result audit" in text
    assert "never pooled or ranked as one benchmark" in text
    assert "No-loss GPU acceleration boundary" in text
    assert "rejected before checkpoint resolution" in normalized
    assert "full SHA256" in normalized
    assert "not an opaque approval token" in normalized
    assert "current hardware/runtime fingerprint" in normalized
    assert "cannot choose a looser tolerance" in normalized
    assert "recomputes its digest" in normalized
    assert "non-empty unique record identities" in normalized
    assert "zero-degradation threshold" in normalized
    assert "partial task admission is deliberately forbidden" in normalized
    assert (
        "Product-level forbidden tasks do not waive accelerator parity coverage"
        in normalized
    )
    assert "live fingerprint hook" in normalized
    assert "auto-selects CUDA" in normalized
    assert "FeNNix-Bio1: public GPU/Lambda source, paper FE integration blocked" in text
    assert "v1.0-finetuneIons" in text
    assert "GPU availability or speed alone is not" in normalized
    assert (
        "at least 10 records spanning at least 10 distinct, predeclared functional "
        "groups" in normalized
    )
    assert "signed error, and absolute error" in normalized
    assert "the SHA256 and locator of the exact molecular payload" in normalized
    assert "rather than a separate free-form SMILES" in normalized
    assert "cannot be constructed from result-table metadata alone" in normalized
    assert "runner-owned `VerifiedMolecularInput.read()` handle" in text
    assert "no input source path, receipt" in normalized
    assert "absence of bubblewrap is fail-closed" in normalized
    assert "runtime/interface smoke only" in normalized
    assert "A smaller panel may reject an accelerator" in normalized
    assert "but it can never unlock one" in normalized
    assert "41 training and 10 validation" in text
    assert "ML-for-charges PBE0-ESP" in text
    assert "tracked exclusion, not an implicit-solvent backend" in text
    assert "one to five records of that same solute" in text
    assert (
        "FlexiSol: pinned external-confirmation candidate, not a final holdout" in text
    )
    assert "seven pure solvents" in text
    assert "reads no model predictions" in text
    assert "AtomicESE: audit-only scalar multi-organic-solvent candidate" in text
    assert "1,317 adjustable parameters" in text
    assert "SI ZIP bytes are currently unavailable" in normalized
    assert "Binary sample timing is not speed evidence" in text
    assert "GPU capability remains unverified" in normalized
    assert "GPU admission is false" in normalized
    assert "no MAPLE integration, model card, adapter, accuracy panel" in normalized
    assert (
        "median(QM full-task seconds) / median(AtomicESE full-task seconds) > 1" in text
    )


def test_admission_record_assigns_each_candidate_to_a_scientific_route():
    text = (HUB / "MODEL_ADMISSION.md").read_text(encoding="utf-8")

    assert "## Research sweep and route assignment" in text
    assert "Route 3 `#solvfe` only" in text
    assert "sealed `anisolv-uma` single-point energy control only" in text
    assert "official inference runtime, weights" in text
    assert "### GNNIS original-runtime audit: identity verified" in text
    assert "### GNNIS upstream hydration-helper audit" in text
    assert "fixes `kT = 2.479`" in text
    assert "gas-to-solution standard-state conversion" in text
    assert "one- to\nfive-record same-solute inference path" in text
    assert "GNNImplicitSolvent Chem. Sci. 2024 is a separate water control" in text
    assert "G-NequIP SMD-water: released water checkpoint" in text
    assert "MACE-OFF24-SC: rigorous article protocol, unreleased checkpoint" in text
    assert "never substitute MACE-OFF23-SC" in text
    assert (
        "FeNNix-Bio1: public weights and GPU/Lambda source, execution still blocked"
        in text
    )
    assert "No-loss acceleration" in text
    assert "blocked by gate 7" in text
    assert "not the same identity as the official 36-record" in text
    assert "ML-for-charges PBE0-ESP: public pretrained charge regressor" in text
    assert "Organic_MPNICE plus MLFF_HFE: published protocol" in text
    assert "ABCG2/GAFF2: strong explicit-solvent control" in text
    assert "legacy filenames that are absent" in text
    assert "silent checkpoint substitution" in text
    assert "CIGIN: FreeSolv-trained scalar sidecar" in text
    assert "SolProp_ML Gsolv: authentic multi-solvent model bundle" in text
    assert "SolProp-mix Exp: released mixture dGsolv ensemble" in text
    assert "Schake GNN v2: released protein correction" in text
    assert "AtomicESE: packaged scalar release, audit only" in text
    assert "record-level split membership remain unresolved" in " ".join(text.split())


def test_atomicese_release_audit_is_hash_bound_nonadmission_evidence():
    artifact = HUB / "benchmarks" / "atomicese-release-audit-2026-07-31.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    watchlist = json.loads(
        (HUB / "research_watchlist.yaml").read_text(encoding="utf-8")
    )
    candidate = {model["model_id"]: model for model in watchlist["models"]}["atomicese"]

    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == (
        candidate["release_audit"]["artifact_sha256"]
    )
    assert payload["acceptance_eligible"] is False
    assert payload["capability_boundary"]["scalar_solvation_estimator_only"] is True
    assert payload["capability_boundary"]["mlip"] is False
    boundary = payload["release_boundary"]
    assert boundary["scope"] == "exact_pinned_official_git_tree_only"
    assert boundary["source_code_present_in_pinned_git_tree"] is False
    assert boundary["license_file_present_in_pinned_git_tree"] is False
    assert boundary["checkpoint_identity_ledger_present_in_pinned_git_tree"] is False
    assert (
        boundary["model_training_or_overlap_ledger_present_in_pinned_git_tree"] is False
    )
    supporting_information = boundary[
        "publication_declared_external_supporting_information"
    ]
    assert supporting_information["bytes_audited"] is False
    assert supporting_information["sha256"] is None
    assert supporting_information["locator"].endswith("jcc70104-sup-0001-supinfo.zip")
    compute = payload["packaged_compute_evidence"]
    assert compute["gpu_runtime_verified"] is False
    assert compute["packaged_gpu_capability_status"] == (
        "unverified_no_selected_gpu_or_cuda_ascii_markers"
    )
    assert payload["admission_gates"] == {
        "acceptance_eligible": False,
        "accuracy_admission_eligible": False,
        "gpu_admission_eligible": False,
        "matched_qm_speed_eligible": False,
    }
    assert payload["validation_scope"]["accuracy_validated"] is False
    assert payload["validation_scope"]["matched_qm_benchmark_performed"] is False
    assert payload["validation_scope"]["wall_time_used_for_admission"] is False
    assert (
        payload["runtime"]["sample"]["wall_times_recorded_as_admission_evidence"]
        is False
    )


def test_upstream_artifact_manifest_pins_real_files_and_unknowns():
    payload = json.loads((HUB / "upstream-artifacts.json").read_text(encoding="utf-8"))
    artifacts = payload["artifacts"]
    assert len(artifacts) >= 7
    assert len({item["model_id"] for item in artifacts}) == len(artifacts)
    for item in artifacts:
        assert len(item["source_revision"]) == 40
        assert item["size_bytes"] > 0
        assert len(item["sha256"]) == 64
        int(item["sha256"], 16)
        assert item["artifact_license"]

    by_id = {item["model_id"]: item for item in artifacts}
    assert by_id["aimnet2-cpcms-v2"]["size_bytes"] == 9280334
    assert by_id["gnnis-reference"]["repository_license"] == "MIT-0"
    assert by_id["g-nequip-smdw-water"]["size_bytes"] == 11436716
    assert by_id["g-nequip-smdw-water"]["sha256"].startswith("5615c000")
    assert by_id["g-nequip-smdw-supplied-sfe-results"]["size_bytes"] == 474332
    assert by_id["g-nequip-smdw-supplied-sfe-results"]["sha256"].startswith("76522218")
    assert by_id["anisolv-compact"]["size_bytes"] == 25522563
    assert by_id["anisolv-compact"]["sha256"].startswith("b79be343")
    assert by_id["c3net-checkpoint-1"]["size_bytes"] == 1018418
    assert by_id["c3net-checkpoint-1"]["sha256"].startswith("9d018d4a")
    assert by_id["c3net-public-split-example"]["size_bytes"] == 186
    assert by_id["c3net-public-split-example"]["sha256"].startswith("01bdf4c1")
    assert "not-a-complete" in by_id["c3net-public-split-example"]["scientific_status"]
    assert by_id["cigin"]["size_bytes"] == 4105491
    assert by_id["cigin"]["sha256"].startswith("79f07c64")
    assert by_id["solprop-ml-gsolv-bundle"]["size_bytes"] == 268574239
    assert by_id["solprop-ml-gsolv-bundle"]["sha256"].startswith("f66bb046")
    assert by_id["solprop-mix-exp-bundle"]["size_bytes"] == 288947625
    assert by_id["solprop-mix-exp-bundle"]["sha256"].startswith("670915e5")
    assert by_id["solprop-mix-exp-bundle"]["selected_ensemble_member_count"] == 10
    assert len(by_id["solprop-mix-exp-bundle"]["selected_ensemble_member_sha256"]) == 10
    assert by_id["solprop-mix-exp-bundle"]["selected_ensemble_member_sha256"][
        "model0.pt"
    ].startswith("8206cea9")
    assert by_id["schake-gnn-v2"]["size_bytes"] == 212619
    assert by_id["schake-gnn-v2"]["sha256"].startswith("1be53976")
    assert by_id["mace-off24-medium"]["sha256"].startswith("e5ccf583")
    assert (
        by_id["mace-off24-medium"]["scientific_status"]
        == "adapter-unit-tested-live-runtime-pending"
    )
    assert by_id["mace-off23-sc"]["scientific_status"].endswith(
        "protocol-bridge-pending"
    )
    assert by_id["fennix-bio1-small"]["size_bytes"] == 29772732
    assert by_id["fennix-bio1-small"]["sha256"] == (
        "82c570c57e95cf164b1a1b0ac2122133cb435c89b07d495246773091541c2f07"
    )
    assert by_id["fennix-bio1-medium"]["size_bytes"] == 38124300
    assert by_id["fennix-bio1-medium"]["sha256"] == (
        "5aac1aa309a387484b7ee4d61fe0229abba7d4af23845be17e204c1feb34870a"
    )
    assert by_id["fennix-bio1-medium"]["runtime_source_revision"] == (
        "d62b8740343b803a2b864140ec79e347f8ba034e"
    )
    assert by_id["fennix-bio1-medium"]["artifact_license"].startswith("ASL")
    fennix_tinker = by_id["fennix-bio1-tinker-hp-gpu-lambda-interface"]
    assert fennix_tinker["source_revision"] == (
        "384bb7d451f85f2ea825dabd888e64ad4448674b"
    )
    assert fennix_tinker["size_bytes"] == 16624
    assert fennix_tinker["sha256"] == (
        "4437e0b93a55156d554ba13a3d7c974a4c537a0792b15c3ec3cac7d2d8758967"
    )
    assert fennix_tinker["header_sha256"] == (
        "17c781e21bb900f1fbd6fc32114e9ebe8f184dc80d1a2b9d6aac236029a113b6"
    )
    assert fennix_tinker["fortran_bridge_sha256"] == (
        "7e18fca4ce1e6c75941cbad8d33d1614838740b1a33125c834c53bd4633a9d84"
    )
    assert "no-loss-gpu-parity-blocked" in fennix_tinker["scientific_status"]
    assert "aceff_examples@3c59fb3" in by_id["aceff-2.0"]["unit_evidence"]


def test_c3net_pilot_is_frozen_as_property_only_nonacceptance_evidence():
    payload = json.loads(
        (HUB / "benchmarks" / "c3net-freesolv10-2026-07-30.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["acceptance_eligible"] is False
    assert payload["summary"]["target_quantity"] == "property_prediction"
    assert payload["validation_panel"]["leakage_audit"]["status"] == "overlap_unknown"
    assert len(payload["records"]) == 10
    assert payload["summary"]["accuracy_evaluation_performed"] is False
    assert payload["summary"]["functional_group_coverage"]["observed_count"] == 0
    assert payload["summary"]["functional_group_coverage"]["passes"] is False
    assert payload["summary"]["per_record_errors"] is None
    assert payload["summary"]["metrics"] is None
    assert payload["summary"]["scope"] == "runtime_or_interface_smoke_only"


def test_c3net_multisolvent_pilot_is_runtime_smoke_without_accuracy_claim():
    payload = json.loads(
        (HUB / "benchmarks" / "c3net-dgsolvdb1-11solvent-2026-07-30.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["acceptance_eligible"] is False
    assert payload["summary"]["target_quantity"] == "property_prediction"
    assert payload["validation_panel"]["leakage_audit"]["status"] == "overlap_unknown"
    assert len(payload["records"]) == 22
    assert len(payload["selection"]["solvents"]) == 11
    assert payload["summary"]["coverage"] == 1.0
    assert payload["summary"]["metrics"] is None
    assert payload["summary"]["per_record_errors"] is None
    assert payload["summary"]["accuracy_evaluation_performed"] is False
    assert payload["summary"]["accuracy_metric_reporting_allowed"] is False
    assert payload["summary"]["functional_group_coverage"]["observed_count"] == 0
    assert payload["summary"]["scope"] == "runtime_or_interface_smoke_only"


def test_property_pilot_artifacts_are_hash_bound_in_the_watchlist():
    watchlist = json.loads(
        (HUB / "research_watchlist.yaml").read_text(encoding="utf-8")
    )
    by_id = {model["model_id"]: model for model in watchlist["models"]}

    for model_id in ("c3net", "cigin"):
        for reference in by_id[model_id]["development_pilot_artifacts"]:
            artifact = HUB / reference["artifact"]
            assert artifact.is_file()
            assert hashlib.sha256(artifact.read_bytes()).hexdigest() == (
                reference["artifact_sha256"]
            )


def test_gnequip_supplied_audit_is_static_nonacceptance_evidence():
    payload = json.loads(
        (
            HUB / "benchmarks" / "gnequip-smdw-freesolv-supplied-2026-07-30.json"
        ).read_text(encoding="utf-8")
    )

    assert payload["acceptance_eligible"] is False
    assert payload["evaluation_type"] == (
        "static_upstream_energy_table_audit_no_checkpoint_inference"
    )
    assert payload["validation_panel"]["record_count"] == 388
    assert payload["validation_panel"]["training_overlap_status"] == "overlap_unknown"
    assert payload["quantity_boundary"]["temperature_kelvin"] is None
    assert (
        payload["quantity_boundary"]["sampling_estimator"]
        == "none; supplied single-point energy difference"
    )
    assert payload["source_panel_b_reproduction"]["recomputed_metrics_match"] == {
        "mae_kcal_mol": True,
        "pearson_r": True,
        "rmse_kcal_mol": True,
    }
    metrics = payload["summary"]["metrics"]
    assert round(metrics["mae_kcal_mol"], 4) == 1.0827
    assert round(metrics["rmse_kcal_mol"], 4) == 1.4258
    assert round(metrics["maximum_absolute_error_kcal_mol"], 3) == 5.446


def test_mace_off24_sc_supplied_audit_is_static_nonacceptance_evidence():
    payload = json.loads(
        (
            HUB / "benchmarks" / "mace-off24-sc-supplied-results-2026-07-30.json"
        ).read_text(encoding="utf-8")
    )

    assert payload["acceptance_eligible"] is False
    assert payload["evaluation_type"] == (
        "static_official_supporting_information_audit_no_checkpoint_inference"
    )
    assert payload["checkpoint_identity"]["status"] == (
        "unreleased_or_unpinned_for_public_reproduction"
    )
    assert payload["provenance"]["protocol_source_revision"] == (
        "c3056287622ba18f9b905e9df45affdff46cb147"
    )
    artifacts = payload["provenance"]["artifacts"]
    assert artifacts["hydration_csv"]["size_bytes"] == 2650
    assert artifacts["hydration_csv"]["sha256"].startswith("0b44cb14")
    assert artifacts["supporting_information_pdf"]["size_bytes"] == 1704085
    assert artifacts["supporting_information_pdf"]["sha256"].startswith("7c675c66")
    assert artifacts["hydration_csv"]["license"] == "CC BY-NC 4.0"
    assert "records" not in payload

    hydration = payload["validation_panels"]["hydration"]
    octanol = payload["validation_panels"]["octanol"]
    assert hydration["record_count"] == 36
    assert octanol["record_count"] == 10
    assert hydration["training_overlap_status"] == "overlap_unknown"
    assert octanol["training_overlap_status"] == "overlap_unknown"
    assert hydration["evidence_identity_complete"] is False
    assert octanol["evidence_identity_complete"] is False
    hydration_identity = hydration["evidence_identity"]
    octanol_identity = octanol["evidence_identity"]
    assert hydration_identity["quantity"] == "absolute hydration free energy"
    assert octanol_identity["quantity"] == (
        "absolute solvation free energy in 1-octanol"
    )
    assert hydration_identity["temperature_kelvin"] is None
    assert hydration_identity["standard_state"] is None
    assert "neutrality at pH 7" in hydration_identity["protonation_policy"]
    assert hydration_identity["conformer_policy"].startswith("unknown")
    assert hydration_identity["estimator"] == "MBAR; not rerun by this audit"
    assert hydration_identity["experimental_provenance"]["dataset"] == "FreeSolv"
    assert octanol_identity["experimental_provenance"]["dataset"] == "MNSol"
    assert "records" not in hydration
    assert "records" not in octanol
    assert round(hydration["metrics"]["mae_kcal_mol"], 6) == 0.763889
    assert round(hydration["metrics"]["maximum_absolute_error_kcal_mol"], 1) == 1.6
    assert hydration["maximum_error_record"]["record_name"] == ("methylsulfinylmethane")
    assert round(octanol["metrics"]["mae_kcal_mol"], 3) == 0.474
    assert round(octanol["metrics"]["maximum_absolute_error_kcal_mol"], 2) == 0.93
    assert octanol["maximum_error_record"]["record_name"] == "phthalimide"
    comparison = payload["article_table_1_hydration_comparison"]
    assert comparison["reported_metrics"] == {
        "mae_kcal_mol": 0.69,
        "rmse_kcal_mol": 0.8,
    }
    assert comparison["rounded_csv_recomputed_metrics_match"] == {
        "mae_kcal_mol": False,
        "rmse_kcal_mol": False,
    }
    assert comparison["metric_gaps_recomputed_minus_reported_kcal_mol"] == {
        "mae_kcal_mol": pytest.approx(0.0738888888888888),
        "rmse_kcal_mol": pytest.approx(0.0343893309214564),
    }
    assert (
        comparison["rounding_bounds_kcal_mol"][
            "combined_conservative_display_upper_bound"
        ]
        == 0.03
    )
    assert comparison["rounding_alone_can_explain"] is False
    assert payload["acceptance_gate"]["overall_passes"] is False


def test_cigin_pilot_is_frozen_as_overlap_unknown_property_only_evidence():
    payload = json.loads(
        (HUB / "benchmarks" / "cigin-freesolv10-2026-07-30.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["acceptance_eligible"] is False
    assert payload["summary"]["target_quantity"] == "property_prediction"
    assert payload["validation_panel"]["leakage_audit"]["status"] == "overlap_unknown"
    assert payload["validation_panel"]["training_evidence"]["training_datasets"] == [
        "FreeSolv"
    ]
    assert len(payload["records"]) == 10
    assert payload["summary"]["accuracy_evaluation_performed"] is False
    assert payload["summary"]["functional_group_coverage"]["observed_count"] == 0
    assert payload["summary"]["functional_group_coverage"]["passes"] is False
    assert payload["summary"]["per_record_errors"] is None
    assert payload["summary"]["metrics"] is None
    assert payload["summary"]["scope"] == "runtime_or_interface_smoke_only"
