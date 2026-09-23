"""Reproduce CHA polar algebra against a pinned local native source in scratch.

No upstream source or executable is stored in the repository.  The generated
GPL Fortran bodies and compiler products stay in the caller's scratch directory.
This is an algebra oracle, not a complete force or hydration-accuracy result.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

SCRIPT = Path(__file__).resolve()
BENCHMARK_DIR = SCRIPT.parent
DEFAULT_SOURCE_MANIFEST = (
    SCRIPT.parents[3] / "tests/solvation/data/chagb_at26_source_manifest_v1.json"
)
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))
import benchmark_core as core

SOURCE_HASHES = {
    "egb.F90": "916de11ba2d18219c38a467b83eabbbbe4ae574ab4c8c333e9f90f3f1f1200fa",
    "gb_read.F90": "0cf1acedb5560a5d170a98c8609e9b4043c31082e5c96a2e3d583b28ef25ddea",
    "pb_init.F90": "c951aff3a6ecd1112b8706d486dceb3d112b435bc5b4754b8dc814af2644398b",
    "sa_driver.F90": "d166afa7f48e0f150c0841d513c15fcf9c0f8c6a45c48c1a35b9d4ade6cdea2f",
    "pb_constants.h": "22aced9d9bd44f11cab18d0625a94da4e2928599268b9095efa09ccacda95c1b",
}
INCLUDE_HASHES = {
    "assert.fh": "78c6452ce3291b57fcf853b21f7e371c30a8f46257772d41a51ecb9bb4923d7a",
    "database.fh": "6731ee79c3e5dd5295712047a50c9eefa149666ead08a5e492075ddf3eae0c71",
    "dprec.fh": "3a2de977b7872cfc4c5fe99bb0e2e13c0985a7b6d1968c7549437a7466c28d0f",
    "md.h": "8dbce7617884c6424d48d36f1b8fdb5020bedfeea78a3c14abf4aa465641c3e3",
    "memory.fh": "3ce3d115f84d4ff74d2f5fbda0be77e585d71a2862aaa5675a427f2f11739cad",
    "memory.h": "eb84b544fbfa9b9b71d47499bb8d75253c554a959682027a785071a99abb814f",
}
SOURCE_MANIFEST_HASH = (
    "fea10fa41adb6a5b36bd09d48bf74e3efb6d9f82a39025e7caa3a9f9c385a65c"
)
PINNED_FULL_REFERENCE_EGB_KCAL_MOL = -6.282267527560081
OBJECTS = (
    "gbnsr6 pb_read gb_read pb_write getcoor runmd runmin force pb_init "
    "sa_driver np_force variable_module pb_exmol NSR6routines pb_list timer "
    "egb ene locmem myopen rdparm decomp rgroup rfree debug svdcmp svbksb "
    "parms pythag memory_module gen_dx_file"
).split()
BRIDGE_TOLERANCE_KCAL_MOL = 1.0e-9


def _verify_sources(source_root: Path, manifest_path: Path) -> None:
    if core.sha256_file(manifest_path) != SOURCE_MANIFEST_HASH:
        raise ValueError("Native source manifest SHA256 mismatch.")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema_version") != 1
        or len(manifest.get("source_files", {})) != 64
    ):
        raise ValueError("Native source manifest is incomplete.")
    expected_paths = {f"src/gbnsr6/{name}" for name in manifest["source_files"]} | {
        f"src/include/{name}" for name in INCLUDE_HASHES
    }
    paths = list(source_root.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("Native source tree contains a symlink.")
    actual_paths = {
        path.relative_to(source_root).as_posix() for path in paths if path.is_file()
    }
    if actual_paths != expected_paths:
        raise ValueError("Native source tree contains missing or unverified files.")
    for name, expected in manifest["source_files"].items():
        if (
            Path(name).name != name
            or core.sha256_file(source_root / "src/gbnsr6" / name) != expected
        ):
            raise ValueError(f"Native source manifest mismatch: {name}.")
    for name, expected in SOURCE_HASHES.items():
        if core.sha256_file(source_root / "src/gbnsr6" / name) != expected:
            raise ValueError(f"Native source SHA256 mismatch: {name}.")
    for name, expected in INCLUDE_HASHES.items():
        if core.sha256_file(source_root / "src/include" / name) != expected:
            raise ValueError(f"Native include SHA256 mismatch: {name}.")


def _once(source: str, old: str, replacement: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"Native source anchor is not unique: {old!r}.")
    return source.replace(old, replacement, 1)


def _routine(source: str, name: str) -> str:
    lower = source.lower()
    start = lower.index("\nsubroutine " + name)
    end = lower.index("end subroutine " + name, start) + len("end subroutine " + name)
    return source[start:end] + "\n"


def _traced_equation(source: str) -> str:
    body = _routine(source, "chagb_equation")
    body = _once(
        body,
        "      self_e = qid2h*temp1",
        "      self_e = qid2h*temp1\n"
        "      write(6,'(a,i8,4(1x,es25.17e3))') 'MAPLE_CHA_ATOM_V1 ', "
        "i, reff(i), qeff(i), mu_in_i, -self_e",
    )
    return _once(
        body,
        "         e = qiqj*temp1",
        "         e = qiqj*temp1\n"
        "         write(6,'(a,2i8,1x,es25.17e3)') 'MAPLE_CHA_PAIR_V1 ', i, j, e",
    )


def _instrumented_full_source(source: str) -> str:
    source = _once(source, _routine(source, "chagb_equation"), _traced_equation(source))
    source = _once(
        source,
        "   ! Add the offset paramber B",
        "   write(6,'(a,i8,6(1x,es25.17e3))') 'MAPLE_CHA_INPUT_HEADER_V1 ', "
        "natom, dprob, gb_ROH, gb_tau, epsin/eps0, epsout/eps0, gb_Rs\n"
        "   do i=1,natom\n"
        "      write(6,'(a,i8,6(1x,es25.17e3))') 'MAPLE_CHA_INPUT_V1 ', "
        "i, acrd(1:3,i), acg(i), radi(i), onereff(i)\n"
        "   enddo\n   ! Add the offset paramber B",
    )
    call = "      call chagb_equation(gb_arad, epol, eel, natex,nshrt  )"
    return _once(
        source,
        call,
        call + "\n      write(6,'(a,3(1x,es25.17e3))') "
        "'MAPLE_CHA_POLAR_V1 ',gb_arad,gb_B,epol",
    )


def _standalone_source(source: str, constants: str) -> str:
    # The whole equation/size/shift body is extracted at run time from verified
    # local GPL source.  Only the host state and IO below are project-authored.
    shift_start = source.index("   ! Add the offset paramber B")
    shift_end = source.index("   ! print if necessary", shift_start)
    header = """module solvent_accessibility
implicit none
real(kind=8) :: dprob
end module
module oracle
implicit none
real(kind=8), parameter :: alpb_alpha=.571412d0
real(kind=8), parameter :: eps0=8.8542D-12/(1.6022D-19)**2/(1.00D+10)**3*(1.00D+12)**2*1.6606D-27
real(kind=8) :: epsin,epsout,pbkappa,gb_tau,gb_ROH,gb_B,gb_arad,molecule_mass,x_cm,y_cm,z_cm
real(kind=8),allocatable :: acrd(:,:),acg(:),radi(:),reff(:),onereff(:),Xcm(:),Ycm(:),Zcm(:)
logical,allocatable :: skipv(:)
integer :: natom,gb_alpb,gb_dgij,gb_cha
"""
    main = """program run_oracle
use oracle
use solvent_accessibility
implicit none
integer :: i
integer,allocatable :: nshrt(:),natex(:)
real(kind=8) :: epol,eel
read(*,*) natom
allocate(acrd(3,natom),acg(natom),radi(natom),reff(natom),onereff(natom),skipv(natom),nshrt(0:natom),natex(1))
read(*,*) dprob
read(*,*) epsin,epsout,gb_tau,gb_ROH
epsin=epsin*eps0;epsout=epsout*eps0;pbkappa=0d0;gb_alpb=1;gb_dgij=0;gb_cha=1
nshrt=0;natex=0
do i=1,natom
read(*,*) acrd(:,i),acg(i),radi(i),onereff(i)
enddo
"""
    text = (
        header
        + constants
        + "\ncontains\n"
        + _traced_equation(source)
        + _routine(source, "mu_inv")
        + _routine(source, "gb_elsize")
        + "subroutine mexit(a,b)\ninteger :: a,b\nerror stop 'invalid algebra domain'\n"
        "end subroutine\nend module\n"
        + main
        + source[shift_start:shift_end]
        + "epol=0d0;eel=0d0\ncall chagb_equation(gb_arad,epol,eel,natex,nshrt)\n"
        "write(6,'(a,3(1x,es25.17e3))') 'MAPLE_CHA_POLAR_V1 ',gb_arad,gb_B,epol\n"
        "end program\n"
    )
    return text.replace("_REAL_", "real(kind=8)").replace(
        '#  include "pb_constants.h"', ""
    )


def _command(command: list[str], cwd: Path, records: list[dict], *, stdin=None) -> str:
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    completed = subprocess.run(
        command,
        cwd=cwd,
        input=stdin,
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
    )
    index = len(records)
    stdout = completed.stdout.encode()
    stderr = completed.stderr.encode()
    (cwd / f"command-{index}.stdout").write_bytes(stdout)
    (cwd / f"command-{index}.stderr").write_bytes(stderr)
    records.append(
        {
            "argv": command,
            "cwd": str(cwd),
            "returncode": completed.returncode,
            "stdin_sha256": (
                None if stdin is None else core.sha256_bytes(stdin.encode())
            ),
            "stdout_sha256": core.sha256_bytes(stdout),
            "stderr_sha256": core.sha256_bytes(stderr),
        }
    )
    if completed.returncode:
        raise RuntimeError(
            f"Native oracle command failed; see {cwd}/command-{index}.stderr"
        )
    return completed.stdout


def _trace(text: str) -> dict:
    atoms, pairs, headers, inputs, energy = [], [], [], [], []
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "MAPLE_CHA_ATOM_V1":
            atoms.append((int(fields[1]), *map(float, fields[2:])))
        elif fields[0] == "MAPLE_CHA_PAIR_V1":
            pairs.append((int(fields[1]), int(fields[2]), float(fields[3])))
        elif fields[0] == "MAPLE_CHA_INPUT_HEADER_V1":
            headers.append((int(fields[1]), *map(float, fields[2:])))
        elif fields[0] == "MAPLE_CHA_INPUT_V1":
            inputs.append((int(fields[1]), *map(float, fields[2:])))
        elif fields[0] == "MAPLE_CHA_POLAR_V1":
            energy.append(list(map(float, fields[1:])))
    if len(energy) != 1 or not atoms:
        raise ValueError("Expected one complete native CHA trace.")
    atoms.sort()
    count = len(atoms)
    if [row[0] for row in atoms] != list(range(1, count + 1)):
        raise ValueError("Native atom trace is incomplete.")
    if len(pairs) != count * (count - 1) // 2:
        raise ValueError("Native pair trace is incomplete.")
    polar = energy[0][2]
    result = {
        "polar_kcal_mol": polar,
        "self_kcal_mol": sum(row[4] for row in atoms),
        "pair_kcal_mol": sum(row[2] for row in pairs),
        "electrostatic_size_angstrom": energy[0][0],
        "inverse_born_shift_per_angstrom": energy[0][1],
        "born_radii_angstrom": [row[1] for row in atoms],
        "effective_charges_e": [row[2] / 18.2223 for row in atoms],
        "cha_factors": [row[3] for row in atoms],
    }
    if abs(polar - result["self_kcal_mol"] - result["pair_kcal_mol"]) > 1e-9:
        raise ValueError("Native self/pair accumulation does not close.")
    if inputs:
        inputs.sort()
        if (
            len(headers) != 1
            or len(headers[0]) != 7
            or len(inputs) != count
            or headers[0][0] != count
            or any(len(row) != 7 for row in inputs)
        ):
            raise ValueError("Full-native input trace is incomplete.")
        expected = (0.88, 0.586, 1.47, 1.0, 78.5, 0.52)
        if any(abs(a - b) > 1e-12 for a, b in zip(headers[0][1:], expected)):
            raise ValueError("Native effective probe/CHA profile drifted.")
        result["inputs"] = {
            "positions_angstrom": [list(row[1:4]) for row in inputs],
            "charges_e": [row[4] / 18.2223 for row in inputs],
            "effective_cha_radii_angstrom": [row[5] for row in inputs],
            "unshifted_inverse_born_per_angstrom": [row[6] for row in inputs],
        }
    return result


def _input_text(inputs: dict) -> str:
    count = len(inputs["charges_e"])
    if any(
        len(inputs[name]) != count
        for name in (
            "positions_angstrom",
            "effective_cha_radii_angstrom",
            "unshifted_inverse_born_per_angstrom",
        )
    ):
        raise ValueError("Standalone oracle inputs have inconsistent atom counts.")
    rows = zip(
        inputs["positions_angstrom"],
        inputs["charges_e"],
        inputs["effective_cha_radii_angstrom"],
        inputs["unshifted_inverse_born_per_angstrom"],
    )
    lines = [str(len(inputs["charges_e"])), format(0.88, ".17g"), "1 78.5 1.47 .586"]
    for xyz, charge, radius, inverse in rows:
        lines.append(
            " ".join(
                format(value, ".17g")
                for value in (
                    *xyz,
                    charge * 18.2223,
                    radius,
                    inverse,
                )
            )
        )
    return "\n".join(lines) + "\n"


def _synthetic_cases() -> list[dict]:
    cases = []
    for charge in (-1.0, 0.0, 1.0):
        cases.append(
            {
                "case_id": f"one-site-charge-{charge}",
                "inputs": {
                    "positions_angstrom": [[1.2, -0.3, 0.5]],
                    "charges_e": [charge],
                    "effective_cha_radii_angstrom": [1.7],
                    "unshifted_inverse_born_per_angstrom": [1 / 2.1],
                },
            }
        )
    for radius in (9.999, 10.0, 10.001):
        cases.append(
            {
                "case_id": f"one-site-size-{radius}",
                "inputs": {
                    "positions_angstrom": [[0.0, 0.0, 0.0]],
                    "charges_e": [1.0],
                    "effective_cha_radii_angstrom": [radius],
                    "unshifted_inverse_born_per_angstrom": [0.5],
                },
            }
        )
    for name, separation in (("compact", 1.3), ("extended", 40.0)):
        cases.append(
            {
                "case_id": name + "-synthetic-born",
                "inputs": {
                    "positions_angstrom": [
                        [0, 0, 0],
                        [separation, 0.3, 0],
                        [0.2, 1.1, 0.1],
                    ],
                    "charges_e": [0.35, -0.55, 0.2],
                    "effective_cha_radii_angstrom": [2.12, 1.88, 1.04],
                    "unshifted_inverse_born_per_angstrom": [0.4, 0.5, 0.48],
                },
            }
        )
    return cases


def _build_full(
    source_root: Path,
    scratch: Path,
    prepared: Path,
    compiler: Path,
    prefix: Path,
    commands: list[dict],
) -> tuple[dict, dict]:
    shutil.copytree(source_root, scratch)
    source = scratch / "src/gbnsr6"
    egb = source / "egb.F90"
    egb.write_text(_instrumented_full_source(egb.read_text()))
    (scratch / "src/config.h").write_text(
        f"FC={compiler}\nPBSAFLAG=\nFPPFLAGS=-cpp\n"
        "FFLAGS=-ffree-line-length-none -fallow-argument-mismatch "
        f"-fno-inline-arg-packing -I{source} -I{scratch}/src/include "
        f"-I{prefix}/include\nAMBERFFLAGS=\nFOPTFLAGS=-O2\nLIBDIR={prefix}/lib\nVB=\n"
    )
    _command(
        ["make", "-B", "-j1", *(name + ".o" for name in OBJECTS)], source, commands
    )
    object_hashes = {
        name + ".o": core.sha256_file(source / (name + ".o")) for name in OBJECTS
    }
    binary = scratch / "gbnsr6"
    _command(
        [
            str(compiler),
            "-o",
            str(binary),
            *(str(source / (name + ".o")) for name in OBJECTS),
            f"-L{prefix}/lib",
            f"-Wl,-rpath,{prefix}/lib",
            "-lamber_common",
            "-lblas",
            "-llapack",
            "-lm",
            "-lxblas-amb",
        ],
        scratch,
        commands,
    )
    linker_inputs = {
        name: core.sha256_file(prefix / "lib" / name)
        for name in ("libamber_common.a", "libxblas-amb.a")
    }
    dynamic_libraries = {}
    for line in _command(["ldd", str(binary)], scratch, commands).splitlines():
        if "not found" in line:
            raise ValueError(f"Native oracle has an unresolved dynamic library: {line}")
        fields = line.split()
        if "=>" in fields:
            linked_path = Path(fields[fields.index("=>") + 1])
        elif fields and fields[0].startswith("/"):
            linked_path = Path(fields[0])
        else:
            continue  # linux-vdso has no file to fingerprint.
        resolved = linked_path.resolve(strict=True)
        dynamic_libraries[str(resolved)] = core.sha256_file(resolved)
    run = scratch / "run"
    run.mkdir()
    for name in ("molecule.prmtop", "molecule.inpcrd", "gbnsr6.in"):
        shutil.copy2(prepared / name, run / name)
    _command(
        [
            str(binary),
            "-O",
            "-i",
            "gbnsr6.in",
            "-o",
            "gbnsr6.out",
            "-p",
            "molecule.prmtop",
            "-c",
            "molecule.inpcrd",
        ],
        run,
        commands,
    )
    return _trace((run / "gbnsr6.out").read_text()), {
        "instrumented_egb_sha256": core.sha256_file(egb),
        "binary_sha256": core.sha256_file(binary),
        "native_output_sha256": core.sha256_file(run / "gbnsr6.out"),
        "object_sha256": object_hashes,
        "static_link_input_sha256": linker_inputs,
        "dynamic_library_sha256": dynamic_libraries,
    }


def generate(
    source_root: Path,
    manifest_path: Path,
    prepared: Path,
    compiler: Path,
    prefix: Path,
    work_dir: Path,
    output: Path,
) -> dict:
    _verify_sources(source_root, manifest_path)
    compiler = compiler.resolve(strict=True)
    if work_dir.exists() or output.exists():
        raise FileExistsError("Use new work/output paths; do not overwrite evidence.")
    work_dir.mkdir(parents=True)
    source = (source_root / "src/gbnsr6/egb.F90").read_text()
    constants = (source_root / "src/gbnsr6/pb_constants.h").read_text()
    payloads, provenance = [], []
    for index in range(2):
        scratch = work_dir / f"repeat-{index}"
        scratch.mkdir()
        commands = []
        full, native_info = _build_full(
            source_root, scratch / "native", prepared, compiler, prefix, commands
        )
        if abs(full["polar_kcal_mol"] - PINNED_FULL_REFERENCE_EGB_KCAL_MOL) > 1e-12:
            raise ValueError("Trace-only full GBNSR6 changed the pinned reference EGB.")
        bridge_inputs = full.pop("inputs")
        generated_source = scratch / "algebra.f90"
        generated_source.write_text(_standalone_source(source, constants))
        _command(
            [
                str(compiler),
                "-O2",
                "-ffree-line-length-none",
                "algebra.f90",
                "-o",
                "algebra",
            ],
            scratch,
            commands,
        )
        cases = [
            {"case_id": "full-native-methyl-hexanoate-bridge", "inputs": bridge_inputs},
            *_synthetic_cases(),
        ]
        for case in cases:
            output_text = _command(
                [str(scratch / "algebra")],
                scratch,
                commands,
                stdin=_input_text(case["inputs"]),
            )
            case["expected"] = _trace(output_text)
        delta = abs(full["polar_kcal_mol"] - cases[0]["expected"]["polar_kcal_mol"])
        if delta > BRIDGE_TOLERANCE_KCAL_MOL:
            raise ValueError("Standalone oracle failed full-native EGB bridge.")
        payload = {"cases": cases, "full_native_bridge_expected": full}
        payloads.append(payload)
        provenance.append(
            {
                **native_info,
                "commands": commands,
                "standalone_source_sha256": core.sha256_file(generated_source),
                "standalone_binary_sha256": core.sha256_file(scratch / "algebra"),
                "numeric_payload_sha256": core.sha256_bytes(
                    core.canonical_json_bytes(payload)
                ),
                "bridge_absolute_error_kcal_mol": delta,
            }
        )
    if core.canonical_json_bytes(payloads[0]) != core.canonical_json_bytes(payloads[1]):
        raise ValueError("Two clean-scratch numeric oracle replays differ.")
    artifact = {
        "schema_version": 1,
        "artifact_type": "route1-chagb-polar-algebra-native-oracle-v1",
        "scope": "Polar algebra only with supplied raw NSR6 inputs; no complete coordinate forces.",
        "label_reads": False,
        "new_qm": False,
        "full_chain_complete": False,
        "source_sha256": {
            **SOURCE_HASHES,
            **{f"include/{name}": digest for name, digest in INCLUDE_HASHES.items()},
        },
        "source_manifest_sha256": SOURCE_MANIFEST_HASH,
        "source_manifest_file_count": 64,
        "verified_source_file_count": 70,
        "pinned_full_reference_egb_kcal_mol": PINNED_FULL_REFERENCE_EGB_KCAL_MOL,
        "prepared_input_sha256": {
            name: core.sha256_file(prepared / name)
            for name in ("molecule.prmtop", "molecule.inpcrd", "gbnsr6.in")
        },
        "compiler": {
            "path": str(compiler),
            "sha256": core.sha256_file(compiler),
            "version": subprocess.check_output(
                [str(compiler), "--version"], text=True
            ).splitlines()[0],
        },
        "generator_sha256": core.sha256_file(SCRIPT),
        "execution_environment": {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        "bridge_tolerance_kcal_mol": BRIDGE_TOLERANCE_KCAL_MOL,
        "two_clean_scratch_numeric_replays_identical": True,
        "runs": provenance,
        **payloads[0],
    }
    core.write_json_atomic(output, core.seal_artifact(artifact))
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--prepared-input-dir", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, default=Path("/usr/bin/gfortran"))
    parser.add_argument("--amber-prefix", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = generate(
        args.source_root.resolve(),
        args.source_manifest.resolve(),
        args.prepared_input_dir.resolve(),
        args.compiler,
        args.amber_prefix.resolve(),
        args.work_dir.resolve(),
        args.output.resolve(),
    )
    print(
        json.dumps(
            {
                "case_count": len(artifact["cases"]),
                "bridge_error_kcal_mol": artifact["runs"][0][
                    "bridge_absolute_error_kcal_mol"
                ],
                "content_sha256": artifact["content_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
