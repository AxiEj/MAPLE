"""Small secure-I/O primitives shared by Route-2 release tools.

The scientific modules stay free of filesystem policy.  Release tools use
this module to capture immutable input bytes once, bind a clean Git identity,
and publish one canonical artifact without replacing prior evidence.
"""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pwd
import stat
import subprocess
import tempfile
from typing import Callable, Iterable, Mapping, cast
from uuid import UUID

_MFD_CLOEXEC = 0x0001
_MFD_ALLOW_SEALING = 0x0002
_F_ADD_SEALS = 1033
_F_GET_SEALS = 1034
_F_SEAL_SEAL = 0x0001
_F_SEAL_SHRINK = 0x0002
_F_SEAL_GROW = 0x0004
_F_SEAL_WRITE = 0x0008
SECURE_PUBLICATION_CONTRACT_ID = (
    "maple-route2-linux-o-tmpfile-linkat-fsync-noreplace-v1"
)


class SecureArtifactError(ValueError):
    """Raised when an input or publication identity is ambiguous."""


def canonical_json_bytes(value: object) -> bytes:
    """Encode strict canonical JSON used for every release digest."""

    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def secure_publication_contract() -> dict[str, object]:
    return {
        "contract_id": SECURE_PUBLICATION_CONTRACT_ID,
        "platform": "linux",
        "anonymous_inode": "O_TMPFILE",
        "file_durability_before_commit": "fsync(tmpfd)-required",
        "commit": "linkat-no-replace",
        "directory_durability_after_commit": "fsync(parent-dirfd)-required",
        "path_boundary": "retained-O_NOFOLLOW-parent-dirfd-plus-lexical-identity",
        "precommit_stability_callback": "required-when-evidence-has-inputs",
        "directory_creation": "mkdirat-each-component-plus-fsync-parent",
    }


def canonical_custody_home() -> Path:
    uid = os.getuid()
    try:
        passwd_home = Path(pwd.getpwuid(uid).pw_dir)
    except KeyError as exc:
        raise SecureArtifactError("current UID has no passwd custody root") from exc
    if not passwd_home.is_absolute():
        raise SecureArtifactError("passwd custody root must be absolute")
    resolved = passwd_home.resolve(strict=True)
    home_environment = os.environ.get("HOME")
    if home_environment is None or Path(home_environment) != resolved:
        raise SecureArtifactError("HOME differs from the canonical passwd custody root")
    return resolved


def host_user_custody_contract() -> dict[str, object]:
    home = canonical_custody_home()
    return {
        "contract_id": "maple-route2-host-user-custody-root-v1",
        "scope": "host-user-global",
        "uid": os.getuid(),
        "passwd_home": os.fspath(home),
        "home_environment_must_match": True,
        "directory_owner": "current-uid",
        "group_or_world_writable": False,
    }


def linux_boot_id() -> str:
    try:
        value = (
            Path("/proc/sys/kernel/random/boot_id")
            .read_text(encoding="utf-8")
            .strip()
            .lower()
        )
        parsed = UUID(value)
    except (OSError, ValueError) as exc:
        raise SecureArtifactError("cannot capture Linux boot identity") from exc
    if str(parsed) != value:
        raise SecureArtifactError("Linux boot identity is not canonical")
    return value


def linux_process_start_ticks(process_id: int) -> int:
    if type(process_id) is not int or process_id < 1:
        raise SecureArtifactError("Linux process ID must be positive")
    try:
        value = Path(f"/proc/{process_id}/stat").read_text(encoding="utf-8")
        fields = value[value.rfind(")") + 2 :].split()
        start_ticks = int(fields[19])
    except (IndexError, OSError, ValueError) as exc:
        raise SecureArtifactError("cannot capture Linux process start time") from exc
    if start_ticks < 1:
        raise SecureArtifactError("Linux process start time is invalid")
    return start_ticks


def _json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SecureArtifactError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def load_json_bytes(data: bytes, *, role: str) -> dict[str, object]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_json_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                SecureArtifactError(f"non-finite JSON value in {role}: {token}")
            ),
        )
    except SecureArtifactError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SecureArtifactError(f"cannot parse {role}: {exc}") from exc
    if not isinstance(value, dict):
        raise SecureArtifactError(f"{role} must contain one JSON object")
    return value


@dataclass(frozen=True, slots=True)
class FileIdentity:
    device: int
    inode: int
    mode: int
    links: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> "FileIdentity":
        return cls(
            device=value.st_dev,
            inode=value.st_ino,
            mode=value.st_mode,
            links=value.st_nlink,
            size=value.st_size,
            mtime_ns=value.st_mtime_ns,
            ctime_ns=value.st_ctime_ns,
        )


def _reject_symlinks(path: Path, *, role: str) -> None:
    absolute = Path(os.path.abspath(path.expanduser()))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            item = current.lstat()
        except FileNotFoundError:
            break
        except OSError as exc:
            raise SecureArtifactError(f"cannot inspect {role}: {exc}") from exc
        if stat.S_ISLNK(item.st_mode):
            raise SecureArtifactError(f"{role} path must not contain symlinks")


@dataclass(frozen=True, slots=True)
class CapturedFile:
    path: Path
    identity: FileIdentity
    data: bytes
    sha256: str

    def assert_stable(self, *, role: str) -> None:
        _reject_symlinks(self.path, role=role)
        try:
            current = FileIdentity.from_stat(self.path.stat())
        except OSError as exc:
            raise SecureArtifactError(f"cannot recheck {role}: {exc}") from exc
        if current != self.identity:
            raise SecureArtifactError(f"{role} changed after capture")


def capture_file(path: str | Path, *, role: str) -> CapturedFile:
    """Read one regular, non-linked file once through an open descriptor."""

    lexical = Path(os.path.abspath(Path(path).expanduser()))
    _reject_symlinks(lexical, role=role)
    descriptor: int | None = None
    try:
        descriptor = os.open(
            lexical,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        before = FileIdentity.from_stat(os.fstat(descriptor))
        if not stat.S_ISREG(before.mode) or before.links != 1:
            raise SecureArtifactError(f"{role} must be a non-hard-linked regular file")
        blocks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            blocks.append(block)
        after = FileIdentity.from_stat(os.fstat(descriptor))
    except SecureArtifactError:
        raise
    except OSError as exc:
        raise SecureArtifactError(f"cannot capture {role}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    data = b"".join(blocks)
    if before != after or len(data) != before.size:
        raise SecureArtifactError(f"{role} changed while being captured")
    resolved = lexical.resolve(strict=True)
    capture = CapturedFile(
        path=resolved,
        identity=before,
        data=data,
        sha256=hashlib.sha256(data).hexdigest(),
    )
    capture.assert_stable(role=role)
    return capture


@contextmanager
def sealed_bytes_path(data: bytes, *, role: str):
    """Expose exact captured bytes through one sealed process-owned memfd."""

    if type(data) is not bytes or not data:
        raise TypeError(f"{role} bytes must be nonempty exact bytes")
    memfd_create = getattr(os, "memfd_create", None)
    if callable(memfd_create):
        create = cast(Callable[[str, int], int], memfd_create)
        descriptor = create(
            "maple-route2-captured-input",
            _MFD_CLOEXEC | _MFD_ALLOW_SEALING,
        )
    else:
        libc = ctypes.CDLL(None, use_errno=True)
        libc_create = getattr(libc, "memfd_create", None)
        if libc_create is None:
            raise SecureArtifactError(f"{role} requires Linux memfd_create")
        libc_create.argtypes = (ctypes.c_char_p, ctypes.c_uint)
        libc_create.restype = ctypes.c_int
        descriptor = int(
            libc_create(
                b"maple-route2-captured-input",
                _MFD_CLOEXEC | _MFD_ALLOW_SEALING,
            )
        )
        if descriptor < 0:
            error_number = ctypes.get_errno()
            raise SecureArtifactError(
                f"{role} memfd creation failed: {os.strerror(error_number)}"
            )
    try:
        _write_all(descriptor, data)
        os.fsync(descriptor)
        if _read_all(descriptor) != data:
            raise SecureArtifactError(f"{role} memfd bytes changed")
        required = _F_SEAL_SEAL | _F_SEAL_SHRINK | _F_SEAL_GROW | _F_SEAL_WRITE
        try:
            fcntl.fcntl(descriptor, _F_ADD_SEALS, required)
            actual = fcntl.fcntl(descriptor, _F_GET_SEALS)
        except OSError as exc:
            raise SecureArtifactError(f"{role} memfd sealing failed") from exc
        if actual & required != required:
            raise SecureArtifactError(f"{role} memfd retained incomplete seals")
        path = Path(f"/proc/self/fd/{descriptor}")
        descriptor_stat = os.fstat(descriptor)
        proc_stat = path.stat()
        if (descriptor_stat.st_dev, descriptor_stat.st_ino) != (
            proc_stat.st_dev,
            proc_stat.st_ino,
        ):
            raise SecureArtifactError(f"{role} procfd identity changed")
        yield path
    finally:
        os.close(descriptor)


@contextmanager
def captured_named_file(data: bytes, *, suffix: str, role: str):
    """Materialize captured bytes in a private directory when suffix matters.

    Some upstream readers dispatch on a filename suffix and cannot consume a
    sealed procfd.  The retained descriptor and private 0700 directory bind the
    reopened path; bytes and inode identity are rechecked after the reader
    returns.
    """

    if type(data) is not bytes or not data:
        raise TypeError(f"{role} bytes must be nonempty exact bytes")
    if not isinstance(suffix, str) or not suffix.startswith(".") or "/" in suffix:
        raise SecureArtifactError("captured file suffix is invalid")
    with tempfile.TemporaryDirectory(prefix="maple-route2-captured-") as directory:
        root = Path(directory)
        root.chmod(0o700)
        path = root / f"input{suffix}"
        descriptor = os.open(
            path,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o400,
        )
        try:
            _write_all(descriptor, data)
            os.fsync(descriptor)
            identity = FileIdentity.from_stat(os.fstat(descriptor))
            if _read_all(descriptor) != data:
                raise SecureArtifactError(f"{role} materialized bytes changed")
            if FileIdentity.from_stat(path.stat()) != identity:
                raise SecureArtifactError(f"{role} materialized inode changed")
            yield path
            if (
                FileIdentity.from_stat(os.fstat(descriptor)) != identity
                or FileIdentity.from_stat(path.stat()) != identity
                or _read_all(descriptor) != data
            ):
                raise SecureArtifactError(f"{role} changed while reopened")
        finally:
            os.close(descriptor)


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise SecureArtifactError(f"Git {arguments!r} failed: {detail}")
    return result.stdout.strip()


@dataclass(frozen=True, slots=True)
class RepositoryIdentity:
    root: Path
    head: str
    tree: str

    def as_dict(self) -> dict[str, object]:
        return {"head": self.head, "tree": self.tree, "clean": True}

    def assert_stable(self) -> None:
        if _git(self.root, "rev-parse", "HEAD") != self.head:
            raise SecureArtifactError("repository HEAD changed after capture")
        if _git(self.root, "rev-parse", "HEAD^{tree}") != self.tree:
            raise SecureArtifactError("repository tree changed after capture")
        if _git(self.root, "status", "--porcelain=v1", "--untracked-files=all"):
            raise SecureArtifactError("repository changed after capture")


def capture_clean_repository(path: str | Path) -> RepositoryIdentity:
    root = Path(path).expanduser().resolve(strict=True)
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise SecureArtifactError("execution requires a clean repository")
    identity = RepositoryIdentity(
        root=root,
        head=_git(root, "rev-parse", "HEAD"),
        tree=_git(root, "rev-parse", "HEAD^{tree}"),
    )
    identity.assert_stable()
    return identity


@dataclass(frozen=True, slots=True)
class StabilityGuard:
    repository: RepositoryIdentity
    captures: tuple[tuple[str, CapturedFile], ...]

    def assert_stable(self) -> None:
        self.repository.assert_stable()
        for role, capture in self.captures:
            capture.assert_stable(role=role)


def capture_repo_files(
    repository: RepositoryIdentity,
    relative_paths: Iterable[str],
) -> dict[str, CapturedFile]:
    captures: dict[str, CapturedFile] = {}
    for relative in relative_paths:
        if not isinstance(relative, str) or not relative:
            raise SecureArtifactError("source path must be a nonempty string")
        candidate = (repository.root / relative).resolve(strict=True)
        try:
            normalized = candidate.relative_to(repository.root).as_posix()
        except ValueError as exc:
            raise SecureArtifactError(
                f"source path escapes repository: {relative}"
            ) from exc
        if normalized != relative:
            raise SecureArtifactError(
                f"source path is not canonical: {relative!r} != {normalized!r}"
            )
        captures[relative] = capture_file(candidate, role=f"source {relative}")
    return captures


def source_sha256s(captures: Mapping[str, CapturedFile]) -> dict[str, str]:
    return {name: captures[name].sha256 for name in sorted(captures)}


def _write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise SecureArtifactError("artifact write made no progress")
        view = view[written:]


def _read_all(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    blocks: list[bytes] = []
    while block := os.read(descriptor, 1024 * 1024):
        blocks.append(block)
    return b"".join(blocks)


def _open_durable_directory_tree(*, root: Path, directory: Path) -> int:
    """Open/create each directory component and durably link every new child."""

    anchor = Path(os.path.abspath(root.expanduser()))
    target = Path(os.path.abspath(directory.expanduser()))
    _reject_symlinks(anchor, role="trusted publication root")
    anchor = anchor.resolve(strict=True)
    try:
        relative = target.relative_to(anchor)
    except ValueError as exc:
        raise SecureArtifactError("artifact output escapes its trusted root") from exc
    descriptor = os.open(
        anchor,
        os.O_RDONLY
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        for component in relative.parts:
            if component in {"", ".", ".."}:
                raise SecureArtifactError("artifact directory component is invalid")
            try:
                child = os.open(
                    component,
                    os.O_RDONLY
                    | os.O_DIRECTORY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                try:
                    os.mkdir(component, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
                else:
                    os.fsync(descriptor)
                try:
                    child = os.open(
                        component,
                        os.O_RDONLY
                        | os.O_DIRECTORY
                        | getattr(os, "O_CLOEXEC", 0)
                        | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=descriptor,
                    )
                except OSError as exc:
                    raise SecureArtifactError(
                        "artifact path must not contain symlinks"
                    ) from exc
                os.fsync(child)
            except OSError as exc:
                raise SecureArtifactError(
                    "artifact path must not contain symlinks"
                ) from exc
            child_stat = os.fstat(child)
            if not stat.S_ISDIR(child_stat.st_mode):
                os.close(child)
                raise SecureArtifactError("artifact path component is not a directory")
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def ensure_host_user_custody_directory(directory: str | Path) -> Path:
    home = canonical_custody_home()
    target = Path(os.path.abspath(Path(directory).expanduser()))
    descriptor = _open_durable_directory_tree(root=home, directory=target)
    try:
        current = home
        for component in (Path("."), *target.relative_to(home).parts):
            if component != Path("."):
                current /= component
            identity = current.lstat()
            if not stat.S_ISDIR(identity.st_mode):
                raise SecureArtifactError("custody path contains a non-directory")
            if identity.st_uid != os.getuid():
                raise SecureArtifactError(
                    "custody directory is not owned by current UID"
                )
            if stat.S_IMODE(identity.st_mode) & 0o022:
                raise SecureArtifactError("custody directory is group/world writable")
    finally:
        os.close(descriptor)
    return target


def _linkat(
    source_descriptor: int,
    destination_directory: int,
    destination_name: str,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "linkat", None)
    if function is None:
        raise SecureArtifactError("atomic publication requires linkat")
    function.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    )
    function.restype = ctypes.c_int
    result = function(
        source_descriptor,
        b"",
        destination_directory,
        os.fsencode(destination_name),
        0x1000,  # AT_EMPTY_PATH
    )
    if result == 0:
        return
    direct_error = ctypes.get_errno()
    if direct_error == errno.EEXIST:
        raise FileExistsError(destination_name)
    if direct_error not in (errno.ENOENT, errno.EPERM, errno.EOPNOTSUPP):
        raise SecureArtifactError(
            "anonymous-inode publication failed: " + os.strerror(direct_error)
        )
    proc_path = f"/proc/self/fd/{source_descriptor}"
    if FileIdentity.from_stat(os.stat(proc_path)) != FileIdentity.from_stat(
        os.fstat(source_descriptor)
    ):
        raise SecureArtifactError("publication procfd identity changed")
    result = function(
        -100,  # AT_FDCWD
        os.fsencode(proc_path),
        destination_directory,
        os.fsencode(destination_name),
        0x400,  # AT_SYMLINK_FOLLOW
    )
    if result == 0:
        return
    fallback_error = ctypes.get_errno()
    if fallback_error == errno.EEXIST:
        raise FileExistsError(destination_name)
    raise SecureArtifactError(
        "procfd publication failed: " + os.strerror(fallback_error)
    )


def publish_bytes_noreplace(
    path: str | Path,
    data: bytes,
    *,
    root: str | Path | None = None,
    stability: Callable[[], None] | None = None,
) -> Path:
    """Publish one verified anonymous inode under a retained parent dirfd."""

    if type(data) is not bytes:
        raise TypeError("published artifact must be exact bytes")
    destination = Path(os.path.abspath(Path(path).expanduser()))
    if root is None:
        raise SecureArtifactError("artifact publication requires a trusted root")
    directory = _open_durable_directory_tree(
        root=Path(root), directory=destination.parent
    )
    _reject_symlinks(destination, role="artifact output")
    parent_identity = FileIdentity.from_stat(os.fstat(directory))
    lexical_parent_identity = FileIdentity.from_stat(destination.parent.stat())
    if parent_identity != lexical_parent_identity:
        os.close(directory)
        raise SecureArtifactError("artifact output parent identity changed")
    descriptor: int | None = None
    try:
        temporary_flag = getattr(os, "O_TMPFILE", 0)
        if not isinstance(temporary_flag, int) or not temporary_flag:
            raise SecureArtifactError("secure publication requires Linux O_TMPFILE")
        descriptor = os.open(
            ".",
            temporary_flag | os.O_RDWR | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=directory,
        )
        before = FileIdentity.from_stat(os.fstat(descriptor))
        if not stat.S_ISREG(before.mode) or before.links != 0:
            raise SecureArtifactError("O_TMPFILE did not create an anonymous inode")
        _write_all(descriptor, data)
        os.fsync(descriptor)
        after_write = FileIdentity.from_stat(os.fstat(descriptor))
        if (
            _read_all(descriptor) != data
            or FileIdentity.from_stat(os.fstat(descriptor)) != after_write
        ):
            raise SecureArtifactError("anonymous artifact bytes changed")
        if stability is not None:
            stability()
        if FileIdentity.from_stat(os.fstat(directory)) != parent_identity:
            raise SecureArtifactError("artifact output parent changed before commit")
        if FileIdentity.from_stat(destination.parent.stat()) != parent_identity:
            raise SecureArtifactError("artifact lexical parent changed before commit")
        if FileIdentity.from_stat(os.fstat(descriptor)) != after_write:
            raise SecureArtifactError("anonymous artifact changed before commit")
        try:
            os.stat(destination.name, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(destination)
        _linkat(descriptor, directory, destination.name)
        linked = FileIdentity.from_stat(
            os.stat(destination.name, dir_fd=directory, follow_symlinks=False)
        )
        current = FileIdentity.from_stat(os.fstat(descriptor))
        if linked != current:
            raise SecureArtifactError("published inode differs from anonymous source")
        os.fsync(directory)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory)
    return destination


def publish_json_noreplace(
    path: str | Path,
    value: object,
    *,
    root: str | Path | None = None,
    stability: Callable[[], None] | None = None,
) -> Path:
    return publish_bytes_noreplace(
        path,
        canonical_json_bytes(value),
        root=root,
        stability=stability,
    )


def claim_execution(
    path: str | Path,
    payload: Mapping[str, object],
    *,
    root: str | Path | None = None,
    stability: Callable[[], None] | None = None,
) -> Path:
    """Durably claim one content-addressed execution before target evaluation."""

    execution_id = payload.get("execution_id")
    if not isinstance(execution_id, str) or len(execution_id) != 64:
        raise SecureArtifactError("execution claim needs a SHA256 execution_id")
    return publish_json_noreplace(
        path,
        dict(payload),
        root=root,
        stability=stability,
    )


__all__ = [
    "CapturedFile",
    "FileIdentity",
    "RepositoryIdentity",
    "SecureArtifactError",
    "SECURE_PUBLICATION_CONTRACT_ID",
    "StabilityGuard",
    "canonical_json_bytes",
    "canonical_sha256",
    "canonical_custody_home",
    "capture_clean_repository",
    "capture_file",
    "capture_repo_files",
    "captured_named_file",
    "claim_execution",
    "ensure_host_user_custody_directory",
    "host_user_custody_contract",
    "load_json_bytes",
    "linux_boot_id",
    "linux_process_start_ticks",
    "publish_bytes_noreplace",
    "publish_json_noreplace",
    "sealed_bytes_path",
    "secure_publication_contract",
    "source_sha256s",
]
