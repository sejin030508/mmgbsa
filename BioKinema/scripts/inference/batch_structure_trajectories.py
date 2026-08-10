#!/usr/bin/env python3
"""
Batch BioKinema inference from structure files (CIF or PDB) per sample, grouped by protein.

**Modes**

* ``protenix`` — same layout as before: ``<category>/<id>/<id>/seed_*/predictions/*_sample_*.cif``
* ``bioemu`` — ``<category>/<id>/`` contains ``samples_sidechain_rec.pdb`` + ``samples_sidechain_rec.xtc``;
  frames are exported to ``<id>/<frames_subdir>/bioemu_sample_*.pdb`` then each PDB is inferred.

Subprocess environment: edit ``INFERENCE_ENV_DEFAULTS`` below.

Examples::

  # Protenix CIF (equivalent to legacy script defaults)
  python scripts/batch_structure_trajectories.py protenix \\
    --protenix-results-root ../../Protenix_v1.0.0/protenix_results \\
    --output-root ./out --checkpoint-path /path/to.ckpt

  # BioEmu PDB + XTC
  python scripts/batch_structure_trajectories.py bioemu \\
    --bioemu-results-root /path/to/bioemu_results \\
    --output-root ./out --checkpoint-path /path/to.ckpt
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Fill these once for your machine (same as exporting in inference.sh).
INFERENCE_ENV_DEFAULTS: dict[str, str] = {
    "CUDA_HOME": "/cto_studio/xtalpi_lab/softwares/cuda-11.8",
    "CUTLASS_PATH": "/cto_studio/xtalpi_lab/fengbin/cutlass",
    "LAYERNORM_TYPE": "fast_layernorm",
    "USE_DEEPSPEED_EVO_ATTTENTION": "true",
    "PYTHONWARNINGS": "ignore::FutureWarning",
    "BIOKINEMA_QUIET_CCD_MSG": "1",
    "TRITON_CACHE_DIR": "",
    "BIOKINEMA_MSA_CACHE_DIR": "/cto_studio/xtalpi_lab/fengbin/Protenix_v0.2.0/BioKinema/msa",
}


def build_inference_subprocess_env(bio_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    for key, val in INFERENCE_ENV_DEFAULTS.items():
        v = (val or "").strip()
        if not v:
            continue
        if not (env.get(key) or "").strip():
            env[key] = v
    if not (env.get("TRITON_CACHE_DIR") or "").strip():
        env["TRITON_CACHE_DIR"] = os.path.join(
            tempfile.gettempdir(), "triton_cache_biokinema"
        )
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{bio_root}{os.pathsep}{prev}".strip(os.pathsep)
    return env


def _frame_sort_key(path: Path) -> tuple[int, str]:
    m = re.search(r"bioemu_sample_(\d+)", path.name, re.IGNORECASE)
    if m:
        return (int(m.group(1)), path.name)
    m = re.search(r"sample_(\d+)", path.name, re.IGNORECASE)
    if m:
        return (int(m.group(1)), path.name)
    return (-1, path.name)


def find_protenix_prediction_cifs(uniprot_inner: Path) -> list[Path]:
    cifs: list[Path] = []
    for seed_dir in sorted(uniprot_inner.glob("seed_*")):
        pred = seed_dir / "predictions"
        if not pred.is_dir():
            continue
        for p in pred.glob("*.cif"):
            if "summary" in p.name.lower():
                continue
            cifs.append(p)
    return sorted(cifs, key=_frame_sort_key)


def resolve_input_json(protein_dir: Path) -> str | None:
    for name in ("input.json", "input-add-msa.json"):
        cand = protein_dir / name
        if cand.is_file():
            return str(cand.resolve())
    inner = protein_dir / protein_dir.name
    if inner.is_dir():
        for name in ("input.json", "input-add-msa.json"):
            cand = inner / name
            if cand.is_file():
                return str(cand.resolve())
    return None


def iter_protenix_jobs(
    root: Path, categories: set[str] | None = None
):
    """Yields (category, uid, inner_dir, list[cif_paths])."""
    for cat_dir in sorted(root.iterdir()):
        if not cat_dir.is_dir():
            continue
        if categories is not None and cat_dir.name not in categories:
            continue
        for uid_dir in sorted(cat_dir.iterdir()):
            if not uid_dir.is_dir():
                continue
            uid = uid_dir.name
            inner = uid_dir / uid
            if not inner.is_dir():
                continue
            cifs = find_protenix_prediction_cifs(inner)
            if not cifs:
                continue
            yield cat_dir.name, uid, inner, cifs


# Above this size, prefer MDAnalysis (streaming) over mdtraj (full load into RAM).
_BIOEMU_LARGE_XTC_BYTES = 512 * 1024 * 1024


def _bioemu_effective_export_backend(cli: str) -> str:
    c = (cli or "auto").lower()
    if c != "auto":
        return c
    env = (os.environ.get("BIOKINEMA_BIOEMU_EXPORT_BACKEND") or "").strip().lower()
    if env in ("mdtraj", "mdanalysis"):
        return env
    return "auto"


def _bioemu_export_log_progress(
    i: int,
    n: int,
    *,
    interval: int,
) -> None:
    if n <= 0:
        return
    if interval <= 0:
        return
    if i == 0 or i == n - 1 or (i + 1) % interval == 0:
        print(
            f"[bioemu]   wrote frame {i + 1}/{n}",
            file=sys.stderr,
            flush=True,
        )


def export_bioemu_xtc_to_pdbs(
    topology: Path,
    trajectory: Path,
    out_dir: Path,
    *,
    prefix: str = "bioemu_sample",
    backend: str = "auto",
    progress_interval: int = 10,
) -> list[Path]:
    """Write one PDB per trajectory frame. Uses mdtraj and/or MDAnalysis."""
    out_dir.mkdir(parents=True, exist_ok=True)
    eff = _bioemu_effective_export_backend(backend)
    xtc_size = trajectory.stat().st_size
    if eff == "auto" and xtc_size > _BIOEMU_LARGE_XTC_BYTES:
        eff = "mdanalysis"

    print(
        f"[bioemu] reading {trajectory.name} ({xtc_size / 1024 / 1024:.2f} MiB), "
        f"backend={eff}",
        file=sys.stderr,
        flush=True,
    )

    def run_mdtraj() -> list[Path]:
        import mdtraj as md  # type: ignore

        traj = md.load(str(trajectory), top=str(topology))
        n = traj.n_frames
        print(
            f"[bioemu] loaded {n} frame(s), writing PDBs under {out_dir}",
            file=sys.stderr,
            flush=True,
        )
        paths: list[Path] = []
        for i in range(n):
            p = out_dir / f"{prefix}_{i}.pdb"
            traj[i].save_pdb(str(p))
            paths.append(p)
            _bioemu_export_log_progress(i, n, interval=progress_interval)
        return paths

    def run_mdanalysis() -> list[Path]:
        import MDAnalysis as mda  # type: ignore

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            u = mda.Universe(
                str(topology), str(trajectory), in_memory=False
            )
        n = len(u.trajectory)
        print(
            f"[bioemu] loaded {n} frame(s), writing PDBs under {out_dir}",
            file=sys.stderr,
            flush=True,
        )
        paths: list[Path] = []
        for i, _ts in enumerate(u.trajectory):
            p = out_dir / f"{prefix}_{i}.pdb"
            u.atoms.write(str(p))
            paths.append(p)
            _bioemu_export_log_progress(i, n, interval=progress_interval)
        return paths

    if eff == "mdtraj":
        try:
            return run_mdtraj()
        except ImportError as e:
            raise RuntimeError(
                "bioemu export backend is mdtraj but mdtraj is not installed."
            ) from e

    if eff == "mdanalysis":
        try:
            return run_mdanalysis()
        except ImportError as e:
            raise RuntimeError(
                "bioemu export backend is mdanalysis but MDAnalysis is not installed."
            ) from e

    # auto: try mdtraj first (usually faster for typical small BioEmu XTCs), then MDAnalysis
    try:
        import mdtraj  # noqa: F401  # type: ignore

        try:
            return run_mdtraj()
        except Exception as e:
            print(
                f"[bioemu] mdtraj export failed ({e!r}), trying MDAnalysis...",
                file=sys.stderr,
                flush=True,
            )
    except ImportError:
        pass

    try:
        return run_mdanalysis()
    except ImportError as e:
        raise RuntimeError(
            "Reading XTC requires mdtraj or MDAnalysis. "
            "Install: pip install mdtraj  (or pip install MDAnalysis)"
        ) from e


def find_bioemu_sample_pdbs(frames_dir: Path) -> list[Path]:
    pdbs = [p for p in frames_dir.glob("bioemu_sample_*.pdb") if p.is_file()]
    return sorted(pdbs, key=_frame_sort_key)


def iter_bioemu_jobs(
    root: Path,
    categories: set[str] | None,
    topology_name: str,
    trajectory_name: str,
):
    """Yields (category, uid, protein_dir, topology, trajectory)."""
    for cat_dir in sorted(root.iterdir()):
        if not cat_dir.is_dir():
            continue
        if categories is not None and cat_dir.name not in categories:
            continue
        for uid_dir in sorted(cat_dir.iterdir()):
            if not uid_dir.is_dir():
                continue
            uid = uid_dir.name
            top = uid_dir / topology_name
            xtc = uid_dir / trajectory_name
            if top.is_file() and xtc.is_file():
                yield cat_dir.name, uid, uid_dir, top, xtc


def collect_jobs_protenix(
    root: Path,
    out_root: Path,
    cat_filter: set[str] | None,
    *,
    verbose_missing_json: bool,
) -> list[tuple[str, str, str, Path, str]]:
    jobs: list[tuple[str, str, str, Path, str]] = []
    missing_json: list[Path] = []

    for category, uid, inner, paths in iter_protenix_jobs(root, cat_filter):
        dump_dir = str(out_root / category / uid)
        os.makedirs(dump_dir, exist_ok=True)
        input_json_path = resolve_input_json(inner)
        if input_json_path is None:
            input_json_path = str((inner / "input.json").resolve())
            missing_json.append(inner)
            if verbose_missing_json:
                print(
                    f"[warn] {inner}: no input.json; placeholder: {input_json_path}",
                    file=sys.stderr,
                )
        for p in paths:
            jobs.append((category, uid, dump_dir, p, input_json_path))

    if missing_json and not verbose_missing_json:
        n = len(missing_json)
        print(
            f"[warn] {n} protein dir(s) missing input.json; using placeholder.",
            file=sys.stderr,
        )
    return jobs


def collect_jobs_bioemu(
    root: Path,
    out_root: Path,
    cat_filter: set[str] | None,
    *,
    topology_name: str,
    trajectory_name: str,
    frames_subdir: str,
    force_export: bool,
    dry_run: bool,
    verbose_missing_json: bool,
    export_backend: str,
    export_progress_interval: int,
) -> list[tuple[str, str, str, Path, str]]:
    jobs: list[tuple[str, str, str, Path, str]] = []
    missing_json: list[Path] = []

    for category, uid, protein_dir, top, xtc in iter_bioemu_jobs(
        root, cat_filter, topology_name, trajectory_name
    ):
        dump_dir = str(out_root / category / uid)
        os.makedirs(dump_dir, exist_ok=True)
        frames_dir = protein_dir / frames_subdir

        input_json_path = resolve_input_json(protein_dir)
        if input_json_path is None:
            input_json_path = str((protein_dir / "input.json").resolve())
            missing_json.append(protein_dir)
            if verbose_missing_json:
                print(
                    f"[warn] {protein_dir}: no input.json; placeholder: {input_json_path}",
                    file=sys.stderr,
                )

        pdbs = find_bioemu_sample_pdbs(frames_dir)
        if force_export or not pdbs:
            if dry_run:
                print(
                    f"[dry-run] would export {top.name}+{xtc.name} -> {frames_dir}/",
                    file=sys.stderr,
                )
                continue
            print(
                f"[bioemu] export frames: {protein_dir} ({top.name} + {xtc.name})",
                file=sys.stderr,
            )
            export_bioemu_xtc_to_pdbs(
                top,
                xtc,
                frames_dir,
                backend=export_backend,
                progress_interval=export_progress_interval,
            )
            pdbs = find_bioemu_sample_pdbs(frames_dir)
        if not pdbs:
            print(f"[warn] no PDBs in {frames_dir} after export, skip {uid}", file=sys.stderr)
            continue
        for p in pdbs:
            jobs.append((category, uid, dump_dir, p, input_json_path))

    if missing_json and not verbose_missing_json:
        print(
            f"[warn] {len(missing_json)} protein dir(s) missing input.json (placeholder).",
            file=sys.stderr,
        )
    return jobs


def _cli_config_scalar(x: float | int) -> str:
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(x)


def build_inference_cmd(
    *,
    bio_root: Path,
    python_exe: str,
    checkpoint_path: str,
    dump_dir: str,
    input_file: str,
    input_json_path: str,
    coarse_frame_num: int,
    coarse_interval: float,
    fine_frame_num: int,
    w_h: int,
    w_g: int,
    seed: int,
    n_step: int,
    n_cycle: int,
    n_sample: int,
    lambda_: float,
    eta: float,
    test_set: str,
    sample_diffusion_chunk_size: int = 1,
) -> list[str]:
    return [
        python_exe,
        str(bio_root / "runner" / "inference.py"),
        "--seeds",
        str(seed),
        "--load_checkpoint_path",
        checkpoint_path,
        "--dump_dir",
        dump_dir,
        "--model.N_cycle",
        str(n_cycle),
        "--model.diffusion_module.causal_mask",
        "false",
        "--data.train_sets",
        test_set,
        "--data.test_sets",
        test_set,
        "--sample_diffusion.N_sample",
        str(n_sample),
        "--sample_diffusion.N_step",
        str(n_step),
        "--sample_diffusion.noise_scale_lambda",
        str(lambda_),
        "--sample_diffusion.step_scale_eta",
        str(eta),
        "--infer_setting.sample_diffusion_chunk_size",
        str(sample_diffusion_chunk_size),
        "--coarse_frame_num",
        str(coarse_frame_num),
        "--coarse_interval",
        _cli_config_scalar(coarse_interval),
        "--fine_frame_num",
        str(fine_frame_num),
        "--W_H",
        str(w_h),
        "--W_G",
        str(w_g),
        "--input_file",
        input_file,
        "--input_json_path",
        input_json_path,
        "--data.num_dl_workers",
        "1",
        "--data.msa.enable",
        "true",
        "--load_strict",
        "false",
    ]


def run_batch(args: argparse.Namespace, jobs: list[tuple[str, str, str, Path, str]]) -> None:
    bio_root = args.bio_root.resolve()
    out_root = args.output_root.resolve()
    base_env = build_inference_subprocess_env(bio_root)
    gpu_ids = parse_gpu_list(args.gpus)
    total = len(jobs)

    def env_for_job(cuda_device: str | None) -> dict[str, str]:
        if cuda_device is None:
            return base_env
        e = base_env.copy()
        e["CUDA_VISIBLE_DEVICES"] = cuda_device
        return e

    def one_cmd(
        category: str,
        uid: str,
        dump_dir: str,
        struct_path: Path,
        input_json_path: str,
    ) -> list[str]:
        return build_inference_cmd(
            bio_root=bio_root,
            python_exe=args.python,
            checkpoint_path=args.checkpoint_path,
            dump_dir=dump_dir,
            input_file=str(struct_path.resolve()),
            input_json_path=input_json_path,
            coarse_frame_num=args.coarse_frame_num,
            coarse_interval=args.coarse_interval,
            fine_frame_num=args.fine_frame_num,
            w_h=args.W_H,
            w_g=args.W_G,
            seed=args.seed,
            n_step=args.N_step,
            n_cycle=args.N_cycle,
            n_sample=args.N_sample,
            lambda_=args.lambda_,
            eta=args.eta,
            test_set=args.test_set,
            sample_diffusion_chunk_size=args.sample_diffusion_chunk_size,
        )

    def run_subprocess(
        job_index: int,
        category: str,
        uid: str,
        dump_dir: str,
        struct_path: Path,
        input_json_path: str,
        cuda_device: str | None,
    ) -> tuple[int, Path]:
        cmd = one_cmd(category, uid, dump_dir, struct_path, input_json_path)
        tag = f"CUDA_VISIBLE_DEVICES={cuda_device}" if cuda_device is not None else "default GPU"
        print(f"[{tag}] [{category}/{uid}] {struct_path.name} -> {dump_dir}")
        if args.dry_run:
            print(" ", " ".join(cmd))
            return 0, struct_path
        r = subprocess.run(cmd, cwd=str(bio_root), env=env_for_job(cuda_device))
        return r.returncode, struct_path

    if total == 0:
        print("No structure jobs found.", file=sys.stderr)
        sys.exit(1)

    if args.dry_run or gpu_ids is None:
        for i, (category, uid, dump_dir, struct_path, input_json_path) in enumerate(jobs):
            dev = gpu_ids[i % len(gpu_ids)] if gpu_ids else None
            rc, _ = run_subprocess(
                i, category, uid, dump_dir, struct_path, input_json_path, dev
            )
            if not args.dry_run and rc != 0:
                print(f"FAILED (exit {rc}): {struct_path}", file=sys.stderr)
                sys.exit(rc)
        print(f"Done. Processed {total} structure file(s).")
        return

    workers = len(gpu_ids)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = []
        for i, (category, uid, dump_dir, struct_path, input_json_path) in enumerate(jobs):
            dev = gpu_ids[i % len(gpu_ids)]
            futs.append(
                ex.submit(
                    run_subprocess,
                    i,
                    category,
                    uid,
                    dump_dir,
                    struct_path,
                    input_json_path,
                    dev,
                )
            )
        for fut in as_completed(futs):
            rc, struct_path = fut.result()
            if rc != 0:
                print(f"FAILED (exit {rc}): {struct_path}", file=sys.stderr)
                ex.shutdown(wait=False, cancel_futures=True)
                sys.exit(rc)

    print(f"Done. Processed {total} structure file(s).")


def parse_gpu_list(s: str | None) -> list[str] | None:
    if not (s or "").strip():
        return None
    ids = [x.strip() for x in s.split(",") if x.strip()]
    return ids if ids else None


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Batch BioKinema inference from Protenix CIFs or BioEmu PDB+XTC.",
    )
    sub = ap.add_subparsers(dest="mode", required=True)

    p_px = sub.add_parser(
        "protenix",
        help="Protenix layout: <cat>/<id>/<id>/seed_*/predictions/*.cif",
    )
    p_px.add_argument(
        "--protenix-results-root",
        type=Path,
        required=True,
        help="Root folder (crypticpocket, domainmotion, ...).",
    )

    p_be = sub.add_parser(
        "bioemu",
        help="BioEmu: <cat>/<id>/samples_sidechain_rec.pdb + .xtc",
    )
    p_be.add_argument(
        "--bioemu-results-root",
        type=Path,
        required=True,
    )
    p_be.add_argument(
        "--bioemu-topology",
        type=str,
        default="samples_sidechain_rec.pdb",
        help="Topology filename under each protein directory.",
    )
    p_be.add_argument(
        "--bioemu-trajectory",
        type=str,
        default="samples_sidechain_rec.xtc",
        help="Trajectory filename under each protein directory.",
    )
    p_be.add_argument(
        "--bioemu-frames-subdir",
        type=str,
        default="_bioemu_sample_pdbs",
        help="Subdir under each protein dir for exported bioemu_sample_*.pdb",
    )
    p_be.add_argument(
        "--bioemu-force-export",
        action="store_true",
        help="Re-export XTC to PDBs even if frames already exist.",
    )
    p_be.add_argument(
        "--bioemu-export-backend",
        choices=("auto", "mdtraj", "mdanalysis"),
        default="auto",
        help="Trajectory reader for PDB export. Default auto: mdtraj for small XTC, "
        "else MDAnalysis; override with env BIOKINEMA_BIOEMU_EXPORT_BACKEND.",
    )
    p_be.add_argument(
        "--bioemu-export-progress-interval",
        type=int,
        default=10,
        metavar="N",
        help="Log every N frames while exporting (default 10). Use 1 for every frame.",
    )

    for p in (p_px, p_be):
        p.add_argument("--output-root", type=Path, required=True)
        p.add_argument("--checkpoint-path", type=str, required=True)
        p.add_argument(
            "--bio-root",
            type=Path,
            default=Path(__file__).resolve().parent.parent,
        )
        p.add_argument("--python", type=str, default=sys.executable)
        p.add_argument("--coarse-frame-num", type=int, default=51)
        p.add_argument("--coarse-interval", type=float, default=20.0)
        p.add_argument("--fine-frame-num", type=int, default=1)
        p.add_argument("--W_H", type=int, default=1)
        p.add_argument("--W_G", type=int, default=50)
        p.add_argument("--seed", type=int, default=101)
        p.add_argument("--N_step", type=int, default=20)
        p.add_argument("--N_cycle", type=int, default=10)
        p.add_argument("--N_sample", type=int, default=1)
        p.add_argument("--lambda", dest="lambda_", type=float, default=1.75)
        p.add_argument("--sample-diffusion-chunk-size", type=int, default=1)
        p.add_argument("--eta", type=float, default=1.5)
        p.add_argument("--test-set", type=str, default="inference")
        p.add_argument("--dry-run", action="store_true")
        p.add_argument(
            "--categories",
            nargs="*",
            default=None,
            metavar="NAME",
        )
        p.add_argument("--gpus", type=str, default=None)
        p.add_argument("--verbose-missing-json", action="store_true")

    return ap


def main(argv: list[str] | None = None) -> None:
    ap = build_parser()
    args = ap.parse_args(argv)
    cat_filter = set(args.categories) if args.categories else None

    bio_root = args.bio_root.resolve()
    if not bio_root.is_dir():
        print(f"Bio root not found: {bio_root}", file=sys.stderr)
        sys.exit(1)

    if args.mode == "protenix":
        root = args.protenix_results_root.resolve()
        if not root.is_dir():
            print(f"Protenix results root not found: {root}", file=sys.stderr)
            sys.exit(1)
        jobs = collect_jobs_protenix(
            root,
            args.output_root.resolve(),
            cat_filter,
            verbose_missing_json=args.verbose_missing_json,
        )
    else:
        root = args.bioemu_results_root.resolve()
        if not root.is_dir():
            print(f"BioEmu results root not found: {root}", file=sys.stderr)
            sys.exit(1)
        jobs = collect_jobs_bioemu(
            root,
            args.output_root.resolve(),
            cat_filter,
            topology_name=args.bioemu_topology,
            trajectory_name=args.bioemu_trajectory,
            frames_subdir=args.bioemu_frames_subdir,
            force_export=args.bioemu_force_export,
            dry_run=args.dry_run,
            verbose_missing_json=args.verbose_missing_json,
            export_backend=args.bioemu_export_backend,
            export_progress_interval=args.bioemu_export_progress_interval,
        )

    run_batch(args, jobs)


if __name__ == "__main__":
    main()
