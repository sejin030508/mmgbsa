import argparse
from biotite.structure.io import load_structure, save_structure

import mdtraj, os, tempfile
from pathlib import Path
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm, trange

parser = argparse.ArgumentParser()
parser.add_argument('--bioemu_dir', type=str) # raw trajectory 
parser.add_argument('--outdir', type=str, default='./data_atlas')
parser.add_argument('--num_workers', type=int, default=1)
args = parser.parse_args()


os.makedirs(args.outdir, exist_ok=True)
mmcif_outdir = os.path.join(args.outdir, 'mmcif')
os.makedirs(mmcif_outdir, exist_ok=True)

targets = Path(args.bioemu_dir).glob('*')
jobs = [f.name for f in targets if f.is_dir()]

print(f"Found {len(jobs)} targets to process.")


def pdb_to_cif(pdb_path, cif_path):
    pdb_structure = load_structure(pdb_path)
    save_structure(cif_path, pdb_structure)

def load_trajectory(name, traj_name):
    traj = mdtraj.load(f'{args.bioemu_dir}/{name}/trajs/{traj_name}.cmprsd.xtc', top=f'{args.bioemu_dir}/{name}/topology.pdb') # .cmprsd.xtc
    # never load pdb as part of structure 
    # may contain invalid atom name
    # ref = mdtraj.load(f'{args.bioemu_dir}/{name}/topology.pdb')
    # traj = ref + traj
    return traj     # mdtraj object

def convert_traj_to_cif(
    pc_name: str,   # xxxx_Y
):
    # load trajectory
    trajs_dir = f'{args.bioemu_dir}/{pc_name}/trajs'
    if not os.path.exists(trajs_dir):
        return
    traj_name_lst = list(os.listdir(trajs_dir))
    traj_name_lst = [x[:-11] for x in traj_name_lst] # x[:-11]
    
    for traj_name in traj_name_lst:
        # if not traj_name.startswith("run"):
        #     continue
            
        try:
            traj = load_trajectory(pc_name, traj_name)
        except:
            continue
            
        save_idx = 0
        for i in trange(0, len(traj)):    # 10 ns interval
            with tempfile.NamedTemporaryFile(suffix='.pdb') as temp:
                traj[i].save_pdb(temp.name)
                # convert to cif 
                save_dir = os.path.join(mmcif_outdir, pc_name)
                if not os.path.exists(save_dir):
                    os.system(f"mkdir -p {save_dir}")
                cif_path = os.path.join(save_dir, f"{traj_name}_{save_idx}.cif")
                if os.path.exists(cif_path):
                    os.system(f"rm {cif_path}") # remove old version 
                save_idx += 1
                pdb_to_cif(temp.name, cif_path) # cif is saved
    
    return True


def main():
    if args.num_workers > 1:
        _ = [
            r for r in tqdm(
                Parallel(n_jobs=args.num_workers, return_as="generator_unordered")(
                    delayed(convert_traj_to_cif)(name)
            for name in jobs
            ),
            total=len(jobs),
        )
        ]
    else:
        for name in tqdm(jobs):
            convert_traj_to_cif(name)    
    return


main()