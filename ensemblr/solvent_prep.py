"""Build water-only GROMACS systems for each structure in an MC path."""

import os
import shutil
import subprocess

from tqdm import tqdm


# Minimal MDP written to each window directory before the genion grompp call.
_IONS_MDP = """\
integrator               = steep
nsteps                   = 0
emtol                    = 1000.0
emstep                   = 0.01
nstlog                   = 0
nstxout-compressed       = 0
"""

# Default steepest-descent EM MDP — emtol and nsteps are filled in at runtime.
_EM_MDP_TEMPLATE = """\
integrator               = steep
emtol                    = {emtol}
emstep                   = 0.01
nsteps                   = {nsteps}
nstlog                   = 500
nstxout                  = 0
nstxout-compressed       = 0
"""


class SolventSystemPrep:
    """
    Builds water-only GROMACS simulation systems for each structure in an
    MC path, with identical composition across all windows.

    For each path structure the pipeline runs:
        pdb2gmx → editconf → solvate → grompp → genion

    Window 0 sets the box dimensions (from protein bounding box + padding).
    All subsequent windows use the same fixed box size so that water counts
    differ only by tiny amounts due to varying protein volume, and any excess
    is trimmed in a final harmonisation step.

    Typical usage::

        prep = SolventSystemPrep(finder, ensemble, output_dir='solvent_prep/')
        prep.build_systems(pdb2gmx_flags=['-ignh', '-his'])
    """

    def __init__(self, path_finder, ensemble, output_dir,
                 force_field='charmm36-jul2022', water_model='tip3p',
                 gmx='gmx', env=None):
        """
        Parameters
        ----------
        path_finder : MCPathFinder
            Completed MCPathFinder (run() already called).
        ensemble : Ensemble
            Parent Ensemble (needs structure_directory set).
        output_dir : str
            Root directory; one window_N/ subdirectory is created per window.
        force_field : str
            Force field name passed to gmx pdb2gmx -ff.
        water_model : str
            Water model name passed to gmx pdb2gmx -water.
        gmx : str
            gmx executable name or full path if not on PATH.
        env : dict, optional
            Environment variables passed to every gmx subprocess.  Defaults to
            None (inherit the current process environment).  Use this to set
            LD_LIBRARY_PATH when the gmx binary requires shared libraries that
            are not already visible, e.g.::

                import os
                env = os.environ.copy()
                env['LD_LIBRARY_PATH'] = '/path/to/cuda/lib:' + env.get('LD_LIBRARY_PATH', '')
        """
        self.path_finder = path_finder
        self.ensemble = ensemble
        self.output_dir = output_dir.rstrip('/') + '/'
        self.force_field = force_field
        self.water_model = water_model
        self.gmx = gmx
        self.env = env
        self._window_dirs = []
        self._box_dims = None   # set from window 0; reused for all others
        self._box_type = None
        self._pname = None
        self._nname = None

    def build_systems(self, path_rank=0, box_padding=1.2, box_type='cubic',
                      pdb2gmx_flags=None, pdb2gmx_stdin=None,
                      neutralise=True, concentration=0.15,
                      pname='NA', nname='CL',
                      aln_selection='protein and name CA'):
        """
        Build a solvated GROMACS system for each path structure.

        Parameters
        ----------
        path_rank : int
            Which MC run to use (0 = best energy).
        box_padding : float
            Padding in nm between protein and box edge, applied to window 0.
            All other windows reuse the resulting box dimensions.
        box_type : str
            GROMACS box geometry: 'cubic', 'dodecahedron', or 'octahedron'.
        pdb2gmx_flags : list of str, optional
            Extra flags forwarded verbatim to gmx pdb2gmx.
            Defaults to ['-ignh'].  Add e.g. '-asp', '-glu', '-his' to
            control protonation states — pair these with pdb2gmx_stdin if
            those flags trigger interactive prompts.
        pdb2gmx_stdin : str, optional
            Text piped to pdb2gmx stdin to answer interactive prompts
            (e.g. histidine tautomer choices).
        neutralise : bool
            Pass -neutral to genion to add counter-ions.
        concentration : float
            NaCl concentration in mol/L for genion -conc.
            Set to 0 for neutralisation only (no extra salt).
        pname : str
            Positive ion residue name, e.g. 'NA' or 'K'.
        nname : str
            Negative ion residue name, e.g. 'CL'.
        aln_selection : str
            MDAnalysis selection used to align each structure onto window 0
            before solvation.  Set to None to skip alignment.

        Returns
        -------
        self
        """
        import MDAnalysis as mda
        import MDAnalysis.analysis.align

        if pdb2gmx_flags is None:
            pdb2gmx_flags = ['-ignh']

        self._box_type = box_type
        self._pname = pname
        self._nname = nname

        os.makedirs(self.output_dir, exist_ok=True)
        path_structures = self.path_finder.runs_df.iloc[path_rank]['path_structures']
        n_digits = len(str(len(path_structures) - 1))

        u_ref = None  # set from window 0, used to align subsequent structures
        self._window_dirs = []
        for i, structure in enumerate(tqdm(path_structures, desc='Building systems')):
            window_dir = os.path.join(self.output_dir, f'window_{i:0{n_digits}d}')
            os.makedirs(window_dir, exist_ok=True)
            self._window_dirs.append(window_dir)

            src = self.ensemble.structure_directory + structure
            input_pdb = os.path.join(window_dir, 'input.pdb')

            if aln_selection and i > 0:
                u = mda.Universe(src, src)
                mda.analysis.align.alignto(u, u_ref, select=aln_selection)
                u.atoms.write(input_pdb)
            else:
                shutil.copy(src, input_pdb)
                if aln_selection and i == 0:
                    u_ref = mda.Universe(src, src)

            self._pdb2gmx(window_dir, pdb2gmx_flags, pdb2gmx_stdin)

            if i == 0:
                self._editconf(window_dir, box_type=box_type, padding=box_padding)
                self._box_dims = self._read_box_dims(
                    os.path.join(window_dir, 'conf_box.gro'))
            else:
                self._editconf(window_dir, box_type=box_type,
                               box_dims=self._box_dims)

            self._solvate(window_dir)
            self._add_ions(window_dir, neutralise=neutralise,
                           concentration=concentration,
                           pname=pname, nname=nname)

        self._harmonise_composition()
        self._write_combined_pdb()
        return self

    def minimise_systems(self, minim_mdp=None, nsteps=50000, emtol=1000.0,
                         mdrun_flags=None):
        """
        Energy-minimise each window using steepest descent.

        Runs gmx grompp then gmx mdrun sequentially for every window.
        Final minimised coordinates are written to em.gro in each window
        directory alongside em.log and em.edr.

        Parameters
        ----------
        minim_mdp : str, optional
            Path to a custom .mdp file.  If None, a built-in steepest-descent
            MDP is used with the emtol and nsteps arguments below.
        nsteps : int
            Maximum EM steps (only used when minim_mdp is None).
        emtol : float
            Force convergence criterion in kJ mol⁻¹ nm⁻¹
            (only used when minim_mdp is None).
        mdrun_flags : list of str, optional
            Extra flags forwarded verbatim to gmx mdrun, e.g.
            ['-ntmpi', '1', '-ntomp', '4'].

        Returns
        -------
        self
        """
        if not self._window_dirs:
            raise RuntimeError('Call build_systems() before minimise_systems()')

        if mdrun_flags is None:
            mdrun_flags = []

        # If -ntomp is in mdrun_flags, OMP_NUM_THREADS must match or GROMACS
        # will refuse to run.  Build a per-mdrun env with the value synced.
        mdrun_env = dict(self.env) if self.env is not None else None
        if '-ntomp' in mdrun_flags and mdrun_env is not None:
            mdrun_env['OMP_NUM_THREADS'] = mdrun_flags[mdrun_flags.index('-ntomp') + 1]

        for window_dir in tqdm(self._window_dirs, desc='Minimising windows'):
            mdp_path = os.path.join(window_dir, 'minim.mdp')
            if minim_mdp is not None:
                shutil.copy(minim_mdp, mdp_path)
            else:
                with open(mdp_path, 'w') as f:
                    f.write(_EM_MDP_TEMPLATE.format(emtol=emtol, nsteps=nsteps))

            self._run(
                [self.gmx, 'grompp',
                 '-f', 'minim.mdp',
                 '-c', 'system.gro',
                 '-p', 'topol.top',
                 '-o', 'em.tpr',
                 '-maxwarn', '2'],
                cwd=window_dir, log='grompp_em.log',
            )
            self._run(
                [self.gmx, 'mdrun', '-v', '-deffnm', 'em'] + mdrun_flags,
                cwd=window_dir, log='mdrun_em.log',
                env=mdrun_env,
            )

        self._write_combined_pdb(gro_name='em.gro',
                                 filename='all_windows_minimised.pdb')
        return self

    def write_plumed_input(self, path_rank=0, cv_values=None,
                           force_constant=1000, distances_dict=None,
                           use_minimised=False):
        """
        Write per-window plumed.dat files and a combined plumed_multidir.dat.

        Mirrors UmbrellaSamplingPrep.write_plumed_input but for solvated
        systems.  molinfo.pdb and pca_ref.pdb are written to the output root
        directory and referenced with ``../`` paths so both per-window and
        multidir modes resolve correctly regardless of window name padding.

        Parameters
        ----------
        path_rank : int
            Which MC run to use (0 = best energy).
        cv_values : np.ndarray, optional
            Override the ideal window CV values.
        force_constant : float
            Harmonic restraint force constant (kJ mol⁻¹ nm⁻²).
        distances_dict : dict, optional
            {cv_name: (mda_selection_1, mda_selection_2)} — required when the
            CV is a named distance rather than a PC.
        use_minimised : bool
            If True, use em.gro as the reference structure instead of
            system.gro (i.e. call this after minimise_systems()).

        Returns
        -------
        self
        """
        import MDAnalysis as mda

        if not self._window_dirs:
            raise RuntimeError('Call build_systems() before write_plumed_input()')

        if cv_values is None:
            _, cv_values = self.path_finder.get_window_values(path_rank)

        gro_name = 'em.gro' if use_minimised else 'system.gro'
        ref_gro  = os.path.join(self._window_dirs[0], gro_name)
        u_template = mda.Universe(ref_gro)

        # molinfo PDB at the output root — referenced as ../molinfo.pdb from
        # each window dir, which works for both per-window and multidir modes.
        molinfo_pdb = os.path.join(self.output_dir, 'molinfo.pdb')
        u_template.atoms.write(molinfo_pdb)

        cv = self.path_finder.cv

        if distances_dict and cv in distances_dict:
            self._write_plumed_distance_cv(cv_values, distances_dict[cv],
                                           force_constant, u_template)
        elif cv.startswith('PC') and cv[2:].isdigit():
            self._write_plumed_pca_cv(cv_values, force_constant, u_template)
        else:
            print(f"Unknown CV '{cv}' — PLUMED input not written. "
                  "Pass distances_dict or use a PC-based CV.")

        return self

    # ── PLUMED helpers ────────────────────────────────────────────────────────

    def _write_plumed_pca_cv(self, cv_values, force_constant, u_template):
        import numpy as np
        from .structure_prep import write_pca_ref_pdb

        if self.ensemble.pca is None:
            raise RuntimeError('PCA not available — call ensemble.run_pca() first')

        pc_idx     = int(self.path_finder.cv[2:]) - 1
        eigenvector = self.ensemble.pca.p_components[:, pc_idx].reshape(-1, 3)
        pca_mean    = self.ensemble.pca.mean.reshape(-1, 3)
        n_pca_atoms = len(eigenvector)

        # Build average-structure universe and set protein coords to PCA mean.
        # GRO files don't carry element info; guess from atom names so that
        # write_pca_ref_pdb can write the element column in the output PDB.
        from MDAnalysis.topology import guessers
        u_avg    = u_template.copy()
        elements = guessers.guess_types(u_avg.atoms.names)
        u_avg.add_TopologyAttr('elements', elements)
        tmpl_sel   = u_avg.select_atoms(self.ensemble.rmsd_selection)
        n_template = len(tmpl_sel)

        if n_template != n_pca_atoms:
            raise RuntimeError(
                f'PCA has {n_pca_atoms} atoms but rmsd_selection matches '
                f'{n_template} atoms in the solvated template. '
                f'Check ensemble.rmsd_selection.'
            )

        tmpl_sel.positions = pca_mean
        u_avg.atoms.write(os.path.join(self.output_dir, 'average_structure.pdb'))

        pca_ref_pdb = os.path.join(self.output_dir, 'pca_ref.pdb')
        write_pca_ref_pdb(
            u_structure=u_avg,
            u_template=u_template.copy(),
            eigenvector=eigenvector,
            pca_selection_token=self.ensemble.rmsd_selection,
            output_file=pca_ref_pdb,
        )

        # Per-window plumed.dat
        for window_dir, cv_val in zip(self._window_dirs, cv_values):
            self._write_pca_plumed_file(
                os.path.join(window_dir, 'plumed.dat'),
                cv_value=float(cv_val),
                n_pca_atoms=n_pca_atoms,
                force_constant=force_constant,
                multidir=False,
            )

        # Combined plumed_multidir.dat
        self._write_pca_plumed_file(
            os.path.join(self.output_dir, 'plumed_multidir.dat'),
            cv_value=cv_values,
            n_pca_atoms=n_pca_atoms,
            force_constant=force_constant,
            multidir=True,
        )
        print(f'  PLUMED ({self.path_finder.cv}) input written → {self.output_dir}')

    def _write_plumed_distance_cv(self, cv_values, com_selections,
                                   force_constant, u_template):
        from .plumed import selection_parser

        com1_sel, com2_sel = com_selections
        com1_atoms = selection_parser(com1_sel, u_template)
        com2_atoms = selection_parser(com2_sel, u_template)

        for window_dir, cv_val in zip(self._window_dirs, cv_values):
            self._write_distance_plumed_file(
                os.path.join(window_dir, 'plumed.dat'),
                cv_value=float(cv_val),
                com1_atoms=com1_atoms,
                com2_atoms=com2_atoms,
                force_constant=force_constant,
                multidir=False,
            )

        self._write_distance_plumed_file(
            os.path.join(self.output_dir, 'plumed_multidir.dat'),
            cv_value=cv_values,
            com1_atoms=com1_atoms,
            com2_atoms=com2_atoms,
            force_constant=force_constant,
            multidir=True,
        )
        print(f'  PLUMED (distance) input written → {self.output_dir}')

    def _write_pca_plumed_file(self, output_file, cv_value, n_pca_atoms,
                                force_constant, multidir):
        with open(output_file, 'w') as f:
            f.write('UNITS LENGTH=A\n\n')
            f.write('MOLINFO STRUCTURE=../molinfo.pdb\n\n')
            f.write('pcproj: PCAVARS REFERENCE=../pca_ref.pdb TYPE=OPTIMAL\n\n')
            f.write(f'CV: MATHEVAL ARG=pcproj.eig-1 FUNC=x*{n_pca_atoms} PERIODIC=NO\n\n')
            if multidir:
                cv_str = ','.join(f'{v:.4f}' for v in cv_value)
                f.write(f'umbrella_restraint: RESTRAINT ARG=CV KAPPA={force_constant} '
                        f'AT=@replicas:{cv_str}\n\n')
                f.write('PRINT ARG=CV.*,umbrella_restraint.* FILE=COLVAR_MULTI\n')
            else:
                f.write(f'umbrella_restraint: RESTRAINT ARG=CV KAPPA={force_constant} '
                        f'AT={cv_value:.4f}\n\n')
                f.write('PRINT ARG=CV.*,umbrella_restraint.* FILE=COLVAR\n')

    def _write_distance_plumed_file(self, output_file, cv_value, com1_atoms,
                                     com2_atoms, force_constant, multidir):
        with open(output_file, 'w') as f:
            f.write('UNITS LENGTH=A\n\n')
            f.write('MOLINFO STRUCTURE=../molinfo.pdb\n\n')
            f.write(f'com1: CENTER ATOMS={com1_atoms}\n')
            f.write(f'com2: CENTER ATOMS={com2_atoms}\n\n')
            f.write('CV: DISTANCE ATOMS=com1,com2\n\n')
            if multidir:
                cv_str = ','.join(f'{v:.4f}' for v in cv_value)
                f.write(f'umbrella_restraint: RESTRAINT ARG=CV KAPPA={force_constant} '
                        f'AT=@replicas:{cv_str}\n\n')
                f.write('PRINT ARG=CV,umbrella_restraint.* FILE=COLVAR_MULTI\n')
            else:
                f.write(f'umbrella_restraint: RESTRAINT ARG=CV KAPPA={force_constant} '
                        f'AT={cv_value:.4f}\n\n')
                f.write('PRINT ARG=CV,umbrella_restraint.* FILE=COLVAR\n')

    # ── gmx wrappers ──────────────────────────────────────────────────────────

    def _pdb2gmx(self, window_dir, flags, stdin):
        self._run(
            [self.gmx, 'pdb2gmx',
             '-f', 'input.pdb',
             '-o', 'conf.gro',
             '-p', 'topol.top',
             '-ff', self.force_field,
             '-water', self.water_model,
             ] + flags,
            cwd=window_dir, stdin=stdin, log='pdb2gmx.log',
        )

    def _editconf(self, window_dir, box_type, padding=None, box_dims=None):
        args = [self.gmx, 'editconf',
                '-f', 'conf.gro',
                '-o', 'conf_box.gro',
                '-bt', box_type,
                '-c']
        if box_dims is not None:
            # Cubic: pass all 3 diagonal values.
            # Non-cubic: GROMACS derives the geometry from the box type; only
            # the 'a' lattice length (first value) is needed.
            if box_type == 'cubic':
                args += ['-box'] + [f'{d:.6f}' for d in box_dims[:3]]
            else:
                args += ['-box', f'{box_dims[0]:.6f}']
        else:
            args += ['-d', str(padding)]
        self._run(args, cwd=window_dir, log='editconf.log')

    def _solvate(self, window_dir):
        self._run(
            [self.gmx, 'solvate',
             '-cp', 'conf_box.gro',
             '-cs', 'spc216.gro',
             '-o', 'solvated.gro',
             '-p', 'topol.top'],
            cwd=window_dir, log='solvate.log',
        )

    def _add_ions(self, window_dir, neutralise, concentration, pname, nname):
        with open(os.path.join(window_dir, 'ions.mdp'), 'w') as f:
            f.write(_IONS_MDP)

        self._run(
            [self.gmx, 'grompp',
             '-f', 'ions.mdp',
             '-c', 'solvated.gro',
             '-p', 'topol.top',
             '-o', 'ions.tpr',
             '-maxwarn', '2'],
            cwd=window_dir, log='grompp_ions.log',
        )

        args = [self.gmx, 'genion',
                '-s', 'ions.tpr',
                '-o', 'system.gro',
                '-p', 'topol.top',
                '-pname', pname,
                '-nname', nname]
        if neutralise:
            args.append('-neutral')
        if concentration > 0:
            args += ['-conc', str(concentration)]
        # genion prompts for the group to replace with ions (always SOL here)
        self._run(args, cwd=window_dir, stdin='SOL\n', log='genion.log')

    # ── Combined PDB ─────────────────────────────────────────────────────────

    def _write_combined_pdb(self, gro_name='system.gro', filename='all_windows.pdb'):
        """Write per-window GRO files as MODEL frames in a single PDB.

        Useful for loading all windows as states of one object in PyMOL.
        Written to the root output directory.
        """
        import MDAnalysis as mda

        gro_files = [os.path.join(d, gro_name) for d in self._window_dirs]
        gro_files = [g for g in gro_files if os.path.exists(g)]
        if not gro_files:
            return

        out_pdb = os.path.join(self.output_dir, filename)
        u0 = mda.Universe(gro_files[0])
        with mda.Writer(out_pdb, u0.atoms.n_atoms, multiframe=True) as W:
            for gro in gro_files:
                u = mda.Universe(gro)
                W.write(u.atoms)
        print(f'  Combined PDB ({len(gro_files)} windows) → {out_pdb}')

    # ── Composition harmonisation ─────────────────────────────────────────────

    def _harmonise_composition(self):
        """Trim all non-protein molecule types to the minimum count across windows.

        Handles SOL and any ion types (NA, CL, K, …) generically.  The first
        molecule type in each topology is assumed to be the protein (always true
        for pdb2gmx output) and is skipped.
        """
        all_mol_counts = [
            self._parse_molecules(os.path.join(d, 'topol.top'))
            for d in self._window_dirs
        ]

        # Collect all mol types in topology order, preserving first-seen order.
        seen = set()
        all_types = []
        for counts in all_mol_counts:
            for t in counts:
                if t not in seen:
                    all_types.append(t)
                    seen.add(t)

        protein_mol = all_types[0]  # pdb2gmx always puts protein first

        print('Composition harmonisation:')
        for mol_name in all_types:
            if mol_name == protein_mol:
                continue
            counts = [c.get(mol_name, 0) for c in all_mol_counts]
            n_min, n_max = min(counts), max(counts)
            if n_min == n_max:
                print(f'  {mol_name}: {n_min} (consistent)')
            else:
                print(f'  {mol_name}: trimming to {n_min} (range {n_min}–{n_max})')
                for d, n in zip(self._window_dirs, counts):
                    if n > n_min:
                        self._trim_molecule(
                            os.path.join(d, 'system.gro'),
                            os.path.join(d, 'topol.top'),
                            mol_name, n_min,
                        )

    def _parse_molecules(self, top_file):
        """Return an OrderedDict of mol_name → count from [molecules] section."""
        from collections import OrderedDict
        result = OrderedDict()
        in_mols = False
        with open(top_file) as f:
            for line in f:
                s = line.strip()
                if s.startswith('[') and 'molecules' in s:
                    in_mols = True
                    continue
                if in_mols:
                    if s.startswith('['):
                        break
                    if s and not s.startswith(';'):
                        parts = s.split()
                        if len(parts) >= 2:
                            result[parts[0]] = int(parts[1])
        return result

    def _trim_molecule(self, gro_file, top_file, mol_name, target_n):
        """Remove excess residues of mol_name from GRO and update topology.

        Groups consecutive atom lines into residues by the 5-char residue-number
        field, keeps the first target_n residues of mol_name, and leaves all
        other residues untouched.  Works for multi-atom residues (SOL) and
        single-atom residues (ions) alike.
        """
        with open(gro_file) as f:
            lines = f.readlines()

        title, box_line = lines[0], lines[-1]
        atom_lines = lines[2:-1]

        # Group consecutive atom lines into (resname, [lines]) blocks.
        # GROMACS wraps residue numbers at 99999; for typical system sizes
        # (< 99999 residues total) this is not an issue.
        blocks = []
        cur_rnum, cur_rname, cur_group = None, None, []
        for line in atom_lines:
            rnum  = line[0:5]
            rname = line[5:10].strip()
            if rnum != cur_rnum:
                if cur_group:
                    blocks.append((cur_rname, cur_group))
                cur_rnum, cur_rname, cur_group = rnum, rname, [line]
            else:
                cur_group.append(line)
        if cur_group:
            blocks.append((cur_rname, cur_group))

        # Keep first target_n of mol_name; keep everything else unchanged.
        mol_seen = 0
        kept = []
        for rname, grp in blocks:
            if rname == mol_name:
                if mol_seen < target_n:
                    kept.append(grp)
                    mol_seen += 1
            else:
                kept.append(grp)

        new_lines = [l for g in kept for l in g]
        with open(gro_file, 'w') as f:
            f.write(title)
            f.write(f'{len(new_lines)}\n')
            f.writelines(new_lines)
            f.write(box_line)

        self._update_mol_count_in_top(top_file, mol_name, target_n)

    def _update_mol_count_in_top(self, top_file, mol_name, count):
        """Overwrite the count for mol_name in [molecules] section of topology."""
        top_lines = open(top_file).readlines()
        in_mols = False
        with open(top_file, 'w') as f:
            for line in top_lines:
                s = line.strip()
                if s.startswith('[') and 'molecules' in s:
                    in_mols = True
                if (in_mols and s and not s.startswith(';')
                        and not s.startswith('[')):
                    parts = s.split()
                    if len(parts) >= 2 and parts[0] == mol_name:
                        f.write(f'{mol_name:<16} {count}\n')
                        continue
                f.write(line)

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _read_box_dims(gro_file):
        """Return the box vector values from the last line of a GRO file."""
        with open(gro_file) as f:
            last = f.readlines()[-1]
        return [float(x) for x in last.split()]

    def _run(self, args, cwd, stdin=None, log=None, env=None):
        """Run a GROMACS command, capture output to log, raise on failure."""
        result = subprocess.run(
            args, cwd=cwd,
            input=stdin, capture_output=True, text=True,
            env=env if env is not None else self.env,
        )
        if log:
            with open(os.path.join(cwd, log), 'w') as f:
                f.write(result.stdout + result.stderr)
        if result.returncode != 0:
            log_hint = os.path.join(cwd, log) if log else '(no log)'
            raise RuntimeError(
                f"gmx command failed in {cwd}:\n"
                f"  {' '.join(args)}\n"
                f"See {log_hint} for details.\n"
                f"Last stderr:\n{result.stderr[-800:]}"
            )


class GridSolventSystemPrep(SolventSystemPrep):
    """
    Builds water-only GROMACS systems for each structure in a 2D grid
    and writes 2D PLUMED input files (PC1 + PC2 restraints).

    Inherits the entire gmx pipeline from SolventSystemPrep; only the
    structure source and PLUMED writing are overridden.

    Typical usage::

        prep = GridSolventSystemPrep(grid_finder, ensemble,
                                     output_dir='solvent_2d/')
        prep.build_systems(pdb2gmx_flags=['-ignh'])
        prep.minimise_systems()
        prep.write_plumed_input()
    """

    def __init__(self, grid_finder, ensemble, output_dir,
                 force_field='charmm36-jul2022', water_model='tip3p',
                 gmx='gmx', env=None):
        super().__init__(None, ensemble, output_dir,
                         force_field=force_field, water_model=water_model,
                         gmx=gmx, env=env)
        self.grid_finder = grid_finder
        self._pc1_values = []
        self._pc2_values = []

    # ── Override: structure source ────────────────────────────────────────────

    def build_systems(self, box_padding=1.2, box_type='cubic',
                      pdb2gmx_flags=None, pdb2gmx_stdin=None,
                      neutralise=True, concentration=0.15,
                      pname='NA', nname='CL',
                      aln_selection='protein and name CA'):
        """
        Build a solvated GROMACS system for each active grid cell.

        Parameters are identical to SolventSystemPrep.build_systems except
        path_rank is absent (the best grid from GridPathFinder.run() is always
        used).  A window_map.csv is written to output_dir alongside the windows.
        """
        import MDAnalysis as mda
        import MDAnalysis.analysis.align

        if pdb2gmx_flags is None:
            pdb2gmx_flags = ['-ignh']

        self._box_type = box_type
        self._pname = pname
        self._nname = nname

        os.makedirs(self.output_dir, exist_ok=True)
        targets = self.grid_finder.get_window_targets()

        self._pc1_values = [t[1] for t in targets]
        self._pc2_values = [t[2] for t in targets]

        u_ref = None
        self._window_dirs = []
        for i, ((bin_i, bin_j), (structure, _pc1, _pc2)) in enumerate(
                tqdm(zip(self.grid_finder._active_cells, targets),
                     total=len(targets), desc='Building systems')):
            window_dir = os.path.join(self.output_dir, f'window_{bin_i}_{bin_j}')
            os.makedirs(window_dir, exist_ok=True)
            self._window_dirs.append(window_dir)

            src = self.ensemble.structure_directory + structure
            input_pdb = os.path.join(window_dir, 'input.pdb')

            if aln_selection and i > 0:
                u = mda.Universe(src, src)
                mda.analysis.align.alignto(u, u_ref, select=aln_selection)
                u.atoms.write(input_pdb)
            else:
                shutil.copy(src, input_pdb)
                if aln_selection and i == 0:
                    u_ref = mda.Universe(src, src)

            self._pdb2gmx(window_dir, pdb2gmx_flags, pdb2gmx_stdin)

            if i == 0:
                self._editconf(window_dir, box_type=box_type, padding=box_padding)
                self._box_dims = self._read_box_dims(
                    os.path.join(window_dir, 'conf_box.gro'))
            else:
                self._editconf(window_dir, box_type=box_type,
                               box_dims=self._box_dims)

            self._solvate(window_dir)
            self._add_ions(window_dir, neutralise=neutralise,
                           concentration=concentration,
                           pname=pname, nname=nname)

        self._harmonise_composition()
        self._write_combined_pdb()
        self.grid_finder.write_grid_summary(
            os.path.join(self.output_dir, 'window_map.csv'))
        return self

    # ── Override: PLUMED ──────────────────────────────────────────────────────

    def write_plumed_input(self, force_constant=1000, use_minimised=False):
        """
        Write per-window plumed.dat files and plumed_multidir.dat with
        two PCA restraints (PC1 and PC2).

        Parameters
        ----------
        force_constant : float
            Harmonic restraint force constant (kJ mol⁻¹ nm⁻²).
        use_minimised : bool
            If True, use em.gro as the topology reference (call after
            minimise_systems()).

        Returns
        -------
        self
        """
        import MDAnalysis as mda

        if not self._window_dirs:
            raise RuntimeError('Call build_systems() before write_plumed_input()')

        gro_name = 'em.gro' if use_minimised else 'system.gro'
        ref_gro = os.path.join(self._window_dirs[0], gro_name)
        u_template = mda.Universe(ref_gro)

        molinfo_pdb = os.path.join(self.output_dir, 'molinfo.pdb')
        u_template.atoms.write(molinfo_pdb)

        self._write_plumed_2d_pca_cv(
            self._pc1_values, self._pc2_values, force_constant, u_template)
        return self

    def _write_plumed_2d_pca_cv(self, pc1_values, pc2_values,
                                  force_constant, u_template):
        import numpy as np
        from MDAnalysis.topology import guessers
        from .structure_prep import write_pca_ref_pdb

        if self.ensemble.pca is None:
            raise RuntimeError('PCA not available — call ensemble.run_pca() first')

        cv1 = self.grid_finder.cv1
        cv2 = self.grid_finder.cv2
        pc1_idx = int(cv1[2:]) - 1 if cv1.startswith('PC') and cv1[2:].isdigit() else 0
        pc2_idx = int(cv2[2:]) - 1 if cv2.startswith('PC') and cv2[2:].isdigit() else 1

        ev1 = self.ensemble.pca.p_components[:, pc1_idx].reshape(-1, 3)
        ev2 = self.ensemble.pca.p_components[:, pc2_idx].reshape(-1, 3)
        pca_mean = self.ensemble.pca.mean.reshape(-1, 3)
        n_pca_atoms = len(ev1)

        u_avg = u_template.copy()
        elements = guessers.guess_types(u_avg.atoms.names)
        u_avg.add_TopologyAttr('elements', elements)
        tmpl_sel = u_avg.select_atoms(self.ensemble.rmsd_selection)
        n_template = len(tmpl_sel)

        if n_template != n_pca_atoms:
            raise RuntimeError(
                f'PCA has {n_pca_atoms} atoms but rmsd_selection matches '
                f'{n_template} atoms in the solvated template. '
                f'Check ensemble.rmsd_selection.'
            )

        tmpl_sel.positions = pca_mean
        u_avg.atoms.write(os.path.join(self.output_dir, 'average_structure.pdb'))

        pca_ref_pc1 = os.path.join(self.output_dir, 'pca_ref_pc1.pdb')
        pca_ref_pc2 = os.path.join(self.output_dir, 'pca_ref_pc2.pdb')

        write_pca_ref_pdb(u_structure=u_avg, u_template=u_template.copy(),
                          eigenvector=ev1,
                          pca_selection_token=self.ensemble.rmsd_selection,
                          output_file=pca_ref_pc1)
        write_pca_ref_pdb(u_structure=u_avg, u_template=u_template.copy(),
                          eigenvector=ev2,
                          pca_selection_token=self.ensemble.rmsd_selection,
                          output_file=pca_ref_pc2)

        for window_dir, pc1_val, pc2_val in zip(
                self._window_dirs, pc1_values, pc2_values):
            self._write_2d_pca_plumed_file(
                os.path.join(window_dir, 'plumed.dat'),
                pc1_value=float(pc1_val), pc2_value=float(pc2_val),
                n_pca_atoms=n_pca_atoms, force_constant=force_constant,
                multidir=False,
            )

        self._write_2d_pca_plumed_file(
            os.path.join(self.output_dir, 'plumed_multidir.dat'),
            pc1_value=pc1_values, pc2_value=pc2_values,
            n_pca_atoms=n_pca_atoms, force_constant=force_constant,
            multidir=True,
        )
        print(f'  PLUMED (PC1 + PC2) input written → {self.output_dir}')

    def _write_2d_pca_plumed_file(self, output_file, pc1_value, pc2_value,
                                   n_pca_atoms, force_constant, multidir):
        with open(output_file, 'w') as f:
            f.write('UNITS LENGTH=A\n\n')
            f.write('MOLINFO STRUCTURE=../molinfo.pdb\n\n')
            f.write('pcproj1: PCAVARS REFERENCE=../pca_ref_pc1.pdb TYPE=OPTIMAL\n')
            f.write('pcproj2: PCAVARS REFERENCE=../pca_ref_pc2.pdb TYPE=OPTIMAL\n\n')
            f.write(f'CV1: MATHEVAL ARG=pcproj1.eig-1 FUNC=x*{n_pca_atoms} PERIODIC=NO\n')
            f.write(f'CV2: MATHEVAL ARG=pcproj2.eig-1 FUNC=x*{n_pca_atoms} PERIODIC=NO\n\n')
            if multidir:
                cv1_str = ','.join(f'{v:.4f}' for v in pc1_value)
                cv2_str = ','.join(f'{v:.4f}' for v in pc2_value)
                f.write(f'restraint_pc1: RESTRAINT ARG=CV1 KAPPA={force_constant} '
                        f'AT=@replicas:{cv1_str}\n')
                f.write(f'restraint_pc2: RESTRAINT ARG=CV2 KAPPA={force_constant} '
                        f'AT=@replicas:{cv2_str}\n\n')
                f.write('PRINT ARG=CV1,CV2,restraint_pc1.*,restraint_pc2.* '
                        'FILE=COLVAR_MULTI\n')
            else:
                f.write(f'restraint_pc1: RESTRAINT ARG=CV1 KAPPA={force_constant} '
                        f'AT={pc1_value:.4f}\n')
                f.write(f'restraint_pc2: RESTRAINT ARG=CV2 KAPPA={force_constant} '
                        f'AT={pc2_value:.4f}\n\n')
                f.write('PRINT ARG=CV1,CV2,restraint_pc1.*,restraint_pc2.* '
                        'FILE=COLVAR\n')
