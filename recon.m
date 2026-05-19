%% PTO6/STO6 labyrinth recon - fold_slice / PtychoShelves (Yi Jiang's fork)
%
% Adapted from fold_slice/ptycho/examples/PSO_science/ptycho_electron_PSO_science.m
% Values are GUESS PARAMS while we wait for Yu Lei's reply. The TODO[Yu]
% lines are the ones to overwrite with his recipe.
%
% Data file: <project>/1/data_roi0_Ndp65_dp.hdf5  -- 65x65 CBED, 6400 pos,
%   produced by `python export_to_foldslice.py --pipeline`.
%
% Run from inside fold_slice/ptycho/:
%   cd c:\Users\Trist\fold_slice\ptycho
%   addpath('c:\Users\Trist\HyperSpy-bundle\MY HYPERSPY CODE\PTOSTO');
%   recon
%
% Sign-convention note: fold_slice probe_df uses NEGATIVE for overfocus
% (matches abTEM, opposite py4DSTEM's C10). Our overfocus is +50 A so
% probe_df = -50 A. The export writes `defocus_a = -50` already.

% Smoke override: set SMOKE_TEST_NITER before launching to shrink iter counts.
%   matlab -batch "SMOKE_TEST_NITER=3; cd 'c:\Users\Trist\fold_slice\ptycho'; addpath('c:\Users\Trist\HyperSpy-bundle\MY HYPERSPY CODE\PTOSTO'); recon"
if exist('SMOKE_TEST_NITER', 'var')
    smoke_iter_override = SMOKE_TEST_NITER;
else
    smoke_iter_override = 0;
end
if exist('SMOKE_SIMPLE', 'var')
    smoke_simple = SMOKE_SIMPLE;
else
    smoke_simple = false;
end
clearvars -except smoke_iter_override smoke_simple
addpath(strcat(pwd,'/utils/'))
addpath(core.find_base_package)

%%%%%%%%%%%%%%%%%%%% data parameters %%%%%%%%%%%%%%%%%%%%
base_path = 'c:\Users\Trist\HyperSpy-bundle\MY HYPERSPY CODE\PTOSTO\';
roi_label = '0_Ndp65';
scan_number = 1;
scan_string_format = '%01d';
Ndpx = 65;            % size of cbed (after square-resample to coarse axis)
alpha0 = 100;         % semi-convergence angle (mrad)  -- ours, vs PSO's 21.4
rbf = 16.21;          % BF disk radius in pixels = alpha0 / d_alpha_mrad
voltage = 300;        % kV
rot_ang = 0;          % cbed vs scan rotation -- abTEM data has known zero

scan_step_size = 0.25;   % angstrom (0.25 A vs PSO's 0.41)
N_scan_y = 80;           % 80x80 = 6400 positions (PSO was 64x64)
N_scan_x = 80;

%%%%%%%%%%%%%%%%%%%% reconstruction parameters %%%%%%%%%%%%%%%%%%%%
gpu_id = 1;
Niter_save_results = 50;
Niter_plot_results = 50;

% Probe modes: PSO paper used 8. With our 8 GB local Windows GPU and
% 74-layer multislice + 80x80 grid, that's tight. Start at 4 and bump
% if convergence is poor. TODO[Yu]: confirm or override.
Nprobe = 4;

% Object thickness/slice plan: 74 A / 37 slices = 2 A per slice.
% Sits AT the optical-depth limit dz_optical = lambda/alpha^2 ~ 1.97 A.
% 1 A slices (=74 layers) caused NaN at iter 1 -- too-fine multislice
% triggers Fresnel propagator instability with the current grid sampling.
thickness = 73.93;
Nlayers = 37;
delta_z = thickness / Nlayers;

% Smoke isolation: collapse to Nprobe=1, Nlayers=1 to diagnose whether NaN-at-
% iter-1 is from data/probe setup vs multislice/multimode. Set SMOKE_SIMPLE
% to a string: 'baseline' (1+1), 'multimode' (Nprobe>1, Nlayers=1),
% 'multislice' (Nprobe=1, Nlayers=74).
if ischar(smoke_simple) || isstring(smoke_simple)
    switch char(smoke_simple)
        case 'baseline'
            Nprobe = 1; Nlayers = 1; delta_z = thickness;
        case 'multimode'
            Nprobe = 4; Nlayers = 1; delta_z = thickness;
        case 'multislice'
            Nprobe = 1; Nlayers = 74; delta_z = thickness / Nlayers;
        case 'multislice37'   % at dz_optical = lambda/alpha^2 ~ 2 A
            Nprobe = 1; Nlayers = 37; delta_z = thickness / Nlayers;
        case 'multislice18'   % 4 A slices, 2x dz_optical
            Nprobe = 1; Nlayers = 18; delta_z = thickness / Nlayers;
    end
    fprintf('[SMOKE-SIMPLE=%s] Nprobe=%d, Nlayers=%d, delta_z=%.2f\n', ...
        char(smoke_simple), Nprobe, Nlayers, delta_z);
elseif smoke_simple
    Nprobe = 1; Nlayers = 1; delta_z = thickness;
    fprintf('[SMOKE-SIMPLE] Forced Nprobe=1, Nlayers=1, delta_z=%.1f\n', delta_z);
end

%% %%%%%%%%%%%%%%%%%% initialize data parameters %%%%%%%%%%%%%%%%%%%%
p = struct();
p.   verbose_level = 2;
p.   use_display = false;
p.   scan_number = scan_number;

% Geometry
p.   z = 1;
p.   asize = [Ndpx, Ndpx];
p.   ctr = [fix(Ndpx/2)+1, fix(Ndpx/2)+1];
p.   beam_source = 'electron';
p.   d_alpha = alpha0/rbf;                  % mrad/px  -- 100/16.21 = 6.17 (our coarse axis)
p.   prop_regime = 'farfield';
p.   focus_to_sample_distance = [];
p.   energy = voltage;

affine_mat  = compose_affine_matrix(1, 0, rot_ang, 0);
p.   affine_matrix = affine_mat;

% Scan meta data
p.   src_metadata = 'none';
p.   queue.lockfile = false;

% Data preparation
p.   detector.name = 'empad';
p.   detector.check_2_detpos = [];
p.   detector.data_prefix = '';
p.   detector.binning = false;
p.   detector.upsampling = false;
p.   detector.burst_frames = 1;

p.   prepare.data_preparator = 'matlab_aps';
p.   prepare.auto_prepare_data = true;          % run matlab_aps prep on our exported source file
p.   prepare.force_preparation_data = true;     % always re-prep on each run
p.   prepare.store_prepared_data = false;
p.   prepare.prepare_data_function = '';
p.   prepare.auto_center_data = false;

% Scan positions -- matlab_pos raster (fold_slice generates grid from below)
p.   src_positions = 'matlab_pos';
p.   positions_file = '';
p.   scan.type = 'raster';
p.   scan.roi_label = roi_label;
p.   scan.format = scan_string_format;
p.   scan.radius_in = 0;
p.   scan.radius_out = 5e-6;
p.   scan.nr = 10;
p.   scan.nth = 3;
p.   scan.lx = 20e-6;
p.   scan.ly = 20e-6;
p.   scan.dr = 1.5e-6;
p.   scan.nx = N_scan_x;
p.   scan.ny = N_scan_y;
p.   scan.step_size_x = scan_step_size;
p.   scan.step_size_y = scan_step_size;
p.   scan.custom_flip = [1,1,1];
p.   scan.step_randn_offset = 0;
p.   scan.b = 0;
p.   scan.n_max = 1e4;
p.   scan.step = 0.5e-6;
p.   scan.cenxy = [0,0];
p.   scan.roi = [];
p.   scan.custom_positions_source = '';
p.   scan.custom_params = [];

% I/O
p.   prefix = '';
p.   suffix = strcat('PTOSTO_W1_test');
p.   scan_string_format = scan_string_format;
p.   base_path = base_path;
p.   specfile = '';
p.   ptycho_matlab_path = '';
p.   cSAXS_matlab_path = '';
p.   raw_data_path{1} = '';
p.   prepare_data_path = '';
p.   prepare_data_filename = [];
p.   save_path{1} = '';
p.   io.default_mask_file = '';
p.   io.default_mask_type = 'binary';
p.   io.file_compression = 0;
p.   io.data_compression = 3;
p.   io.load_prep_pos = false;

p.   io.data_descriptor = 'PTOSTO labyrinth W1 test';
p.   io.phone_number = '';
p.   io.send_failed_scans_SMS = false;
p.   io.send_finished_recon_SMS = false;
p.   io.send_crashed_recon_SMS = false;
p.   io.SMS_sleep = 1800;
p.   io.script_name = mfilename;

p.   artificial_data_file = 'template_artificial_data';

%% Reconstruction
% Initial iterate object
p.   model_object = true;
p.   model.object_type = 'rand';
p.   initial_iterate_object_file{1} = '';

% Initial iterate probe (model from optical params)
p.   model_probe = true;
p.   model.probe_alpha_max = alpha0;
% Sign convention: fold_slice probe_df = -overfocus.  Our overfocus = +50 A
% (crossover above sample) => probe_df = -50 A.
p.   model.probe_df = -50;                  % TODO[Yu]: confirm sign
p.   model.probe_c3 = 0;
p.   model.probe_c5 = 0;
p.   model.probe_c7 = 0;
p.   model.probe_f_a2 = 0;
p.   model.probe_theta_a2 = 0;
p.   model.probe_f_a3 = 0;
p.   model.probe_theta_a3 = 0;
p.   model.probe_f_c3 = 0;
p.   model.probe_theta_c3 = 0;

p.   initial_probe_file = '';
p.   probe_file_propagation = 0.0e-3;
p.   normalize_init_probe = true;

p.   share_probe  = 0;
p.   share_object = 0;

% Modes
p.   probe_modes  = Nprobe;
p.   object_modes = 1;

p.   mode_start_pow = 0.02;
p.   mode_start = 'herm';
p.   ortho_probes = true;

%% Plot, save and analyze
p.   plot.prepared_data = false;
p.   plot.interval = [];
p.   plot.log_scale = [0 0];
p.   plot.realaxes = true;
p.   plot.remove_phase_ramp = false;
p.   plot.fov_box = false;
p.   plot.fov_box_color = 'r';
p.   plot.positions = true;
p.   plot.mask_bool = true;
p.   plot.windowautopos = true;
p.   plot.obj_apod = false;
p.   plot.prop_obj = 0;
p.   plot.show_layers = true;
p.   plot.show_layers_stack = false;
p.   plot.object_spectrum = [];
p.   plot.probe_spectrum = [];
p.   plot.conjugate = false;
p.   plot.horz_fact = 2.5;
p.   plot.FP_maskdim = 180e-6;
p.   plot.calc_FSC = false;
p.   plot.show_FSC = false;
p.   plot.residua = false;

p.   save.external = true;
p.   save.store_images = true;
p.   save.store_images_intermediate = false;
p.   save.store_images_ids = 1:4;
p.   save.store_images_format = 'png';
p.   save.store_images_dpi = 150;
p.   save.exclude = {'fmag', 'fmask', 'illum_sum'};
p.   save.save_reconstructions_intermediate = false;
p.   save.save_reconstructions = true;
p.   save.output_file = 'mat';                     % .mat so import_foldslice_result.py can read it

%% %%%%%%%%%%%%%%%%%% Stage A: presolve at low resolution %%%%%%%%%%%%%%%%%%%%
eng = struct();
eng. name = 'GPU_MS';
eng. use_gpu = true;
eng. keep_on_gpu = true;
eng. compress_data = false;
eng. gpu_id = gpu_id;
eng. check_gpu_load = true;

if smoke_iter_override > 0
    eng. number_iterations = smoke_iter_override;
    fprintf('[SMOKE] Stage A iterations overridden to %d\n', smoke_iter_override);
else
    eng. number_iterations = 40;               % [Yu]: 40 for test, 200 for prod -- Stage A = warmup
end
eng. asize_presolve = [];                      % full res in Stage A (CBED already only 65x65)
eng. align_shared_objects = false;

eng. method = 'MLs';                           % LSQ-ML (Stochastic) -- what the paper uses
eng. opt_errmetric = 'L1';                     % [Yu]: 'L1'
eng. grouping = 32;                            % [Yu]: smaller = slower but better convergence; cap is GPU mem (8 GB)
eng. probe_modes  = p.probe_modes;
eng. object_change_start = 1;
eng. probe_change_start = eng.number_iterations;  % Stage A: probe fixed entire stage (warmup)

% regularizations -- Yu reply gave defaults (=0), but the L-curve sweep
% showed BOTH per-slice std and kz-FRC ratios drop monotonically with iter.
% Hypothesis: multislice has a gauge degeneracy (phase swappable between
% adjacent slices without changing diffraction prediction). On Yu's
% experimental data, Poisson noise breaks the degeneracy. On our noiseless
% abTEM data the engine has nothing to constrain it -> spurious gauge
% swaps amplify with iter. Trial: gentle Tikhonov + weak layer-symmetry
% regularisation. PSO_science example used regularize_layers=1; we use 0.01
% because 1 would force layers identical = the failure mode we're testing for.
eng. reg_mu = 1e-3;
eng. delta = 0;
eng. positivity_constraint_object = 0;
eng. apply_multimodal_update = false;
eng. probe_backpropagate = 0;
eng. probe_support_radius = [];
eng. probe_support_fft = false;

% basic recon
eng. beta_object = 1;
eng. beta_probe = 1;
eng. beta_LSQ = 0.5;                           % reduced from 1 (default) — fold_slice's NaN-recovery suggestion when error diverges at iter 1
eng. delta_p = 0.1;                            % LSQ preconditioner (THE py4DSTEM-vs-fold_slice differentiator)
eng. momentum = 0;                             % [Yu]: 0 (not 0.5; momentum was NOT the engine differentiator)
eng. accelerated_gradients_start = inf;

eng. pfft_relaxation = 0.05;
eng. probe_regularization = 0.1;

% Position refinement -- the OTHER engine feature py4DSTEM doesn't have
% (or has but doesn't use). Stage A: don't refine yet, wait for Stage B.
eng. apply_subpix_shift = true;
eng. probe_position_search = inf;              % Stage A: positions fixed
eng. probe_geometry_model = {};
eng. probe_position_error_max = inf;
eng. apply_relaxed_position_constraint = false;
eng. update_pos_weight_every = inf;

% Multilayer
eng. delta_z = delta_z * ones(Nlayers, 1);
% Layer-symmetrisation OFF for sweep #3. The 0.01 setting in the previous
% sweep killed kz-FRC (0.087 -> 0.047) because it literally penalises
% slice-to-slice differences = our depth signal. Hypothesis: reg_mu alone
% (in-plane Tikhonov) is enough to break the gauge degeneracy without
% directly suppressing depth structure.
eng. regularize_layers = 0;
eng. preshift_ML_probe = false;
eng. layer4pos = [];
eng. init_layer_select = [];
eng. init_layer_preprocess = '';
eng. init_layer_interp = [];
eng. init_layer_append_mode = '';
eng. init_layer_scaling_factor = 1;

% other extensions
eng. background = 0;
eng. background_width = inf;
eng. clean_residua = false;

% wavefront & camera geometry refinement
eng. probe_fourier_shift_search = inf;
eng. estimate_NF_distance = inf;
eng. detector_rotation_search = inf;           % abTEM rotation is known zero
eng. detector_scale_search = inf;
eng. variable_probe = true;
eng. variable_probe_modes = 1;
eng. variable_probe_smooth = 0;
eng. variable_intensity = false;

eng. get_fsc_score = false;
eng. mirror_objects = false;

eng.auto_center_data = false;
eng.auto_center_probe = false;
eng.custom_data_flip = [0,0,0];
eng.apply_tilted_plane_correction = '';

% I/O
eng.plot_results_every = Niter_plot_results;
% Save at the final iter (== number_iterations). Without this, smoke runs
% with Niter < default-save-every produce no .mat at all.
eng.save_results_every = eng.number_iterations;
eng.save_images = {'obj_ph_stack','obj_ph_sum','probe','probe_mag','probe_prop_mag'};
eng.extraPrintInfo = strcat('PTOSTO_StageA');

resultDir = strcat(p.base_path, sprintf(p.scan.format, p.scan_number), '/roi', p.scan.roi_label, '/');
[eng.fout, p.suffix] = generateResultDir(eng, resultDir);

[p, ~] = core.append_engine(p, eng);

%% %%%%%%%%%%%%%%%%%% Stage B: full recon with probe + position refinement %%%%
if smoke_iter_override > 0
    eng. number_iterations = smoke_iter_override;
    eng. probe_position_search = inf;          % smoke: positions fixed (sim recipe)
    fprintf('[SMOKE] Stage B iterations overridden to %d\n', smoke_iter_override);
else
    % Sweep #3: reg_mu=1e-3 ONLY (regularize_layers=0). Looking for whether
    % Tikhonov alone gives a clean L-curve AND preserves kz-FRC > 0.087.
    % Bumping back to 80 iter to see the full curve.
    eng. number_iterations = 80;
    eng. probe_position_search = inf;          % [Yu, sim recipe]: never refine
end
eng. asize_presolve = [];
eng. grouping = 32;                            % [Yu]: GPU-mem limited; 32 fits 8 GB
% [Yu, sim recipe]: probe is perfect (abTEM-constructed) -> never refine.
% Original `1` released probe at Stage B iter 1, and after Stage A had
% deeply converged the object with the fixed probe, the first probe-update
% step overshot to NaN. For experimental data, set this back to 20.
eng. probe_change_start = inf;
% L-curve sweep mode: dump every 10 iter for snapshots at 10, 20, ..., 80.
% Switch back to `eng.number_iterations` for a final single-iter save.
eng. save_results_every = 10;

[eng.fout, p.suffix] = generateResultDir(eng, resultDir);
[p, ~] = core.append_engine(p, eng);

%% Run
tic
out = core.ptycho_recons(p);
toc

%% Locate the final .mat for import_foldslice_result.py.
% fold_slice saves to <base_path>analysis/S<XXXXX-YYYYY>/S<scan>/*_recons.mat,
% NOT into eng.fout. Grab the most recent one.
analysis_glob = fullfile(p.base_path, 'analysis', '**', sprintf('S%05d_*_recons.mat', p.scan_number));
mats = dir(analysis_glob);
if isempty(mats)
    fprintf('\n[warn] No *_recons.mat found under %s — recon may have errored\n', fullfile(p.base_path, 'analysis'));
    final_mat = '<MAT_NOT_FOUND>';
else
    [~, ix] = max([mats.datenum]);
    final_mat = fullfile(mats(ix).folder, mats(ix).name);
end
fprintf('\nDONE. Recon saved to:\n  %s\n\nNext step (in python):\n', final_mat);
fprintf('  python import_foldslice_result.py --mat "%s" --out ptycho_recon_foldslice.zarr\n', final_mat);
fprintf('  python validate.py --recon ptycho_recon_foldslice.zarr\n');
