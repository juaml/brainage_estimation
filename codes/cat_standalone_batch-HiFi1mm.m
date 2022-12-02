% Batch file for CAT12 segmentation, mean GM ROI extraction to csv,
% TIV & tissue volume extraction for SPM12 standalone installation
%
%   ** eats GZIP for real now **
%
% ----> CREATE INDIVIDUAL OUTPUT FOLDER BEFORE <----
%
%  input:
%  <*T1w.nii.gz>
%_______________________________________________________________________
% $Id: cat_standalone_batch.m r1871
%-----------------------------------------------------------------------

% Used CAT12.8 r1871

% INPUT FILE
matlabbatch{1}.spm.tools.cat.estwrite.data(1) = '<UNDEFINED>';
matlabbatch{1}.spm.tools.cat.estwrite.data_wmh = {''};
matlabbatch{1}.spm.tools.cat.estwrite.nproc = 0;
matlabbatch{1}.spm.tools.cat.estwrite.useprior = '';
% Remove comments if you would like to change TPM by using additional arguments in cat_standalone.sh
% or change this field manually by editing the "<UNDEFINED" field
% Otherwise the default value from cat_defaults.m is used.
% 2st parameter field, that will be dynamically replaced by cat_standalone.sh
%matlabbatch{1}.spm.tools.cat.estwrite.opts.tpm = {'<UNDEFINED>'};

% Affine regularisation (SPM12 default = mni) - '';'mni';'eastern';'subj';'none';'rigid'
matlabbatch{1}.spm.tools.cat.estwrite.opts.affreg = 'mni';

% Strength of the bias correction that controls the biasreg and biasfwhm parameter (CAT only!)
% 0 - use SPM parameter; eps - ultralight, 0.25 - light, 0.5 - medium, 0.75 - strong, and 1 - heavy corrections
% job.opts.biasreg	= min(  10 , max(  0 , 10^-(job.opts.biasstr*2 + 2) ));
% job.opts.biasfwhm	= min( inf , max( 30 , 30 + 60*job.opts.biasstr ));
matlabbatch{1}.spm.tools.cat.estwrite.opts.biasstr = 0.5;
%Overview  of parameters:   accstr:  0.50   0.75   1.00  samp:    3.00   2.00   1.00 (in mm)  tol:     1e-4   1e-8   1e-16SPM default is samp
matlabbatch{1}.spm.tools.cat.estwrite.opts.accstr = 0.8;
% Use center-of-mass to roughly correct for differences in the position between image and template. This will internally correct the origin.
matlabbatch{1}.spm.tools.cat.estwrite.extopts.segmentation.setCOM = 1;
% Affine PreProcessing (APP) with rough bias correction and brain extraction for special anatomies (nonhuman/neonates)
% 0 - none; 1070 - default; [1 - light; 2 - full; 1144 - update of 1070, 5 - animal (no affreg)]
matlabbatch{1}.spm.tools.cat.estwrite.extopts.segmentation.APP = 1070;
matlabbatch{1}.spm.tools.cat.estwrite.extopts.segmentation.affmod = 0;
% Strength of the noise correction: 0 to 1; 0 - no filter, -Inf - auto, 1 - full, 2 - ISARNLM (else SANLM), default -Inf
matlabbatch{1}.spm.tools.cat.estwrite.extopts.segmentation.NCstr = -Inf;
matlabbatch{1}.spm.tools.cat.estwrite.extopts.segmentation.spm_kamap = 0;
% Strength of the local adaption: 0 to 1; default 0.5
matlabbatch{1}.spm.tools.cat.estwrite.extopts.segmentation.LASstr = 0.5;
matlabbatch{1}.spm.tools.cat.estwrite.extopts.segmentation.LASmyostr = 0;
% Strength of skull-stripping: 0 - SPM approach; eps to 1  - gcut; 2 - new APRG approach; -1 - no skull-stripping (already skull-stripped); default = 2
matlabbatch{1}.spm.tools.cat.estwrite.extopts.segmentation.gcutstr = 2;
% Strength of the cleanup process: 0 to 1; default 0.5
matlabbatch{1}.spm.tools.cat.estwrite.extopts.segmentation.cleanupstr = 0.5;
matlabbatch{1}.spm.tools.cat.estwrite.extopts.segmentation.BVCstr = 0.5;
% Correction of WM hyperintensities: 0 - no correction, 1 - only for Dartel/Shooting
% 2 - also correct segmentation (to WM), 3 - handle as separate class; default 1
matlabbatch{1}.spm.tools.cat.estwrite.extopts.segmentation.WMHC = 2;
% Stroke lesion correction (SLC): 0 - no correction, 1 - handling of manual lesion that have to be set to zero!
% 2 - automatic lesion detection (in development)
matlabbatch{1}.spm.tools.cat.estwrite.extopts.segmentation.SLC = 0;
matlabbatch{1}.spm.tools.cat.estwrite.extopts.segmentation.mrf = 1;
% % resolution handling: 'native','fixed','best', 'optimal'
% matlabbatch{1}.spm.tools.cat.estwrite.extopts.segmentation.restypes.optimal = [1 0.1];
% Remove comments and edit entry if you would like to change the Dartel/Shooting approach
% Otherwise the default value from cat_defaults.m is used.
% entry for choosing shooting approach
matlabbatch{1}.spm.tools.cat.estwrite.extopts.registration.regmethod.shooting.shootingtpm = {fullfile('/', 'templates_1mm', 'Template_0_GS1mm.nii')};
%matlabbatch{1}.spm.tools.cat.estwrite.extopts.registration.regmethod.shooting.shootingtpm = {fullfile(spm('dir'),'toolbox','cat12','templates_MNI152NLin2009cAsym','Template_1_GS.nii')};
% entry for choosing dartel approach
%matlabbatch{1}.spm.tools.cat.estwrite.extopts.registration.regmethod.dartel.darteltpm = {fullfile(spm('dir'),'toolbox','cat12','templates_MNI152NLin2009cAsym','Template_1_Dartel.nii')};

% Strength of Shooting registration: 0 - Dartel, eps (fast), 0.5 (default) to 1 (accurate) optimized Shooting, 4 - default Shooting; default 0.5
matlabbatch{1}.spm.tools.cat.estwrite.extopts.registration.regmethod.shooting.regstr = 1;
matlabbatch{1}.spm.tools.cat.estwrite.extopts.registration.vox = 1;
matlabbatch{1}.spm.tools.cat.estwrite.extopts.registration.bb = 45;

% surface and thickness creation:   0 - no (default), 1 - lh+rh, 2 - lh+rh+cerebellum,
% 3 - lh, 4 - rh, 5 - lh+rh (fast, no registration, only for quick quality check and not for analysis),
% 6 - lh+rh+cerebellum (fast, no registration, only for quick quality check and not for analysis)
% 9 - thickness only (for ROI analysis, experimental!)
% +10 to estimate WM and CSF width/depth/thickness (experimental!)
matlabbatch{1}.spm.tools.cat.estwrite.output.surface = 1;
matlabbatch{1}.spm.tools.cat.estwrite.output.surf_measures = 1;
% surface options
matlabbatch{1}.spm.tools.cat.estwrite.extopts.surface.pbtres = 0.5;
matlabbatch{1}.spm.tools.cat.estwrite.extopts.surface.pbtmethod = 'pbt2x';
 % surface recontruction pipeline & self-intersection correction: 0/1 - CS1 without/with/with-optimized SIC; 20/21/22 - CS2 without/with/with-optimized SIC;
matlabbatch{1}.spm.tools.cat.estwrite.extopts.surface.SRP = 22;
 % optimize surface sampling: 0 - PBT res. (slow); 1 - optimal res. (default); 2 - internal res.; 3 - SPM init; 4 - MATLAB init; 5 - SPM full;
matlabbatch{1}.spm.tools.cat.estwrite.extopts.surface.reduce_mesh = 1;
matlabbatch{1}.spm.tools.cat.estwrite.extopts.surface.vdist = 2;
% % reduce myelination effects (experimental, not yet working properly!)
% matlabbatch{1}.spm.tools.cat.estwrite.extopts.pbtlas         = 0;
% % distance method for estimating thickness:  1 - Tfs: Freesurfer method using mean(Tnear1,Tnear2) (default in 12.7+); 0 - Tlink: linked distance (used before 12.7)
% matlabbatch{1}.spm.tools.cat.estwrite.extopts.thick_measure  = 1;
% % upper limit for Tfs thickness measure similar to Freesurfer (only valid if cat.extopts.thick_measure is set to "1"
% matlabbatch{1}.spm.tools.cat.estwrite.extopts.thick_limit    = 5;

matlabbatch{1}.spm.tools.cat.estwrite.extopts.surface.scale_cortex = 0.7;
matlabbatch{1}.spm.tools.cat.estwrite.extopts.surface.add_parahipp = 0.1;
matlabbatch{1}.spm.tools.cat.estwrite.extopts.surface.close_parahipp = 1;
matlabbatch{1}.spm.tools.cat.estwrite.extopts.admin.experimental = 0;
matlabbatch{1}.spm.tools.cat.estwrite.extopts.admin.new_release = 0;
% set this to 1 for skipping preprocessing if already processed data exist
matlabbatch{1}.spm.tools.cat.estwrite.extopts.admin.lazy = 0;
% catch errors: 0 - stop with error (default); 1 - catch preprocessing errors (requires MATLAB 2008 or higher);
matlabbatch{1}.spm.tools.cat.estwrite.extopts.admin.ignoreErrors = 1;
% verbose output: 1 - default; 2 - details; 3 - write debugging files
matlabbatch{1}.spm.tools.cat.estwrite.extopts.admin.verb = 2;
% display and print out pdf-file of results: 0 - off, 1 - volume only, 2 - volume and surface (default)
matlabbatch{1}.spm.tools.cat.estwrite.extopts.admin.print = 2;
matlabbatch{1}.spm.tools.cat.estwrite.output.BIDS.BIDSno = 1;

% define here volume atlases
matlabbatch{1}.spm.tools.cat.estwrite.output.ROImenu.atlases.neuromorphometrics = 1;
matlabbatch{1}.spm.tools.cat.estwrite.output.ROImenu.atlases.lpba40 = 1;
matlabbatch{1}.spm.tools.cat.estwrite.output.ROImenu.atlases.cobra = 1;
matlabbatch{1}.spm.tools.cat.estwrite.output.ROImenu.atlases.hammers = 1;
matlabbatch{1}.spm.tools.cat.estwrite.output.ROImenu.atlases.ibsr = 1;
matlabbatch{1}.spm.tools.cat.estwrite.output.ROImenu.atlases.aal3 = 1;
matlabbatch{1}.spm.tools.cat.estwrite.output.ROImenu.atlases.mori = 1;
matlabbatch{1}.spm.tools.cat.estwrite.output.ROImenu.atlases.anatomy3 = 1;
matlabbatch{1}.spm.tools.cat.estwrite.output.ROImenu.atlases.julichbrain = 1;
matlabbatch{1}.spm.tools.cat.estwrite.output.ROImenu.atlases.Schaefer2018_100Parcels_17Networks_order = 1;
matlabbatch{1}.spm.tools.cat.estwrite.output.ROImenu.atlases.Schaefer2018_200Parcels_17Networks_order = 1;
matlabbatch{1}.spm.tools.cat.estwrite.output.ROImenu.atlases.Schaefer2018_400Parcels_17Networks_order = 1;
matlabbatch{1}.spm.tools.cat.estwrite.output.ROImenu.atlases.Schaefer2018_600Parcels_17Networks_order = 1;
matlabbatch{1}.spm.tools.cat.estwrite.output.ROImenu.atlases.ownatlas = {''};

% % { name fileid GUIlevel use } - in development
% matlabbatch{1}.spm.tools.cat.estwrite.extopts.satlas      = { ...
%   'Desikan'                fullfile(spm('dir'),'toolbox','cat12','atlases_surfaces','lh.aparc_a2009s.freesurfer.annot')                        1   1;
%   'Destrieux'              fullfile(spm('dir'),'toolbox','cat12','atlases_surfaces','lh.aparc_DK40.freesurfer.annot')                          1   1;
%   'HCP'                    fullfile(spm('dir'),'toolbox','cat12','atlases_surfaces','lh.aparc_HCP_MMP1.freesurfer.annot')                      1   1;
%   ... Schaefer atlases ...
%   'Schaefer2018_100P_17N'  fullfile(spm('dir'),'toolbox','cat12','atlases_surfaces','lh.Schaefer2018_100Parcels_17Networks_order.annot')       1   1;
%   'Schaefer2018_200P_17N'  fullfile(spm('dir'),'toolbox','cat12','atlases_surfaces','lh.Schaefer2018_200Parcels_17Networks_order.annot')       1   1;
%   'Schaefer2018_400P_17N'  fullfile(spm('dir'),'toolbox','cat12','atlases_surfaces','lh.Schaefer2018_400Parcels_17Networks_order.annot')       1   1;
%   'Schaefer2018_600P_17N'  fullfile(spm('dir'),'toolbox','cat12','atlases_surfaces','lh.Schaefer2018_600Parcels_17Networks_order.annot')       1   1;
% };


% Writing options (see cat_defaults for the description of parameters)
%   native    0/1     (none/yes)
%   warped    0/1     (none/yes)
%   mod       0/1/2/3 (none/affine+nonlinear/nonlinear only/both)
%   dartel    0/1/2/3 (none/rigid/affine/both)

% GM/WM/CSF/WMH
matlabbatch{1}.spm.tools.cat.estwrite.output.GM.native = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.GM.warped = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.GM.mod = 3;
matlabbatch{1}.spm.tools.cat.estwrite.output.GM.dartel = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.WM.native = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.WM.warped = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.WM.mod = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.WM.dartel = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.CSF.native = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.CSF.warped = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.CSF.mod = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.CSF.dartel = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.ct.native = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.ct.warped = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.ct.dartel = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.pp.native = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.pp.warped = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.pp.dartel = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.WMH.native = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.WMH.warped = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.WMH.mod = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.WMH.dartel = 0;

% stroke lesion tissue maps (only for opt.extopts.SLC>0) - in development
matlabbatch{1}.spm.tools.cat.estwrite.output.SL.native = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.SL.warped = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.SL.mod = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.SL.dartel = 0;

% Tissue classes 4-6 to create own TPMs
matlabbatch{1}.spm.tools.cat.estwrite.output.TPMC.native = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.TPMC.warped = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.TPMC.mod = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.TPMC.dartel = 0;

% atlas maps (for evaluation)
matlabbatch{1}.spm.tools.cat.estwrite.output.atlas.native = 1;
matlabbatch{1}.spm.tools.cat.estwrite.output.atlas.warped = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.atlas.dartel = 0;

% label
% background=0, CSF=1, GM=2, WM=3, WMH=4 (if opt.extopts.WMHC==3), SL=1.5 (if opt.extopts.SLC>0)matlabbatch{1}.spm.tools.cat.estwrite.output.label.native = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.label.native = 1;
matlabbatch{1}.spm.tools.cat.estwrite.output.label.warped = 1;
matlabbatch{1}.spm.tools.cat.estwrite.output.label.dartel = 0;

% bias and noise corrected, global intensity normalized
matlabbatch{1}.spm.tools.cat.estwrite.output.bias.native = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.bias.warped = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.bias.dartel = 0;

% bias and noise corrected, (locally - if LAS>0) intensity normalized
matlabbatch{1}.spm.tools.cat.estwrite.output.las.native = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.las.warped = 0;
matlabbatch{1}.spm.tools.cat.estwrite.output.las.dartel = 0;

% jacobian determinant 0/1 (none/yes)
matlabbatch{1}.spm.tools.cat.estwrite.output.jacobianwarped = 1;

% deformations, order is [forward inverse]
matlabbatch{1}.spm.tools.cat.estwrite.output.warps = [1 1];

% deformation matrices (affine and rigid)
matlabbatch{1}.spm.tools.cat.estwrite.output.rmat = 1;
