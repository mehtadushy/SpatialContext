%% Compute Keypoint Avg Count Between DoG Keypoints and SIFT Keypoints 

% This has SIFT Keypoints at the moment
data_dir = '/scratch/common/pool0/dmetha/SceneCategoriesDataset_Data/AxisAlignedKeypointSIFT';
load(fullfile(data_dir,'..', 'scenes_names.mat'));

keypt_count = 0;
featureSuffix = '_keypoint_sift.mat';
for j = 1:1000
        imageFName = data.filenames{j};
        [dirN, base] = fileparts(imageFName);
        baseFName = fullfile(dirN, base);
        inFName = fullfile(data_dir, sprintf('%s%s', baseFName, featureSuffix));
        dat = load(inFName);
        keypt_count= keypt_count + size(dat.features.vl_frame,2);
end
%%

