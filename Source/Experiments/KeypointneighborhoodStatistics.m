% Compile Keypoint Neighbourhood Pooling Size statistics

image_dir = '~/DirectionalPooling/directionalpooling/Data/SceneCategoriesDataset_Images';
data_dir = '/scratch/common/pool0/dmetha/SceneCategoriesDataset_Data/SoftAssign_2D';
data_dir2 = '/scratch/common/pool0/dmetha/SceneCategoriesDataset_Data';
load(fullfile(data_dir2, 'scenes_names.mat'));

%Pick some images and create a scale and feature size vector

featureSuffix = '_keypoint_sift.mat';
feature2Suffix = '_6_10_neighborhood_histsize100.mat';

H = zeros(20, 20);
X = zeros(20, 20);
scales = [];
neigh_sizes = [];
for i = 1:1000

    imageFName = data.filenames{i};
    [dirN, base] = fileparts(imageFName);
    baseFName = fullfile(dirN, base);
    inFName = fullfile(data_dir, sprintf('%s%s', baseFName, featureSuffix));
    dat = load(inFName);
    scales = [scales dat.features.vl_frame(3,:)];
    inFName = fullfile(data_dir, sprintf('%s%s', baseFName, feature2Suffix));
    dat = load(inFName);
    neigh_sizes = [neigh_sizes dat.features.neighSize(1:length(dat.features.x))];
end
%%
    [h, x] = hist3([log(scales)' neigh_sizes'], [25 20]);
    
%%

%figure(2);
%bar3(h)
%%
%figure(3);
HeatMap(log(h'), 'RowLabels', x{2}, 'ColumnLabels', x{1}, 'Colormap', redbluecmap, 'Symmetric', false)
%axis([0.4 3.5 0 0.3])