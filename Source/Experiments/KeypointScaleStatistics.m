%% Compute Keypoint Scale Statistics

data_dir = '/scratch/common/pool0/dmetha/SceneCategoriesDataset_Data';
load(fullfile(data_dir, 'scenes_names.mat'));

num_images_per_category = 100;
num_categories = length(unique(data.labels));
featureSuffix = '_keypoint_sift.mat';
H = zeros(num_categories, 20);
X = zeros(num_categories, 20);
parfor i = 1:num_categories
    category_imgs = data.filenames(data.labels == i);
    scales = [];
    for j = 1:num_images_per_category
        imageFName = category_imgs{j};
        [dirN, base] = fileparts(imageFName);
        baseFName = fullfile(dirN, base);
        inFName = fullfile(data_dir, sprintf('%s%s', baseFName, featureSuffix))
        dat = load(inFName);
        scales = [scales dat.features.vl_frame(3,:)];
    end
    [h, x] = hist(log(scales), 20);
    H(i,:) = h(:)';
    X(i,:) = x(:)';
end
%%
H = H./repmat(sum(H,2),1, 20);
figure(2);
bar(X',H')
axis([0.4 3.5 0 0.3])