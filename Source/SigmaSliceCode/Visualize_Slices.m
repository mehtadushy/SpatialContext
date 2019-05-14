%Visualize neighbourhoods
load('.\SceneCategoriesData\scenes_names.mat');
image_dir = 'D:\HLCV15\scene_categories\';
data_dir = '.\SceneCategoriesData';

%%
%Let us visualize the interest points and stuff for some images
%pick a few images from category i
category = 7;
idx = find(data.labels == category);
idx_idx = randsample(length(idx), 1);
idx = idx(idx_idx);

for i = 1:length(idx)
    iname = fullfile(image_dir, data.filenames{idx(i)});
    img = sp_load_image(iname);
    figure(i);
    imshow(img);
    visualizeNeighbourhood([], interest_points{idx(i)}, 30 , visData{idx(i)});
end

%%
category = 6;
idx = find(data.labels == category);
idx_idx = randsample(length(idx), 1);
idx = idx(idx_idx);

    iname = fullfile(image_dir, data.filenames{idx(1)});
    img = sp_load_image(iname);
 img = single(img);
figure(1); imshow(img); hold on; vl_plotframe(vl_covdet(img, 'Method', 'DoG'));
figure(2); imshow(img); hold on; vl_plotframe(vl_covdet(img, 'Method', 'HessianLaplace'));
figure(3); imshow(img); hold on; vl_plotframe(vl_covdet(img, 'Method', 'HarrisLaplace'));