%Compute interest points for the scenes categories dataset
image_dir = '..\..\Data\SceneCategoriesDataset_Images';
data_dir = '..\..\Data\SceneCategoriesDataset_Data';
load(fullfile(data_dir, 'scenes_names.mat'));

interest_points = ComputeInterestPoints(data.filenames, image_dir, data_dir, [], 100, 1);

%%
%Normalize interest points wrt the size of the largest scale
scales = zeros(length(interest_points),30);
norm_scales = zeros(length(interest_points),30);

for i = 1:length(interest_points)
    scales(i, 1: min( length(interest_points{i}.scale),size(scales,2))) = interest_points{i}.scale(1: min( length(interest_points{i}.scale),size(scales,2)));
    norm_scales(i,:) = scales(i,:)/scales(i,1);
end
%%
classes = unique(data.labels);
%Standard deviation of the scales (normalized to the largest scale)
scale_sd = zeros(length(classes),30);

for i = 1:length(classes)
    class_scales = norm_scales(data.labels == i,:);
    mean_scales = mean(class_scales);
    var_scales = var(class_scales);
    scale_sd(i,:) = sqrt(var_scales)./ (mean_scales + 1e-15);
end
all_class_sd = sqrt(var(norm_scales)) ./ (mean_scales + 1e-15);
figure(1); plot(scale_sd(:,1:10)');
title('(Standard Deviation / Mean) of Scales of Interest Points \n Ordered in Decreasing Order of Scale per Class');
xlabel('Interest Points In Decreasing order of Scale (normalized by the largest scale per image)');
ylabel('(standard deviation of scale within class) / (mean of scale within class)');
figure(2); plot(all_class_sd(1:10));
%%